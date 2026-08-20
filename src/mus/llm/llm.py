import asyncio
import json
from json_repair import repair_json
import logging
import random
import typing as t
import uuid
from textwrap import dedent
import sys
from dataclasses import replace

from .types import (
    Delta,
    LLM,
    LLMClientStreamArgs,
    QueryType,
    System,
    LLMDecoratedFunctionType,
    LLMDecoratedFunctionReturnType,
    Query,
    LLMPromptFunctionArgs,
    ToolCallableType,
    ToolResult,
    RetryPolicy,
    STREAM_EXTRA_ARGS,
    MODEL_TYPE,
    History,
    QueryStreamArgs,
    Usage,
    CLIENT_TYPE,
    Assistant,
    CacheOptions,
    FunctionSchemaNoAnnotations,
    DeltaText,
    DeltaToolUse,
    DeltaToolResult,
    DeltaHistory,
    DeltaStreamReset,
    DeltaToolInputUpdate,
    ensure_tool_value,
    FallbackToolCallableType,
    StopReason,
    StopRecovery,
    StopRecoveryContinue,
    StopRecoveryReset,
)
from .exceptions import (
    LLMException,
    LLMRateLimitException,
    LLMServerException,
    LLMTimeoutException,
    LLMConnectionException,
    LLMModelException,
    LLMStoppedException,
)
from ..functions import (
    to_schema,
    schema_to_example,
    parse_tools,
    ToolCallable,
    verify_schema_inputs,
)
from ..types import FillableType

from ..exceptions import ToolNotFoundError

logger = logging.getLogger(__name__)

_TRANSIENT_EXCEPTIONS = (
    LLMRateLimitException,
    LLMServerException,
    LLMTimeoutException,
    LLMConnectionException,
    LLMModelException,
)


def _compute_backoff(
    attempt: int,
    policy: RetryPolicy,
    retry_after: t.Optional[float] = None,
) -> float:
    """Compute sleep duration for a retry attempt (0-indexed)."""
    if retry_after is not None and retry_after > 0:
        return retry_after
    backoff = min(
        policy.initial_backoff * (policy.backoff_multiplier**attempt),
        policy.max_backoff,
    )
    backoff *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
    return max(0.0, backoff)


def merge_history(history: History) -> History:
    merged = []
    for msg in history:
        if (
            merged
            and isinstance(msg, Delta)
            and isinstance(msg.content, DeltaText)
            and msg.content.subtype == "text"
        ):
            if (
                isinstance(merged[-1], Delta)
                and isinstance(merged[-1].content, DeltaText)
                and merged[-1].content.subtype == "text"
            ):
                # Create new Delta and DeltaText instead of mutating in place
                # Using replace() ensures all fields are properly copied
                merged[-1] = replace(
                    merged[-1],
                    content=replace(
                        merged[-1].content,
                        data=merged[-1].content.data + msg.content.data,
                    ),
                )
            else:
                merged.append(msg)

        else:
            merged.append(msg)

    # prune empty text
    merged = [
        m
        for m in merged
        if not (
            isinstance(m, Delta)
            and isinstance(m.content, DeltaText)
            and not m.content.data.strip()
        )
    ]

    return merged


class IterableResult:
    def __init__(self, iterable: t.AsyncIterable[Delta]):
        self.iterable = iterable
        self.history: History = []
        self.has_iterated = False
        self.total = ""
        self.usage = Usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_written_input_tokens=0,
        )
        # Why the turn ended. Unplanned stops raise, so anything observable here
        # is planned -- it distinguishes end_turn from stop_sequence.
        self.stop_reason: t.Optional[StopReason] = None
        self._stream_text: t.Dict[str, str] = {}

    def _rollback_stream(self, stream_id: str) -> None:
        """Remove text and history contributions from a specific stream_id."""
        if text := self._stream_text.pop(stream_id, ""):
            if self.total.endswith(text):
                self.total = self.total[: -len(text)]
            else:
                self.total = self.total.replace(text, "", 1)
        self.history = [
            h
            for h in self.history
            if not (isinstance(h, Delta) and h.stream_id == stream_id)
        ]

    def _track_text(self, stream_id: t.Optional[str], text: str) -> None:
        if stream_id is not None:
            self._stream_text[stream_id] = self._stream_text.get(stream_id, "") + text
        self.total += text

    async def __aiter__(self):
        async for msg in self.iterable:
            if isinstance(msg.content, DeltaStreamReset):
                self._rollback_stream(msg.content.stream_id)
                yield msg
                continue
            if isinstance(msg.content, DeltaText) and msg.content.subtype == "text":
                self._track_text(msg.stream_id, msg.content.data)
            elif isinstance(msg.content, DeltaToolUse):
                self._track_text(
                    msg.stream_id, f"Running tool: {msg.content.data.name}"
                )
            elif isinstance(msg.content, DeltaToolResult):
                self._track_text(msg.stream_id, "Tool applied")
            # Last non-tool_use reason wins, rather than simply the last one. A
            # DeltaToolUse makes Bot.query recurse *inline*, so for providers
            # that report usage after finish_reason the outer turn's trailing
            # "tool_use" arrives after the inner turn's real terminal reason.
            if msg.stop_reason is not None and msg.stop_reason.kind != "tool_use":
                self.stop_reason = msg.stop_reason
            if msg.usage:
                self.usage = Usage(
                    input_tokens=self.usage.input_tokens + msg.usage.input_tokens,
                    output_tokens=self.usage.output_tokens + msg.usage.output_tokens,
                    cache_read_input_tokens=self.usage.cache_read_input_tokens
                    + msg.usage.cache_read_input_tokens,
                    cache_written_input_tokens=self.usage.cache_written_input_tokens
                    + msg.usage.cache_written_input_tokens,
                )
            if isinstance(msg.content, DeltaHistory):
                # TODO: Merge deltas here
                self.history.extend(merge_history(msg.content.data))
            else:
                yield msg
        self.has_iterated = True

    async def string(self):
        if not self.has_iterated:
            async for a in self:
                pass
        return self.total


class TransformDeltaHook(t.Protocol):
    async def __call__(self, delta: Delta) -> Delta: ...


class TransformHistoryHook(t.Protocol):
    async def __call__(self, history: History) -> History: ...


class ErrorRecoveryHook(t.Protocol):
    """Called when an LLM call fails *pre-stream* (no delta yielded yet).

    Given the history that failed, the error, and a zero-based retry attempt,
    return a *modified* history to re-issue with, or ``None`` to give up (the
    error propagates). The hook fully owns the retry history — mus re-issues
    exactly what it returns. The motivating case is context-window overflow
    (return a smaller history), but it is general: inspect ``error`` and recover
    only what you know how to.
    """

    async def __call__(
        self, history: History, error: LLMException, attempt: int
    ) -> t.Optional[History]: ...


class StopRecoveryHook(t.Protocol):
    """Called when the model stops for a reason the caller did not ask for.

    Distinct from ``ErrorRecoveryHook``, which handles calls that fail *before*
    the stream starts. Here the request succeeded and deltas have already been
    yielded -- the model simply stopped early, typically on ``max_tokens``.

    Return :class:`StopRecoveryContinue` to keep the partial output and generate
    onward from it, :class:`StopRecoveryReset` to discard it and re-issue the
    turn, or ``None`` to give up (``LLMStoppedException`` propagates, which is
    also what happens when no hook is set).

    Recovering here is what keeps a long tool-calling flow alive: a truncation
    three calls deep would otherwise unwind the whole turn. The exception
    carries what you need to decide -- ``stop_reason``, ``partial_text``,
    ``history`` and ``pending_tool_call``.

    A stop with ``pending_tool_call`` cannot be continued: the turn holds a
    half-emitted tool block that providers reject on the next request. Returning
    ``StopRecoveryContinue`` for one is coerced to a reset, with a warning.
    """

    async def __call__(
        self, error: "LLMStoppedException", attempt: int
    ) -> t.Optional[StopRecovery]: ...


class _LLMInitAndQuerySharedKwargs(QueryStreamArgs, total=False):
    functions: t.Optional[t.Sequence[ToolCallableType | ToolCallable]]
    function_choice: t.Optional[t.Literal["auto", "any"]]
    fallback_function: t.Optional[FallbackToolCallableType]
    no_stream: t.Optional[bool]
    cache: t.Optional[CacheOptions]
    transform_delta_hook: t.Optional[TransformDeltaHook]
    transform_history_hook: t.Optional[TransformHistoryHook]
    error_recovery_hook: t.Optional[ErrorRecoveryHook]
    stop_recovery_hook: t.Optional[StopRecoveryHook]


class _LLMCallArgs(_LLMInitAndQuerySharedKwargs, total=False):
    previous: t.Optional[IterableResult]


QueryOrSystem = t.Union[QueryType, System]


def get_exception_depth():
    """Get the depth of the current exception's traceback"""
    _, _, exc_traceback = sys.exc_info()
    if exc_traceback is None:
        return 0

    depth = 0
    tb = exc_traceback
    while tb is not None:
        depth += 1
        tb = tb.tb_next
    return depth


async def invoke_function(
    func_name: str, input: t.Mapping[str, t.Any], func_map: dict[str, ToolCallable]
):
    try:
        tool_callable = func_map[func_name]
    except KeyError:
        raise ToolNotFoundError(
            f"Tool {func_name} not found (have {', '.join(list(func_map.keys()))})"
        ) from None
    # Validate input against schema
    try:
        input = verify_schema_inputs(tool_callable.schema, input)
    except ValueError as e:
        return json.dumps(
            {
                "error": f"{str(e)}",
            }
        )

    try:
        result = await tool_callable.function(**input)
    except TypeError as e:
        depth = get_exception_depth()
        if depth == 1 and type(e).__name__ == "TypeError":
            return json.dumps(
                {
                    "error": f"Tool {func_name} was called with incorrect arguments: {input}. Please check the function signature and the input provided.",
                }
            )
        else:
            raise e from e

    return result


class Bot(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE]):
    def __init__(
        self,
        prompt: t.Optional[str] = None,
        *,
        model: LLM[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE],
        client_kwargs: t.Optional[STREAM_EXTRA_ARGS] = None,
        retry_policy: t.Optional[RetryPolicy] = None,
        **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs],
    ) -> None:
        self.client = model
        self.prompt = prompt
        self.client_kwargs = client_kwargs
        self.retry_policy = retry_policy or RetryPolicy()
        self.default_args = kwargs

    async def query(
        self,
        query: t.Optional[QueryOrSystem] = None,
        /,
        *,
        history: History = [],
        **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs],
    ) -> t.AsyncGenerator[Delta, None]:
        kwargs = {**self.default_args, **kwargs}
        functions = kwargs.get("functions") or []
        tools = parse_tools(functions)
        transform_delta_hook = kwargs.get("transform_delta_hook", None)
        transform_history_hook = kwargs.get("transform_history_hook", None)
        error_recovery_hook = kwargs.get("error_recovery_hook", None)
        stop_recovery_hook = kwargs.get("stop_recovery_hook", None)
        policy = self.retry_policy

        function_schemas = [
            FunctionSchemaNoAnnotations(
                {
                    "description": tool.schema["description"],
                    "name": tool.schema["name"],
                    "schema": tool.schema["schema"],
                }
            )
            for tool in tools
        ]

        func_map = {tool.schema["name"]: tool for tool in tools}

        parsed_query: t.Optional[Query] = None
        prompt = self.prompt
        if query:
            if isinstance(query, System):
                prompt = query.val
                query = query.query

            parsed_query = Query.parse(query) if query else None

        dedented_prompt = dedent(prompt) if prompt else None

        if parsed_query:
            history = history + parsed_query.to_deltas()

        if parsed_query:
            if isinstance(last := parsed_query.val[-1], Assistant):
                # if the last part of the query is a prefill
                # assistant message, with echo true, we send that as the
                # first message, and then continue with the rest of the query
                # this is helpful when doing structured generation
                if last.echo:
                    yield Delta(content=DeltaText(data=last.val))

        pre_stream_history = list(history)

        if transform_history_hook:
            history = await transform_history_hook(history)

        stream_id = uuid.uuid4().hex

        stream_kwargs = LLMClientStreamArgs(
            prompt=dedented_prompt,
            history=history,
            kwargs=self.client_kwargs,
            function_choice=kwargs.get("function_choice", None),
            functions=function_schemas,
            no_stream=kwargs.get("no_stream", None),
            max_tokens=kwargs.get("max_tokens", None),
            top_k=kwargs.get("top_k", None),
            top_p=kwargs.get("top_p", None),
            stop_sequences=kwargs.get("stop_sequences", None),
            temperature=kwargs.get("temperature", None),
            cache=kwargs.get("cache", None),
        )

        last_exception: t.Optional[Exception] = None
        # The terminal stop reason reported by the provider, plus the text this
        # turn produced -- both handed to LLMStoppedException so an unplanned
        # stop can be recovered from at the call site.
        last_stop_reason: t.Optional[StopReason] = None
        partial_text = ""

        recovery_attempt = 0
        stop_recovery_attempt = 0
        # Where a StopRecoveryReset rewinds to. Starts at the top of the turn and
        # advances past each completed tool call: tools run as their deltas
        # arrive, so rewinding past one would re-execute it.
        rollback_history = list(pre_stream_history)
        while True:
            yielded_any = False
            try:
                for attempt in range(policy.max_transport_retries + 1):
                    if attempt > 0:
                        retry_after = getattr(last_exception, "retry_after", None)
                        sleep_time = _compute_backoff(attempt - 1, policy, retry_after)
                        logger.warning(
                            "Retrying stream (attempt %d/%d) after %.1fs due to %s: %s",
                            attempt + 1,
                            policy.max_transport_retries + 1,
                            sleep_time,
                            type(last_exception).__name__,
                            last_exception,
                        )
                        yield Delta(
                            content=DeltaStreamReset(
                                stream_id=stream_id,
                                reason=str(last_exception),
                                attempt=attempt,
                            )
                        )
                        await asyncio.sleep(sleep_time)
                        history = list(pre_stream_history)
                        rollback_history = list(pre_stream_history)
                        stream_kwargs["history"] = history
                        stream_id = uuid.uuid4().hex
                        last_stop_reason = None
                        partial_text = ""

                    tool_id_to_uuid: dict[str, str] = {}
                    try:
                        async for msg in self.client.stream(**stream_kwargs):
                            # Assign tool_invocation_id for tool-related deltas
                            if isinstance(msg.content, DeltaToolInputUpdate):
                                provider_id = msg.content.id
                                if provider_id not in tool_id_to_uuid:
                                    tool_id_to_uuid[provider_id] = uuid.uuid4().hex
                                msg = replace(
                                    msg,
                                    stream_id=stream_id,
                                    tool_invocation_id=tool_id_to_uuid[provider_id],
                                )
                            elif isinstance(msg.content, DeltaToolUse):
                                provider_id = msg.content.data.id
                                if provider_id not in tool_id_to_uuid:
                                    tool_id_to_uuid[provider_id] = uuid.uuid4().hex
                                msg = replace(
                                    msg,
                                    stream_id=stream_id,
                                    tool_invocation_id=tool_id_to_uuid[provider_id],
                                )
                            else:
                                msg = replace(msg, stream_id=stream_id)

                            if transform_delta_hook:
                                msg = await transform_delta_hook(msg)

                            if msg.stop_reason is not None:
                                last_stop_reason = msg.stop_reason
                            if (
                                isinstance(msg.content, DeltaText)
                                and msg.content.subtype == "text"
                            ):
                                partial_text += msg.content.data

                            yield msg
                            yielded_any = True

                            history = history + [msg]
                            if isinstance(msg.content, DeltaToolUse):
                                print(
                                    "Invoking tool:",
                                    msg.content.data.name,
                                    "with input:",
                                    msg.content.data.input,
                                )
                                try:
                                    func_result = ensure_tool_value(
                                        await invoke_function(
                                            msg.content.data.name,
                                            msg.content.data.input,
                                            func_map,
                                        )
                                    )
                                except ToolNotFoundError as e:
                                    if fallback_function := kwargs.get(
                                        "fallback_function", None
                                    ):
                                        func_result = ensure_tool_value(
                                            await fallback_function(
                                                original_tool_name=msg.content.data.name,
                                                original_input=msg.content.data.input,
                                            )
                                        )
                                    else:
                                        raise e from e
                                fd = Delta(
                                    content=DeltaToolResult(
                                        ToolResult(id=msg.content.data.id, content=func_result)
                                    ),
                                    stream_id=stream_id,
                                    tool_invocation_id=tool_id_to_uuid[msg.content.data.id],
                                )
                                if transform_delta_hook:
                                    fd = await transform_delta_hook(fd)
                                yield fd
                                yielded_any = True
                                history.append(fd)
                                async for msg in self.query(history=history, **kwargs):
                                    if isinstance(msg.content, DeltaHistory):
                                        history.extend(msg.content.data[len(history) :])
                                    else:
                                        yield msg  # NOTE: we don't need to transform here, as the recursive call to self.query will have already done so
                                # This tool call has run; a reset must not rewind
                                # past it and fire its side effects a second time.
                                rollback_history = list(history)
                        # Stream completed successfully
                        break

                    except _TRANSIENT_EXCEPTIONS as e:
                        last_exception = e
                        if attempt >= policy.max_transport_retries:
                            raise
                        continue

                if last_stop_reason is not None and not last_stop_reason.is_planned:
                    # Raised inside the try so stop_recovery_hook gets a shot at
                    # it. A nested turn's history already holds everything the
                    # outer turn accumulated -- Bot.query passes its history down
                    # on recursion -- so this carries the whole turn.
                    raise LLMStoppedException(
                        f"Model stopped unexpectedly ({last_stop_reason.kind}); "
                        f"provider reported {last_stop_reason.raw!r}",
                        provider=self.client.provider,
                        stop_reason=last_stop_reason,
                        history=history,
                        partial_text=partial_text,
                    )
                break
            except LLMStoppedException as _stop:
                # Recovering in-loop is the point: a truncation several tool
                # calls deep would otherwise unwind the whole flow.
                if (
                    stop_recovery_hook is None
                    or stop_recovery_attempt >= policy.max_stop_recovery_attempts
                    # Already offered to the hook by a nested turn; don't ask
                    # again on the way up with a staler history.
                    or getattr(_stop, "_mus_recovery_declined", False)
                ):
                    _stop._mus_recovery_declined = True  # type: ignore[attr-defined]
                    raise
                _recovery = await stop_recovery_hook(_stop, stop_recovery_attempt)
                if _recovery is None:
                    _stop._mus_recovery_declined = True  # type: ignore[attr-defined]
                    raise
                stop_recovery_attempt += 1

                if isinstance(_recovery, StopRecoveryContinue) and _stop.pending_tool_call:
                    logger.warning(
                        "stop_recovery_hook returned StopRecoveryContinue for a stop with a "
                        "half-emitted tool call; coercing to StopRecoveryReset, since "
                        "providers reject the malformed tool block on the next request."
                    )
                    _recovery = StopRecoveryReset(
                        append=_recovery.append, history=_recovery.history
                    )

                if isinstance(_recovery, StopRecoveryReset):
                    _base = (
                        _recovery.history
                        if _recovery.history is not None
                        else rollback_history
                    )
                    yield Delta(
                        content=DeltaStreamReset(
                            stream_id=stream_id,
                            reason="stop_recovery",
                            attempt=stop_recovery_attempt,
                        )
                    )
                    partial_text = ""
                else:
                    _base = (
                        _recovery.history if _recovery.history is not None else history
                    )
                    if not any(
                        isinstance(h, Delta) and h.stream_id == stream_id for h in _base
                    ):
                        logger.warning(
                            "StopRecoveryContinue returned a history without this turn's "
                            "partial output; consumers will hold text the model cannot see."
                        )

                history = list(_base) + list(_recovery.append or [])
                pre_stream_history = list(history)
                rollback_history = list(history)
                stream_kwargs["history"] = history
                stream_id = uuid.uuid4().hex
                last_stop_reason = None
            except LLMException as _exc:
                # A pre-stream failure escaped the transport loop (non-transient,
                # or transient-exhausted). If a recovery hook is set, no delta was
                # yielded, and we're under the cap, ask it for a modified history
                # and re-issue. Otherwise propagate.
                if (
                    error_recovery_hook is None
                    or yielded_any
                    or recovery_attempt >= policy.max_recovery_attempts
                ):
                    raise
                _recovered = await error_recovery_hook(
                    stream_kwargs["history"], _exc, recovery_attempt
                )
                if _recovered is None:
                    raise
                recovery_attempt += 1
                yield Delta(
                    content=DeltaStreamReset(
                        stream_id=stream_id,
                        reason="error_recovery",
                        attempt=recovery_attempt,
                    )
                )
                history = list(_recovered)
                pre_stream_history = list(_recovered)
                stream_kwargs["history"] = history
                stream_id = uuid.uuid4().hex
                last_stop_reason = None
                partial_text = ""

        yield Delta(content=DeltaHistory(data=history))

    @t.overload
    def __call__(
        self, query: QueryOrSystem, /, **kwargs: t.Unpack[_LLMCallArgs]
    ) -> IterableResult: ...

    @t.overload
    def __call__(
        self,
        query: t.Callable[LLMPromptFunctionArgs, QueryOrSystem],
        /,
        **kwargs: t.Unpack[_LLMCallArgs],
    ) -> t.Callable[LLMPromptFunctionArgs, IterableResult]: ...

    def __call__(
        self,
        query: t.Union[QueryOrSystem, t.Callable[LLMPromptFunctionArgs, QueryOrSystem]],
        /,
        **kwargs: t.Unpack[_LLMCallArgs],
    ) -> t.Union[IterableResult, t.Callable[LLMPromptFunctionArgs, IterableResult]]:
        if callable(query):
            a = self.bot(query)
            return a
        else:
            previous = kwargs.pop("previous", None)
            _q = self.query(
                query,
                history=previous.history if previous is not None else [],
                **kwargs,
            )
            return IterableResult(_q)

    async def fill(
        self,
        query: QueryType,
        structure: t.Type[FillableType],
        strategy: t.Literal["tool_use", "prefill"] = "tool_use",
    ) -> FillableType:
        if strategy == "tool_use":
            as_tool = ToolCallable(
                function=structure,  # type: ignore
                schema=to_schema(structure),
            )
            async for msg in self.query(
                query, functions=[as_tool], function_choice="any", no_stream=True
            ):
                if isinstance(msg.content, DeltaToolUse):
                    input = msg.content.data.input
                    input = verify_schema_inputs(as_tool.schema, input)
                    return structure(**input)
            else:
                raise ValueError("No structured response found")
        elif strategy == "prefill":
            schema = to_schema(structure)
            first_prop = list(schema["schema"]["properties"].keys())[0]
            query = Query.parse(query)
            agumented_query = (
                query
                + f"""
                Return a JSON object with that follows this structure:
                <example>
                    {schema_to_example(schema)}
                </example>
                """
                + Assistant(
                    """\
                    ```
                    {
                        \""""
                    + first_prop
                    + '": ',
                    echo=True,
                )
            )
            result = (
                await self(agumented_query, stop_sequences=["```"]).string()
            ).strip()
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            input = repair_json(result, return_objects=True)
            if not isinstance(input, dict):
                raise ValueError(f"Failed to decode JSON: {result}")
            input = verify_schema_inputs(schema, input)
            return structure(**input)

    def fun(self, function: LLMDecoratedFunctionType[LLMDecoratedFunctionReturnType]):
        async def decorated_function(
            query: QueryType,
        ) -> LLMDecoratedFunctionReturnType:
            async for msg in self.query(
                query, functions=[function], function_choice="any", no_stream=True
            ):  # type: ignore
                if isinstance(msg.content, DeltaToolUse):
                    return await function(**(msg.content.data.input))
            else:
                raise ValueError("LLM did not invoke the function")

        return decorated_function

    def bot(
        self, function: t.Callable[LLMPromptFunctionArgs, QueryOrSystem]
    ) -> t.Callable[LLMPromptFunctionArgs, IterableResult]:
        def decorated(
            *args: LLMPromptFunctionArgs.args, **kwargs: LLMPromptFunctionArgs.kwargs
        ):
            prompt = function(*args, **kwargs)
            return self(prompt)

        return decorated
