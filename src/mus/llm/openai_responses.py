"""OpenAI via the Responses API (``/v1/responses``).

A sibling of ``OpenAILLM`` rather than a replacement: that one speaks Chat
Completions, which OpenAI-compatible gateways (OpenRouter and friends) and a
few OpenAI models (audio / search previews) still need. This one is for
OpenAI itself, and is the only way to reach the Responses-only models.

History is always sent in full with ``store=False``: mus owns the conversation
(and DBOS replays it), so nothing may depend on OpenAI keeping state for a
``previous_response_id``.
"""

import logging
import typing as t

from .types import (
    LLM,
    Delta,
    DeltaToolInputUpdate,
    ToolUse,
    ToolResult,
    File,
    Query,
    Assistant,
    CachePoint,
    LLMClientStreamArgs,
    FunctionSchemaNoAnnotations,
    DeltaText,
    DeltaToolUse,
    DeltaToolResult,
    Usage,
    DeltaHistory,
    DeltaStreamReset,
    StopReason,
    StopReasonKind,
    normalize_stop_reason,
)
from .exceptions import (
    LLMException,
    LLMRateLimitException,
    LLMConnectionException,
    LLMBadRequestException,
    LLMContextLengthExceededException,
    LLMServerException,
    LLMToolParseException,
    is_context_length_error,
)
from .openai import _map_openai_exception, file_to_image, _TOOL_CHOICE

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputContentParam,
    ResponseInputItemParam,
    ResponseInputImageParam,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.shared_params import Reasoning
from openai._types import Omit
import json
from json_repair import repair_json


OMIT = Omit()

PROVIDER = "openai"

# Delta.metadata key for the reasoning items that preceded a tool call. See
# ``_reasoning_metadata``.
REASONING_METADATA_KEY = "openai_reasoning"

logger = logging.getLogger(__name__)


def func_to_tool(func: FunctionSchemaNoAnnotations) -> FunctionToolParam:
    return {
        "type": "function",
        "name": func["name"],
        "description": func["description"],
        "parameters": func["schema"],
        # mus's schemas aren't written for strict mode (every property
        # required, additionalProperties false), so opting in would 400.
        "strict": False,
    }


def functions_for_llm(
    functions: t.Sequence[FunctionSchemaNoAnnotations],
) -> t.List[FunctionToolParam]:
    return [func_to_tool(func) for func in (functions or [])]


def file_to_image_part(file: File) -> ResponseInputImageParam:
    return {"type": "input_image", "image_url": file_to_image(file), "detail": "auto"}


def parse_content(query: t.Union[str, File]) -> ResponseInputContentParam:
    if isinstance(query, str):
        return {"type": "input_text", "text": query}
    elif isinstance(query, File):
        return file_to_image_part(query)
    else:
        raise ValueError(f"Invalid query type: {type(query)}")


def query_to_input(query: Query) -> t.List[ResponseInputItemParam]:
    items: t.List[ResponseInputItemParam] = []
    for q in query.val:
        if isinstance(q, CachePoint):
            # OpenAI caches automatically; manual cache points don't apply.
            continue
        if isinstance(q, Assistant):
            items.append({"role": "assistant", "content": q.val})
        elif isinstance(q, str):
            items.append({"role": "user", "content": q})
        else:
            items.append({"role": "user", "content": [parse_content(q)]})
    return items


def split_tool_result(
    tool_result: ToolResult,
) -> t.Tuple[t.List[str], t.List[ResponseInputImageParam]]:
    """Split a tool result into text strings and image content parts.

    ``function_call_output`` takes a string, so images are returned separately
    to be sent in a following user message.
    """
    val = tool_result.content.val
    items = val if isinstance(val, list) else [val]
    texts: t.List[str] = []
    images: t.List[ResponseInputImageParam] = []
    for c in items:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, File):
            images.append(file_to_image_part(c))
        else:
            raise ValueError(f"Invalid tool result content type: {type(c)}")
    return texts, images


def _reasoning_metadata(
    model: str, items: t.List[ResponseReasoningItem]
) -> t.Optional[t.Dict[str, t.Any]]:
    """Provider-replay metadata for the reasoning that led up to a tool call.

    Reasoning models expect the reasoning items that preceded a function call
    to be sent back with it; without them the follow-up turn reasons from
    scratch. With ``store=False`` an item can only be sent back by its
    ``encrypted_content``, so items without one are useless and dropped.

    Carried on the DeltaToolUse rather than on a reasoning DeltaText: tool uses
    are never merged or pruned from history, whereas an empty text delta --
    which a reasoning item with no summary would be -- is.

    The model is recorded because encrypted reasoning is only valid for the
    model that produced it; a history replayed against another one leaves it
    out.
    """
    replayable = [
        item.model_dump(
            include={"id", "type", "summary", "encrypted_content"}, exclude_none=True
        )
        for item in items
        if item.encrypted_content
    ]
    if not replayable:
        return None
    return {REASONING_METADATA_KEY: {"model": model, "items": replayable}}


def _replay_reasoning(delta: Delta, model: t.Optional[str]) -> t.List[t.Any]:
    meta = (delta.metadata or {}).get(REASONING_METADATA_KEY)
    if not meta or meta.get("model") != model:
        return []
    return list(meta.get("items", []))


def deltas_to_input(
    deltas: t.Iterable[t.Union[Query, Delta]],
    model: t.Optional[str] = None,
) -> t.List[ResponseInputItemParam]:
    items: t.List[ResponseInputItemParam] = []
    # Images from tool results, held until the run of function_call_outputs
    # ends so the outputs of one turn stay together.
    pending_images: t.List[ResponseInputItemParam] = []
    # The assistant message streamed text is being gathered into. Within a
    # turn the Bot hands history on unmerged, one delta per streamed chunk;
    # sent as-is, each chunk would become an assistant message of its own.
    text_run: t.Optional[EasyInputMessageParam] = None

    def flush_images():
        nonlocal text_run
        if pending_images:
            text_run = None
        items.extend(pending_images)
        pending_images.clear()

    for delta in deltas:
        if isinstance(delta, Delta) and isinstance(delta.content, DeltaToolResult):
            tool_result = delta.content.data
            texts, images = split_tool_result(tool_result)
            if texts:
                output = "\n".join(texts)
            elif images:
                output = "[Tool returned image(s); see the following user message.]"
            else:
                output = ""
            text_run = None
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result.id,
                    "output": output,
                }
            )
            if images:
                pending_images.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Image output from tool call {tool_result.id}:",
                            },
                            *images,
                        ],
                    }
                )
            continue

        flush_images()
        if isinstance(delta, Delta):
            if isinstance(delta.content, DeltaText):
                # Reasoning is replayed from the tool-use metadata, not from
                # its summary text -- so it doesn't break a run of text either.
                if delta.content.subtype == "text" and delta.content.data:
                    if text_run is None:
                        text_run = {"role": "assistant", "content": delta.content.data}
                        items.append(text_run)
                    else:
                        text_run["content"] = (
                            f"{text_run['content']}{delta.content.data}"
                        )
            elif isinstance(delta.content, DeltaToolUse):
                text_run = None
                items.extend(_replay_reasoning(delta, model))
                items.append(
                    {
                        # No item id: sending a function_call's own ``fc_`` id
                        # makes OpenAI demand the reasoning item it came with,
                        # which a history that crossed models won't have.
                        "type": "function_call",
                        "call_id": delta.content.data.id,
                        "name": delta.content.data.name,
                        "arguments": json.dumps(delta.content.data.input),
                    }
                )
            elif isinstance(
                delta.content,
                (
                    # Tool results are handled above, before the image flush.
                    DeltaToolResult,
                    DeltaToolInputUpdate,
                    DeltaHistory,
                    DeltaStreamReset,
                ),
            ):
                pass
            else:
                t.assert_never(delta.content)
        else:
            text_run = None
            items.extend(query_to_input(delta))
    flush_images()
    return items


_STOP_REASONS: t.Mapping[str, StopReasonKind] = {
    # Responses has no finish_reason; the response status (or, when incomplete,
    # incomplete_details.reason) is the closest thing. A completed response
    # that called tools is promoted to tool_use by ``_map_response_stop_reason``.
    "completed": "end_turn",
    "max_output_tokens": "max_tokens",
    "content_filter": "content_filter",
    "refusal": "content_filter",
    "failed": "error",
    "cancelled": "error",
}


def _map_response_stop_reason(
    status: t.Optional[str],
    incomplete_reason: t.Optional[str] = None,
    *,
    has_tool_calls: bool = False,
    refused: bool = False,
) -> t.Optional[StopReason]:
    raw = incomplete_reason if status == "incomplete" and incomplete_reason else status
    stop = normalize_stop_reason(raw, _STOP_REASONS, pending_tools=has_tool_calls)
    if stop is not None and stop.kind == "end_turn":
        if has_tool_calls:
            return StopReason(kind="tool_use", raw=stop.raw)
        if refused:
            # A refusal still completes normally, but it isn't the answer the
            # caller asked for -- the same call Anthropic's "refusal" makes.
            return StopReason(kind="content_filter", raw="refusal")
    return stop


def _map_error_code(code: t.Optional[str], message: str) -> LLMException:
    """Map an error reported *inside* the stream (an ``error`` event or a
    failed response) -- these carry a code, not an HTTP status."""
    if is_context_length_error(message, code=code):
        return LLMContextLengthExceededException(message, provider=PROVIDER)
    if code == "rate_limit_exceeded":
        return LLMRateLimitException(message, provider=PROVIDER)
    if code == "server_error":
        return LLMServerException(message, provider=PROVIDER)
    if code and code.startswith("invalid_"):
        return LLMBadRequestException(message, provider=PROVIDER)
    return LLMException(message, provider=PROVIDER)


def _usage(usage: t.Optional[ResponseUsage]) -> t.Optional[Usage]:
    if usage is None:
        return None
    return Usage(
        input_tokens=usage.input_tokens,
        # Already includes reasoning tokens.
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=(
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details
            else 0
        )
        or 0,
        cache_written_input_tokens=0,
    )


def _parse_tool_call(call: ResponseFunctionToolCall) -> ToolUse:
    parsed_input = repair_json(call.arguments, return_objects=True)
    if not isinstance(parsed_input, dict):
        raise LLMToolParseException(
            f"Model returned malformed tool JSON for {call.name}: {call.arguments}",
            provider=PROVIDER,
        )
    return ToolUse(id=call.call_id, name=call.name, input=parsed_input)


class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]
    reasoning: Reasoning


STREAM_ARGS = StreamArgs
MODEL_TYPE = str


class OpenAIResponsesLLM(LLM[StreamArgs, MODEL_TYPE, openai.AsyncClient]):
    provider = PROVIDER

    def __init__(
        self,
        model: MODEL_TYPE,
        client: t.Optional[openai.AsyncClient] = None,
        *,
        include_reasoning: bool = True,
    ):
        """
        ``include_reasoning`` asks for encrypted reasoning so it can be sent
        back across tool calls. Models without reasoning accept the request
        and simply return none; set it to ``False`` to leave it out anyway.
        """
        if not client:
            client = openai.AsyncClient()
        self.client = client
        self.model = model
        self.include_reasoning = include_reasoning

    async def _create(self, **request: t.Any) -> t.Any:
        include = list(request.pop("include", None) or [])
        if self.include_reasoning:
            include.append("reasoning.encrypted_content")
        try:
            return await self.client.responses.create(
                **request, include=include or OMIT
            )
        except openai.APIError as e:
            raise _map_openai_exception(e) from e

    async def stream(
        self, **kwargs: t.Unpack[LLMClientStreamArgs[StreamArgs, MODEL_TYPE]]
    ):
        input_items = deltas_to_input(kwargs.get("history", []), model=self.model)

        if functions := kwargs.get("functions", None):
            tools: t.Any = functions_for_llm(functions)
        else:
            tools = OMIT

        if stop_sequences := kwargs.get("stop_sequences", None):
            logger.warning(
                "The OpenAI Responses API does not support stop sequences; "
                "ignoring stop_sequences=%r",
                stop_sequences,
            )

        temperature = kwargs.get("temperature", None)
        top_p = kwargs.get("top_p", None)
        stream = not kwargs.get("no_stream", False)
        extra_kwargs: t.Dict[str, t.Any] = dict(kwargs.get("kwargs", None) or {})

        response = await self._create(
            model=self.model,
            input=input_items,
            instructions=kwargs.get("prompt", None) or OMIT,
            tools=tools,
            tool_choice=_TOOL_CHOICE.get(
                kwargs.get("function_choice", None) or "", OMIT
            ),
            max_output_tokens=kwargs.get("max_tokens", None) or OMIT,
            temperature=temperature if temperature is not None else OMIT,
            top_p=top_p if top_p is not None else OMIT,
            store=False,
            stream=stream,
            **extra_kwargs,
        )

        if stream:
            async for delta in self._handle_stream(response):
                yield delta
        else:
            for delta in self._handle_response(response):
                yield delta

    def _finish(
        self,
        response: Response,
        calls: t.List[t.Tuple[ResponseFunctionToolCall, t.List[ResponseReasoningItem]]],
        *,
        calls_started: bool,
        refused: bool,
    ) -> t.Iterator[Delta]:
        if response.status == "failed" and response.error:
            raise _map_error_code(response.error.code, response.error.message)

        stop = _map_response_stop_reason(
            response.status,
            response.incomplete_details.reason if response.incomplete_details else None,
            has_tool_calls=calls_started,
            refused=refused,
        )
        # Tool calls are only handed on when the stop was planned. On e.g. a
        # max_output_tokens stop the arguments may be truncated, and invoking a
        # tool with fabricated arguments is worse than not invoking it at all.
        if stop is None or stop.is_planned:
            for call, reasoning in calls:
                yield Delta(
                    content=DeltaToolUse(data=_parse_tool_call(call)),
                    metadata=_reasoning_metadata(self.model, reasoning),
                )
        yield Delta(
            content=DeltaText(data=""),
            stop_reason=stop,
            usage=_usage(response.usage),
        )

    async def _handle_stream(self, response: t.Any) -> t.AsyncIterator[Delta]:
        # Function calls in arrival order, each with the reasoning items that
        # came before it. Held back until the response ends: only then is it
        # known whether the calls are complete.
        calls: t.List[
            t.Tuple[ResponseFunctionToolCall, t.List[ResponseReasoningItem]]
        ] = []
        reasoning: t.List[ResponseReasoningItem] = []
        # item_id -> function call, for naming argument fragments.
        started: t.Dict[str, ResponseFunctionToolCall] = {}
        refused = False
        try:
            async for event in response:
                if event.type == "response.output_text.delta":
                    yield Delta(content=DeltaText(data=event.delta))
                elif event.type in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    yield Delta(
                        content=DeltaText(data=event.delta, subtype="reasoning")
                    )
                elif event.type == "response.refusal.delta":
                    refused = True
                elif event.type == "response.output_item.added":
                    if event.item.type == "function_call" and event.item.id:
                        started[event.item.id] = event.item
                elif event.type == "response.function_call_arguments.delta":
                    if call := started.get(event.item_id):
                        yield Delta(
                            content=DeltaToolInputUpdate(
                                name=call.name, id=call.call_id, data=event.delta
                            )
                        )
                elif event.type == "response.output_item.done":
                    if event.item.type == "function_call":
                        calls.append((event.item, reasoning))
                        reasoning = []
                    elif event.item.type == "reasoning":
                        reasoning.append(event.item)
                elif event.type in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    for delta in self._finish(
                        event.response,
                        calls,
                        calls_started=bool(started or calls),
                        refused=refused,
                    ):
                        yield delta
                    return
                elif event.type == "error":
                    # The SDK raises these itself (below); kept for streams
                    # that hand the event through.
                    raise _map_error_code(event.code, event.message)
        except openai.APIStatusError as e:
            raise _map_openai_exception(e) from e
        except openai.APIError as e:
            # An ``error`` event mid-stream: the SDK raises it as a bare
            # APIError, with the event's code but no HTTP status -- which the
            # status-based mapping would leave as a generic LLMException. This
            # is how OpenAI reports e.g. a context-window overflow when
            # streaming.
            raise _map_error_code(e.code, e.message) from e

        # Without a terminal event, any tool calls can't be known complete.
        raise LLMConnectionException(
            "Response stream ended before the response completed",
            provider=PROVIDER,
        )

    def _handle_response(self, response: Response) -> t.Iterator[Delta]:
        calls: t.List[
            t.Tuple[ResponseFunctionToolCall, t.List[ResponseReasoningItem]]
        ] = []
        reasoning: t.List[ResponseReasoningItem] = []
        refused = False
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        if part.text:
                            yield Delta(content=DeltaText(data=part.text))
                    elif part.type == "refusal":
                        refused = True
            elif item.type == "reasoning":
                reasoning.append(item)
                for summary in item.summary:
                    if summary.text:
                        yield Delta(
                            content=DeltaText(data=summary.text, subtype="reasoning")
                        )
            elif item.type == "function_call":
                calls.append((item, reasoning))
                reasoning = []
        yield from self._finish(
            response, calls, calls_started=bool(calls), refused=refused
        )
