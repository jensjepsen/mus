"""Durable execution for mus, on DBOS.

A mus run is normally ephemeral: if the process dies mid-conversation the turn is
lost, completed tool calls are forgotten, and a reconnecting client has nothing
to reattach to. Wrapping a bot here makes a run survive a crash or a deploy --
without re-billing completed provider calls or re-firing completed tools -- and
lets a client tail it from anywhere by workflow id.

    durable(bot)               wrap a Bot so a run is checkpointed and streamed
    read(workflow_id, offset)  tail a run's deltas from anywhere
    attach(workflow_id)        the same, as a mus IterableResult

Shape, and why it is this shape:

  * ``bot.query`` runs in the WORKFLOW body, never inside a step. A step nested
    in a step executes but is *not* checkpointed, so tools invoked from inside a
    turn-step would re-fire on replay -- silently.
  * Each provider call is a step that streams its deltas to the wrapper over
    an in-memory queue as they arrive, and returns them so a replay can be
    served from the checkpoint. mus therefore receives tokens as they are
    produced rather than when the turn ends, so its tagging, transform hooks
    and tool-result synthesis all apply in real time.
  * Each tool call is a step wrapping mus's own ``invoke``, so schema
    validation and the fallback function still apply.
  * Every delta mus yields is written to the public key from workflow scope,
    where writes are exactly-once.

Step arguments are never persisted (DBOS records outputs only), so
non-picklable values -- model objects, closures, tools defined anywhere at all
-- cross a step boundary freely.

Without ``dbos`` installed this module still imports, so mus has no hard
dependency on it -- but ``durable``, ``read`` and ``attach`` all raise. A
``durable()`` that quietly wasn't durable would be worse than an error: callers
would believe completed tools never re-fire, and only discover otherwise during
an incident. A bot that doesn't need durability doesn't need wrapping.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t
import uuid

from .llm.llm import Bot, IterableResult
from .llm.types import LLM, Delta, DeltaStreamReset, DeltaText, ToolUse, ToolValue

logger = logging.getLogger(__name__)

__all__ = [
    "durable",
    "read",
    "attach",
    "DurableBot",
    "OnDelta",
    "sleep",
    "HAS_DBOS",
]

# Typed as Any so the module type-checks whether or not dbos is installed;
# every use is guarded by HAS_DBOS.
DBOS: t.Any
try:  # pragma: no cover - trivial
    from dbos import DBOS as _DBOS

    DBOS = _DBOS
    HAS_DBOS = True
except ImportError:  # pragma: no cover - trivial
    DBOS = None
    HAS_DBOS = False


def _require_dbos(what: str) -> None:
    if not HAS_DBOS:
        raise RuntimeError(
            f"{what} needs the 'dbos' extra: pip install 'mus[dbos]'"
        )


async def sleep(seconds: float) -> None:
    """Durable sleep where DBOS is available, plain sleep otherwise.

    A replayed workflow skips a DBOS sleep but would re-serve an asyncio one, so
    injecting this into mus's retry backoff avoids re-waiting the full delay on
    recovery.
    """
    if HAS_DBOS:
        await DBOS.sleep_async(seconds)
    else:
        await asyncio.sleep(seconds)


# --- tool steps -----------------------------------------------------------

_TOOL_STEP: t.Optional[t.Callable] = None


def _tool_step() -> t.Callable:
    """The step wrapping one tool call, registered once for every tool.

    One generic step rather than one per tool name. The wrapper is
    interchangeable -- it takes ``invoke`` as an argument and holds no
    reference to any tool -- so a per-name variant bought only a label, at the
    cost of a registration per distinct name that DBOS never releases. A
    process minting a fresh tool name per request grew that registry without
    bound; this cannot.

    The tool name is recorded on the tracing span instead, so it stays visible
    where it is useful. It is *not* in ``list_workflow_steps``, which now shows
    ``mus.tool`` for every call.
    """
    global _TOOL_STEP
    if _TOOL_STEP is None:

        @DBOS.step(name="mus.tool")
        async def _run(
            tool_name: str, invoke: t.Callable[[], t.Awaitable[ToolValue]]
        ) -> ToolValue:
            # Wraps mus's own invoke, so validation and the fallback function
            # are not reimplemented and cannot drift. A closure is not
            # picklable, but step arguments are never persisted, and a replayed
            # step is not executed -- the closure is rebuilt and never called.
            try:
                DBOS.span.set_attribute("mus.tool", tool_name)
            except Exception:  # pragma: no cover - tracing must never break a run
                pass
            return await invoke()

        _TOOL_STEP = _run
    assert _TOOL_STEP is not None
    return _TOOL_STEP


async def _tool_runner(
    tool_use: ToolUse, invoke: t.Callable[[], t.Awaitable[ToolValue]]
) -> ToolValue:
    return await _tool_step()(tool_use.name, invoke)


# --- provider step --------------------------------------------------------

_END = object()


def _provider_turn_step() -> t.Callable:
    """Registered lazily so this module imports without dbos present."""
    global _PROVIDER_TURN
    if _PROVIDER_TURN is None:

        @DBOS.step(name="mus.provider_turn")
        async def _provider_turn(
            inner: LLM,
            call_kwargs: dict,
            queue: "asyncio.Queue",
            finished: t.Optional["asyncio.Queue"],
            key: t.Optional[str],
            on_delta: t.Optional[OnDelta],
        ) -> list:
            """One provider call: streams deltas live, writes them, returns them.

            The queues are in-memory -- the step runs as a task in the same event
            loop as the workflow body, so tokens need no durable channel to get
            there.

            The deltas are written to the durable stream *from inside this step*,
            which is the load-bearing detail. A write issued from the workflow
            body is a positional, determinism-checked operation, so writing one
            per delta would make the number of deltas part of the workflow's
            determinism contract. A turn interrupted mid-stream is not
            checkpointed, so recovery re-runs it against a live model, which
            answers with a different number of deltas -- and the run would then
            abort with DBOSUnexpectedStepError, permanently. Writes issued inside
            a step are not positional, so the count is free to vary.

            Each delta makes a round trip before being written: it goes up to
            mus, which stamps stream_id and tool_invocation_id and applies the
            transform hook, and the finished delta comes back here. That keeps
            the durable stream identical to what ``query`` yields without
            duplicating any of mus's tagging. The trip is exactly 1:1 because
            ``Bot.query`` yields every provider delta exactly once; the deltas it
            synthesises itself are emitted outside that loop and written by the
            body instead.

            The return value is what makes replay work: it is checkpointed, so a
            recovered run gets the turn back without re-calling the provider.
            Returning this turn's deltas only -- never accumulated history --
            keeps that storage linear in conversation size.
            """
            out = []
            opened = False
            try:
                async for delta in inner.stream(**call_kwargs):
                    out.append(delta)
                    queue.put_nowait(delta)
                    if finished is None:
                        # Nobody is consuming this turn's deltas for the stream
                        # -- fill() and fun() drive Bot.query directly, and
                        # their deltas have never been published. Don't wait for
                        # a finished delta that is not coming.
                        continue
                    tagged = await finished.get()
                    if tagged is _END:
                        # The consumer stopped reading, so no finished delta is
                        # coming. Stop rather than block the workflow forever.
                        break
                    if not opened and tagged.stream_id is not None:
                        opened = True
                        # An interrupted attempt leaves its deltas in the stream
                        # -- streams are append-only, so they cannot be taken
                        # out. They carry this same stream_id, because the id
                        # comes from a checkpointed step and replays identically,
                        # which makes them indistinguishable from what follows.
                        # Opening the turn with a reset tells a reader to discard
                        # whatever it accumulated under this id before believing
                        # what comes next. On a first attempt there is nothing to
                        # discard and it costs a reader nothing.
                        await DBOS.write_stream_async(
                            key,
                            Delta(
                                content=DeltaStreamReset(
                                    stream_id=tagged.stream_id,
                                    reason="provider turn (re)started",
                                    attempt=0,
                                ),
                                stream_id=tagged.stream_id,
                            ),
                        )
                    await DBOS.write_stream_async(key, tagged)
                    if on_delta is not None:
                        await on_delta(tagged)
            finally:
                # Always signals "this step actually executed", so the consumer
                # can tell a live run from a replay off the checkpoint.
                queue.put_nowait(_END)
            return out

        _PROVIDER_TURN = _provider_turn
    assert _PROVIDER_TURN is not None
    return _PROVIDER_TURN


class OnDelta(t.Protocol):
    """Called for each delta, from inside the step that produced it.

    The point is *where* it runs. A write issued from the workflow body is a
    positional, determinism-checked operation, so a consumer writing one record
    per delta would put the delta count back into the workflow's determinism
    contract -- and a turn re-run live after a crash emits a different number of
    deltas, which would then abort the run permanently. Writes issued inside a
    step cost no positional op, so this is the seam for a consumer that wants
    its own durable output per delta.

    Two consequences of running inside a step. Writes are at-least-once rather
    than exactly-once, since a retried step runs the hook again. And a replayed
    turn does not execute its step, so the hook does not fire for it: it runs
    exactly when the provider was actually called.

    Raising propagates: the step fails and the turn fails with it, rather than
    the error being swallowed where nobody would see it.
    """

    async def __call__(self, delta: Delta) -> None: ...


_PROVIDER_TURN: t.Optional[t.Callable] = None

# Every in-flight provider step, held so it cannot be garbage collected while
# it is still running -- the same pattern DBOS uses for its own workflow tasks.
# A step task that is collected while pending has its coroutine closed
# synchronously inside whatever task happens to be running at the time, and the
# step's ``EnterDBOSStepCtx.__exit__`` then restores its own workflow context
# into that unrelated task. The victim records its next operation under the
# wrong workflow -- or, once the abandoned workflow has ended, under a blank one
# -- and DBOS fails an assertion deep in _sys_db. See issue #86.
_PENDING_STEPS: "set[asyncio.Task]" = set()


class _DurableLLM(LLM):
    """Wraps any mus LLM so each provider call is a checkpointed step.

    Uses the existing LLM protocol, so mus needs no seam for this.
    """

    provider = "durable"

    def __init__(self, inner: LLM, key: str, on_delta: t.Optional[OnDelta] = None):
        self.inner = inner
        self.key = key
        self.on_delta = on_delta
        # Set while a turn's step is streaming; the finished deltas go back to
        # it so the step can write them (see _provider_turn).
        self._finished: t.Optional[asyncio.Queue] = None
        self._awaiting = 0
        # Deltas coming off a checkpoint: the step that produced them wrote
        # them the first time round and does not run again.
        self._replayed = 0
        # Set only while DurableBot.query is the consumer. Other entry points
        # (fill, fun) drive Bot.query directly and publish nothing.
        self.writing = False

    def deliver(self, delta: Delta) -> bool:
        """Hand a finished delta back to the running step to be written.

        False means no step is waiting for it -- either the turn's stream is
        done and this delta was synthesised by mus (a tool result, the history),
        or the turn was replayed off its checkpoint and was never written by a
        step at all. Either way the caller writes it instead.
        """
        if self._replayed > 0:
            # Already in the stream, written by the step on the run that
            # actually called the provider. Writing again would duplicate it.
            self._replayed -= 1
            return True
        if self._finished is None or self._awaiting <= 0:
            return False
        self._awaiting -= 1
        self._finished.put_nowait(delta)
        return True

    async def stream(self, **kwargs):
        if DBOS.workflow_id is None:
            raise RuntimeError("durable() bots must run inside a DBOS workflow")

        # Everything is forwarded -- notably `functions`, without which the
        # model never sees the tools at all.
        call_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        queue: asyncio.Queue = asyncio.Queue()
        writing = self.writing
        finished: t.Optional[asyncio.Queue] = asyncio.Queue() if writing else None
        self._finished = finished
        self._awaiting = 0
        task = asyncio.ensure_future(
            _provider_turn_step()(
                self.inner,
                call_kwargs,
                queue,
                finished,
                self.key if writing else None,
                self.on_delta if writing else None,
            )
        )
        # The generator frame is otherwise the only reference to the task, and
        # an abandoned generator is torn down without reaching ``await task``
        # below -- see _PENDING_STEPS.
        _PENDING_STEPS.add(task)
        task.add_done_callback(_PENDING_STEPS.discard)

        streamed = False
        getter: t.Optional[asyncio.Future] = None
        try:
            while True:
                if getter is None:
                    getter = asyncio.ensure_future(queue.get())
                await asyncio.wait(
                    {getter, task}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter.done():
                    item = getter.result()
                    getter = None
                    if item is _END:
                        break
                    streamed = True
                    if writing:
                        # One finished delta is owed back per raw one handed up.
                        self._awaiting += 1
                    yield item
                elif task.done():
                    # The step returned without executing -- a replay off its
                    # checkpoint -- so no _END is coming.
                    getter.cancel()
                    getter = None
                    break
        finally:
            if getter is not None:
                getter.cancel()
            # Release the step if the consumer stopped before the turn ended,
            # rather than leaving it blocked on a delta that will never come.
            # Keyed off the task rather than off ``_awaiting``, which is back at
            # zero for the whole window between mus handing a finished delta
            # back and the consumer taking the next raw one -- and that window
            # is exactly where a consumer that stops early lands. An _END the
            # step no longer waits for is simply never read.
            if finished is not None and not task.done():
                finished.put_nowait(_END)
            self._finished = None
            self._awaiting = 0

        deltas = await task
        if not streamed:
            # Replayed: the deltas come from the checkpoint rather than live.
            if writing:
                self._replayed = len(deltas)
            for delta in deltas:
                yield delta


# --- the public surface ---------------------------------------------------


def _error_delta(exc: BaseException) -> Delta:
    """A terminal delta describing a failure.

    Carried in ``metadata`` rather than as a new DeltaContent member: adding to
    that union would force edits to every provider adapter's exhaustiveness
    check and the cattrs converter, the same trade already made for
    ``Delta.stop_reason``. The text is left empty so a failure does not end up
    spliced into ``IterableResult.total``.
    """
    return Delta(
        content=DeltaText(data=""),
        metadata={
            "mus.error": {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        },
    )


_NEW_ID: t.Optional[t.Callable] = None


def _id_step() -> t.Callable:
    """The step that mints one correlation id, registered once.

    Cached because the name is constant and DBOS keys its function registry by
    name: re-registering would log a duplicate-registration warning per call.
    """
    global _NEW_ID
    if _NEW_ID is None:

        @DBOS.step(name="mus.id")
        async def _step() -> str:
            return uuid.uuid4().hex

        _NEW_ID = _step
    assert _NEW_ID is not None
    return _NEW_ID


async def _new_id() -> str:
    """A correlation id that a replay takes from the checkpoint.

    ``bot.query`` runs in the workflow body, so every id it mints is minted
    again on replay -- but workflow-scope writes are exactly-once per
    function_id, so deltas already in the stream keep the originals. Fresh ids
    on the replayed remainder leave a reader with tool results pairing to no
    tool use, and one turn split across several stream_ids.

    A step rather than a value derived from the workflow id. Deriving is also
    correct while the body re-executes identically, and costs no write -- but
    it goes back to failing silently once that stops holding, which is the
    failure this exists to remove. A recorded step instead makes DBOS compare
    the name at each function_id and raise DBOSUnexpectedStepError. The writes
    are a rounding error next to the one this bot already makes per delta.
    """
    return await _id_step()()


class DurableBot:
    """A Bot whose runs are checkpointed and whose deltas are streamed durably."""

    def __init__(
        self, bot: Bot, key: str = "mus", on_delta: t.Optional[OnDelta] = None
    ):
        _require_dbos("durable()")
        self._bot = bot
        self._key = key
        self._closed = False
        self._client = _DurableLLM(bot.client, key, on_delta)
        bot.client = self._client
        bot.default_args = t.cast(
            t.Any,
            {
                **bot.default_args,
                "tool_runner": _tool_runner,
                "id_generator": _new_id,
            },
        )

    def query(self, *args, **kwargs) -> t.AsyncGenerator[Delta, None]:
        async def _gen():
            self._client.writing = True
            try:
                async for delta in self._bot.query(*args, **kwargs):
                    # Provider deltas go back to the step that produced them, to
                    # be written from inside it -- a write issued here in the
                    # workflow body is a positional op, and one per delta would
                    # make the delta count part of the determinism contract, so
                    # a turn re-run live after a crash could never recover.
                    #
                    # What is left is the deltas mus synthesises itself: tool
                    # results, the history, a stream reset. No step is waiting
                    # for those, and their number is fixed by the checkpointed
                    # steps around them, so writing them here is safe.
                    if not self._client.deliver(delta):
                        await DBOS.write_stream_async(self._key, delta)
                    yield delta
            except BaseException as exc:
                # Otherwise a failed run is indistinguishable from a truncated
                # one: the stream just stops, with no reason in it.
                #
                # Best-effort: if the write itself fails -- no workflow context,
                # a closed stream, a dead database -- the original failure still
                # has to be what the caller sees, not the bookkeeping error.
                try:
                    await DBOS.write_stream_async(self._key, _error_delta(exc))
                    await self.close()
                except Exception:
                    logger.warning(
                        "could not write the failure into the stream", exc_info=True
                    )
                raise
            finally:
                self._client.writing = False

        return _gen()

    def __call__(self, query, **kwargs) -> IterableResult:
        return IterableResult(self.query(query, **kwargs))

    # Delegated so callers get the whole Bot surface, not just query(). These
    # drive the provider through the same checkpointed step; they simply don't
    # stream, so there is nothing extra to write.
    async def fill(self, *args, **kwargs):
        return await self._bot.fill(*args, **kwargs)

    def fun(self, *args, **kwargs):
        return self._bot.fun(*args, **kwargs)

    def bot(self, *args, **kwargs):
        return self._bot.bot(*args, **kwargs)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await DBOS.close_stream_async(self._key)


def durable(
    bot: Bot, key: str = "mus", on_delta: t.Optional[OnDelta] = None
) -> DurableBot:
    """Make a bot's runs durable. Raises without the ``dbos`` extra installed.

    Deliberately not a silent passthrough: the name promises a guarantee, and
    quietly not providing it would leave callers believing completed tools never
    re-fire. Run the bot unwrapped if you don't want durability.
    """
    return DurableBot(bot, key, on_delta)


async def read(
    workflow_id: str, offset: int = 0, key: str = "mus"
) -> t.AsyncGenerator[Delta, None]:
    """Tail a run's deltas from anywhere, resuming at ``offset``."""
    _require_dbos("read()")
    async for delta in DBOS.read_stream_async(workflow_id, key, offset=offset):
        yield delta


def attach(workflow_id: str, offset: int = 0, key: str = "mus") -> IterableResult:
    """The same, as the mus result object callers already know."""
    _require_dbos("attach()")
    return IterableResult(read(workflow_id, offset, key))
