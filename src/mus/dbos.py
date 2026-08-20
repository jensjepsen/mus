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
  * Each tool call is a step -- one registered per tool name -- wrapping mus's
    own ``invoke`` so schema validation and the fallback function still apply.
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

from .llm.llm import Bot, IterableResult
from .llm.types import LLM, Delta, DeltaText, ToolUse, ToolValue

logger = logging.getLogger(__name__)

__all__ = ["durable", "read", "attach", "DurableBot", "sleep", "HAS_DBOS"]

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

_TOOL_STEPS: dict[str, t.Callable] = {}


def _step_for(tool_name: str):
    """One registered step per tool name, created on first use.

    Registering lazily works after launch and survives recovery in a fresh
    process, and gives each tool its own name in the checkpoint record instead
    of one generic entry.

    The tool function itself is never registered or pickled -- only this generic
    wrapper is -- which is what lets tools be closures, callable objects, or
    built dynamically per run.
    """
    if tool_name not in _TOOL_STEPS:

        @DBOS.step(name=f"mus.tool:{tool_name}")
        async def _run(invoke: t.Callable[[], t.Awaitable[ToolValue]]) -> ToolValue:
            # Wraps mus's own invoke, so validation and the fallback function
            # are not reimplemented and cannot drift. A closure is not
            # picklable, but step arguments are never persisted, and a replayed
            # step is not executed -- the closure is rebuilt and never called.
            return await invoke()

        _TOOL_STEPS[tool_name] = _run
    return _TOOL_STEPS[tool_name]


async def _tool_runner(
    tool_use: ToolUse, invoke: t.Callable[[], t.Awaitable[ToolValue]]
) -> ToolValue:
    return await _step_for(tool_use.name)(invoke)


# --- provider step --------------------------------------------------------

_END = object()


def _provider_turn_step() -> t.Callable:
    """Registered lazily so this module imports without dbos present."""
    global _PROVIDER_TURN
    if _PROVIDER_TURN is None:

        @DBOS.step(name="mus.provider_turn")
        async def _provider_turn(
            inner: LLM, call_kwargs: dict, queue: "asyncio.Queue"
        ) -> list:
            """One provider call: streams deltas live and returns them.

            The queue is in-memory -- the step runs as a task in the same event
            loop as the workflow body, so tokens need no durable channel to get
            there. An earlier version wrote them to a DBOS stream instead, which
            was worse in two ways: a crash mid-turn left an abandoned prefix
            that cannot be deleted (streams are append-only), and the reader had
            to distinguish it from the retry's output by counting offsets.

            The return value is what makes replay work: it is checkpointed, so a
            recovered run gets the turn back without re-calling the provider.
            Returning this turn's deltas only -- never accumulated history --
            keeps that storage linear in conversation size.
            """
            out = []
            try:
                async for delta in inner.stream(**call_kwargs):
                    out.append(delta)
                    queue.put_nowait(delta)
            finally:
                # Always signals "this step actually executed", so the consumer
                # can tell a live run from a replay off the checkpoint.
                queue.put_nowait(_END)
            return out

        _PROVIDER_TURN = _provider_turn
    assert _PROVIDER_TURN is not None
    return _PROVIDER_TURN


_PROVIDER_TURN: t.Optional[t.Callable] = None


class _DurableLLM(LLM):
    """Wraps any mus LLM so each provider call is a checkpointed step.

    Uses the existing LLM protocol, so mus needs no seam for this.
    """

    provider = "durable"

    def __init__(self, inner: LLM):
        self.inner = inner

    async def stream(self, **kwargs):
        if DBOS.workflow_id is None:
            raise RuntimeError("durable() bots must run inside a DBOS workflow")

        # Everything is forwarded -- notably `functions`, without which the
        # model never sees the tools at all.
        call_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.ensure_future(
            _provider_turn_step()(self.inner, call_kwargs, queue)
        )

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

        deltas = await task
        if not streamed:
            # Replayed: the deltas come from the checkpoint rather than live.
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


class DurableBot:
    """A Bot whose runs are checkpointed and whose deltas are streamed durably."""

    def __init__(self, bot: Bot, key: str = "mus"):
        _require_dbos("durable()")
        self._bot = bot
        self._key = key
        self._closed = False
        bot.client = _DurableLLM(bot.client)
        bot.default_args = t.cast(
            t.Any, {**bot.default_args, "tool_runner": _tool_runner}
        )

    def query(self, *args, **kwargs) -> t.AsyncGenerator[Delta, None]:
        async def _gen():
            try:
                async for delta in self._bot.query(*args, **kwargs):
                    # Written from workflow scope: exactly-once, and already
                    # tagged and hook-transformed by mus.
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


def durable(bot: Bot, key: str = "mus") -> DurableBot:
    """Make a bot's runs durable. Raises without the ``dbos`` extra installed.

    Deliberately not a silent passthrough: the name promises a guarantee, and
    quietly not providing it would leave callers believing completed tools never
    re-fire. Run the bot unwrapped if you don't want durability.
    """
    return DurableBot(bot, key)


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
