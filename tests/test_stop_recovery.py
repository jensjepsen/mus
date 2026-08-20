"""Tests for the stop-recovery hook (recover from an unplanned stop in-loop).

``ErrorRecoveryHook`` handles calls that fail *before* the stream starts. This
hook handles the opposite: the request succeeded, deltas were yielded, and the
model stopped early -- typically on ``max_tokens``.

Recovering in-loop is the whole point. Raising unwinds the turn, so a truncation
several tool calls deep would throw away the calls that already succeeded and
force the caller to restart the flow from outside.
"""

import typing as t

import pytest

from mus import (
    Bot,
    Delta,
    DeltaStreamReset,
    DeltaText,
    DeltaToolUse,
    Query,
    RetryPolicy,
    StopReason,
    StopRecoveryContinue,
    StopRecoveryReset,
)
from mus.llm.types import LLM, ToolUse
from mus.llm.exceptions import LLMStoppedException


class StreamArgs(t.TypedDict, total=False):
    pass


class ScriptedLLM(LLM[StreamArgs, str, None]):
    """Yields a scripted list of deltas per call, recording histories issued."""

    provider = "scripted"

    def __init__(self, script: list[list[Delta]]):
        self.script = script
        self.call_count = 0
        self.histories: list[list] = []

    async def stream(self, **kwargs):
        self.histories.append(list(kwargs.get("history") or []))
        idx = self.call_count
        self.call_count += 1
        deltas = self.script[idx] if idx < len(self.script) else self.script[-1]
        for d in deltas:
            yield d


def _text(data: str) -> Delta:
    return Delta(content=DeltaText(data=data))


def _stop(kind, raw=None) -> Delta:
    return Delta(
        content=DeltaText(data=""),
        stop_reason=StopReason(kind=kind, raw=raw if raw is not None else kind),
    )


def _tool(name="look_up", tool_id="t1", **args) -> Delta:
    return Delta(content=DeltaToolUse(data=ToolUse(id=tool_id, name=name, input=args)))


# --- The two recovery styles ---------------------------------------------


@pytest.mark.asyncio
async def test_continue_keeps_partial_and_generates_onward():
    model = ScriptedLLM(
        [
            [_text("Cartography, the art and"), _stop("max_tokens", "length")],
            [_text(" science of map-making."), _stop("end_turn")],
        ]
    )

    seen = []

    async def on_stop(error, attempt):
        seen.append((error.stop_reason.kind, attempt))
        return StopRecoveryContinue(append=[Query("You were cut off; continue.")])

    result = Bot(prompt="test", model=model, stop_recovery_hook=on_stop)("write it")
    deltas = [d async for d in result]

    assert seen == [("max_tokens", 0)]
    # The partial output stays: no reset is yielded, so what the consumer has
    # already rendered remains valid.
    assert not [d for d in deltas if isinstance(d.content, DeltaStreamReset)]
    assert result.total == "Cartography, the art and science of map-making."
    # The re-issue carried the partial output plus the nudge.
    reissued = model.histories[1]
    assert any(isinstance(h, Query) for h in reissued)


@pytest.mark.asyncio
async def test_reset_discards_partial_and_reissues():
    model = ScriptedLLM(
        [
            [_text("I cannot hel"), _stop("content_filter", "refusal")],
            [_text("Here is a safe answer."), _stop("end_turn")],
        ]
    )

    async def on_stop(error, attempt):
        return StopRecoveryReset(append=[Query("Rephrase that.")])

    result = Bot(prompt="test", model=model, stop_recovery_hook=on_stop)("ask")
    deltas = [d async for d in result]

    resets = [d.content for d in deltas if isinstance(d.content, DeltaStreamReset)]
    assert len(resets) == 1
    assert resets[0].reason == "stop_recovery"
    # The discarded text is gone from the accumulated output.
    assert result.total == "Here is a safe answer."
    # The re-issue did not carry the filtered partial.
    reissued = model.histories[1]
    assert not any(
        isinstance(h, Delta) and "cannot hel" in getattr(h.content, "data", "")
        for h in reissued
    )


@pytest.mark.asyncio
async def test_returning_none_propagates():
    model = ScriptedLLM([[_text("half"), _stop("max_tokens", "length")]])

    calls = []

    async def on_stop(error, attempt):
        calls.append(attempt)
        return None

    bot = Bot(prompt="test", model=model, stop_recovery_hook=on_stop)
    with pytest.raises(LLMStoppedException):
        await bot("go").string()
    assert calls == [0]


@pytest.mark.asyncio
async def test_no_hook_still_raises():
    model = ScriptedLLM([[_text("half"), _stop("max_tokens", "length")]])
    with pytest.raises(LLMStoppedException):
        await Bot(prompt="test", model=model)("go").string()


# --- Guard rails ----------------------------------------------------------


@pytest.mark.asyncio
async def test_continue_is_coerced_to_reset_for_a_half_emitted_tool_call():
    """Continuing a truncated tool call is never valid.

    The turn holds a malformed tool block that providers reject on the next
    request, so mus overrides the hook rather than re-sending it.
    """
    model = ScriptedLLM(
        [
            [_text("let me look"), _stop("malformed_tool_call", "max_tokens")],
            [_text("done"), _stop("end_turn")],
        ]
    )

    async def on_stop(error, attempt):
        assert error.pending_tool_call is True
        return StopRecoveryContinue()  # wrong; mus should coerce

    result = Bot(prompt="test", model=model, stop_recovery_hook=on_stop)("go")
    deltas = [d async for d in result]

    assert len([d for d in deltas if isinstance(d.content, DeltaStreamReset)]) == 1
    assert result.total == "done"


@pytest.mark.asyncio
async def test_recovery_attempts_are_capped():
    # Always truncates; the hook always wants to continue.
    model = ScriptedLLM([[_text("x"), _stop("max_tokens", "length")]])

    calls = []

    async def on_stop(error, attempt):
        calls.append(attempt)
        return StopRecoveryContinue()

    bot = Bot(
        prompt="test",
        model=model,
        stop_recovery_hook=on_stop,
        retry_policy=RetryPolicy(max_stop_recovery_attempts=3),
    )
    with pytest.raises(LLMStoppedException):
        await bot("go").string()
    assert calls == [0, 1, 2]


@pytest.mark.asyncio
async def test_hook_is_offered_a_nested_stop_only_once():
    """A declined stop must not be re-offered as it unwinds the tool loop."""

    async def look_up(query: str) -> str:
        """Look something up"""
        return "42"

    model = ScriptedLLM(
        [
            [_tool(query="x")],
            [_text("based on that,"), _stop("max_tokens", "length")],
        ]
    )

    calls = []

    async def on_stop(error, attempt):
        calls.append(attempt)
        return None

    bot = Bot(
        prompt="test", model=model, functions=[look_up], stop_recovery_hook=on_stop
    )
    with pytest.raises(LLMStoppedException):
        await bot("look up x").string()

    assert len(calls) == 1


# --- The motivating case --------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_deep_in_a_tool_flow_does_not_break_the_flow():
    tool_calls = []

    async def look_up(query: str) -> str:
        """Look something up"""
        tool_calls.append(query)
        return "42"

    model = ScriptedLLM(
        [
            [_tool(query="first")],  # turn 1: tool call
            [_text("partial "), _stop("max_tokens", "length")],  # turn 2: truncated
            [_text("and done."), _stop("end_turn")],  # turn 2 continued
        ]
    )

    async def on_stop(error, attempt):
        return StopRecoveryContinue(append=[Query("Continue.")])

    result = Bot(
        prompt="test", model=model, functions=[look_up], stop_recovery_hook=on_stop
    )("look up first")
    await result.string()

    # The flow survived the truncation rather than unwinding.
    assert tool_calls == ["first"]
    assert "partial and done." in result.total


@pytest.mark.asyncio
async def test_reset_does_not_re_run_a_completed_tool_call():
    """A reset rewinds to after the last completed tool call, not past it.

    Tools run as their deltas arrive, so rewinding to the top of the turn would
    fire their side effects a second time.
    """
    tool_calls = []

    async def record(value: str) -> str:
        """Record something"""
        tool_calls.append(value)
        return "recorded"

    model = ScriptedLLM(
        [
            # One stream: a tool call completes, then the turn is truncated.
            [
                _tool(name="record", value="once"),
                _text("now some prose"),
                _stop("content_filter", "refusal"),
            ],
            [_text("nested ok"), _stop("end_turn")],  # the tool call's turn
            [_text("clean retry"), _stop("end_turn")],  # after the reset
        ]
    )

    async def on_stop(error, attempt):
        return StopRecoveryReset()

    result = Bot(
        prompt="test", model=model, functions=[record], stop_recovery_hook=on_stop
    )("go")
    await result.string()

    assert tool_calls == ["once"]

    # The load-bearing assertion: the history the reset re-issued with still
    # contains the completed call and its result, so the model has no reason to
    # repeat it. Rewinding to the top of the turn would drop both and invite a
    # second execution.
    reissued = model.histories[2]
    kinds = [type(h.content).__name__ for h in reissued if isinstance(h, Delta)]
    assert "DeltaToolUse" in kinds
    assert "DeltaToolResult" in kinds
    # ...but the prose after the tool call, which the filter rejected, is gone.
    assert not any(
        isinstance(h, Delta)
        and isinstance(h.content, DeltaText)
        and "now some prose" in h.content.data
        for h in reissued
    )
