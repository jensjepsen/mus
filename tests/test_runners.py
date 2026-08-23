"""Tests for the tool runner.

A durable backend needs each tool call wrapped in a checkpointed step, so a
replay after a crash returns the recorded result instead of firing the tool's
side effects a second time.

mus doesn't depend on any such backend, so it takes the runner by injection, the
way it already takes ``transform_delta_hook`` and friends. It defaults to the
behaviour mus has always had.

The follow-up turn needs no equivalent seam: with the whole recursion inside one
durable workflow, replay reaches it through cached steps on its own.
"""

import typing as t

import pytest

from mus import (
    Bot,
    Delta,
    DeltaText,
    DeltaToolResult,
    DeltaToolUse,
)
from mus.llm.types import LLM, ToolUse, ToolValue, ensure_tool_value


class StreamArgs(t.TypedDict, total=False):
    pass


class ScriptedLLM(LLM[StreamArgs, str, None]):
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


def _use(name: str, tool_id: str, **args) -> Delta:
    return Delta(content=DeltaToolUse(data=ToolUse(id=tool_id, name=name, input=args)))


def _text(data: str) -> Delta:
    return Delta(content=DeltaText(data=data))


@pytest.fixture
def tools():
    calls: list[str] = []

    async def alpha(x: int) -> str:
        """Alpha tool"""
        calls.append(f"alpha({x})")
        return "A"

    async def beta(y: int) -> str:
        """Beta tool"""
        calls.append(f"beta({y})")
        return "B"

    return alpha, beta, calls


# --- tool_runner ----------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_runner_is_called_once_per_tool_call(tools):
    alpha, beta, calls = tools
    seen: list[ToolUse] = []

    async def runner(tool_use, invoke):
        seen.append(tool_use)
        return await invoke()

    model = ScriptedLLM(
        [[_use("alpha", "t1", x=1), _use("beta", "t2", y=2)], [_text("done")]]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta], tool_runner=runner)
    await bot("go").string()

    assert [tu.name for tu in seen] == ["alpha", "beta"]
    assert [tu.id for tu in seen] == ["t1", "t2"]
    # The default invoke still ran the real tools.
    assert calls == ["alpha(1)", "beta(2)"]


@pytest.mark.asyncio
async def test_tool_runner_owns_execution(tools):
    """A runner that doesn't call invoke() replaces the result entirely.

    This is what lets a durable runner return a checkpointed result on replay
    instead of re-running the tool.
    """
    alpha, _beta, calls = tools

    async def runner(tool_use, invoke) -> ToolValue:
        return ensure_tool_value("recorded-result")

    model = ScriptedLLM([[_use("alpha", "t1", x=1)], [_text("done")]])
    bot = Bot(prompt="t", model=model, functions=[alpha], tool_runner=runner)
    deltas = [d async for d in bot.query("go")]

    # The real tool never ran...
    assert calls == []
    # ...and the recorded value reached the conversation.
    results = [
        d.content.data.content.val
        for d in deltas
        if isinstance(d.content, DeltaToolResult)
    ]
    assert results == ["recorded-result"]


@pytest.mark.asyncio
async def test_without_a_tool_runner_behaviour_is_unchanged(tools):
    alpha, beta, calls = tools
    model = ScriptedLLM(
        [[_use("alpha", "t1", x=1), _use("beta", "t2", y=2)], [_text("done")]]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    await bot("go").string()
    assert calls == ["alpha(1)", "beta(2)"]
