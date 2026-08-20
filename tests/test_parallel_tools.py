"""Tests for parallel tool calls -- several tool uses in a single turn.

Every supported provider can emit more than one tool call in one assistant turn,
and expects them answered together: one assistant message holding N ``tool_use``
blocks, replied to by one user message holding N ``tool_result`` blocks.

mus used to invoke each tool the moment its delta arrived and immediately recurse,
so the second call in a turn was not even read until the model had already been
asked to continue from the first result alone. That produced a conversation the
model never had, an extra request per extra call, and answers reasoned from
partial results.

Tools are now collected while the stream drains and invoked once it ends.
"""

import typing as t

import pytest

from mus import (
    Bot,
    Delta,
    DeltaHistory,
    DeltaText,
    DeltaToolResult,
    DeltaToolUse,
)
from mus.functions import to_schema
from mus.llm.types import LLM, ToolUse


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


def _use(name: str, tool_id: str, **args) -> Delta:
    return Delta(content=DeltaToolUse(data=ToolUse(id=tool_id, name=name, input=args)))


def _text(data: str) -> Delta:
    return Delta(content=DeltaText(data=data))


def _kinds(deltas) -> list[str]:
    out = []
    for d in deltas:
        c = d.content
        if isinstance(c, DeltaToolUse):
            out.append(f"USE:{c.data.name}")
        elif isinstance(c, DeltaToolResult):
            out.append(f"RES:{c.data.id}")
        elif isinstance(c, DeltaText):
            out.append(f"TXT:{c.data}")
        elif isinstance(c, DeltaHistory):
            out.append("HISTORY")
    return out


@pytest.fixture
def two_tools():
    order: list[str] = []

    async def alpha(x: int) -> str:
        """Alpha tool"""
        order.append("alpha")
        return "A"

    async def beta(y: int) -> str:
        """Beta tool"""
        order.append("beta")
        return "B"

    return alpha, beta, order


@pytest.mark.asyncio
async def test_both_tools_run_before_the_model_is_asked_to_continue(two_tools):
    alpha, beta, order = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1), _use("beta", "t2", y=2)],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    await bot("go").string()

    assert order == ["alpha", "beta"]
    # Two calls, not three: one turn asking for both tools, one continuing with
    # both results. A recursion per tool would make three.
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_the_continuation_sees_both_results(two_tools):
    alpha, beta, _ = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1), _use("beta", "t2", y=2)],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    await bot("go").string()

    followup = model.histories[1]
    uses = [
        h.content.data.name
        for h in followup
        if isinstance(h, Delta) and isinstance(h.content, DeltaToolUse)
    ]
    results = [
        h.content.data.id
        for h in followup
        if isinstance(h, Delta) and isinstance(h.content, DeltaToolResult)
    ]
    # The model asked for both; it must be answered about both.
    assert uses == ["alpha", "beta"]
    assert results == ["t1", "t2"]


@pytest.mark.asyncio
async def test_delta_order_groups_uses_then_results(two_tools):
    alpha, beta, _ = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1), _use("beta", "t2", y=2)],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    deltas = [d async for d in bot.query("go")]

    assert _kinds(deltas) == [
        "USE:alpha",
        "USE:beta",
        "RES:t1",
        "RES:t2",
        "TXT:done",
        "HISTORY",
    ]


@pytest.mark.asyncio
async def test_results_are_pairable_by_tool_invocation_id(two_tools):
    alpha, beta, _ = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1), _use("beta", "t2", y=2)],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    deltas = [d async for d in bot.query("go")]

    uses = {
        d.content.data.id: d.tool_invocation_id
        for d in deltas
        if isinstance(d.content, DeltaToolUse)
    }
    results = {
        d.content.data.id: d.tool_invocation_id
        for d in deltas
        if isinstance(d.content, DeltaToolResult)
    }
    # Adjacency no longer pairs them, so the invocation id must.
    assert uses.keys() == results.keys()
    for tool_id, inv in uses.items():
        assert inv is not None
        assert results[tool_id] == inv


@pytest.mark.asyncio
async def test_trailing_text_after_a_tool_use_is_still_yielded(two_tools):
    """The provider may emit content after a tool call in the same stream."""
    alpha, _beta, order = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1), _text("trailing")],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha])
    deltas = [d async for d in bot.query("go")]

    assert order == ["alpha"]
    # The trailing text belongs to the turn that requested the tool, so it is
    # yielded before the result of that call.
    assert _kinds(deltas) == [
        "USE:alpha",
        "TXT:trailing",
        "RES:t1",
        "TXT:done",
        "HISTORY",
    ]


@pytest.mark.asyncio
async def test_single_tool_call_is_unchanged(two_tools):
    alpha, _beta, order = two_tools
    model = ScriptedLLM([[_use("alpha", "t1", x=1)], [_text("done")]])
    bot = Bot(prompt="t", model=model, functions=[alpha])
    deltas = [d async for d in bot.query("go")]

    assert order == ["alpha"]
    assert model.call_count == 2
    assert _kinds(deltas) == ["USE:alpha", "RES:t1", "TXT:done", "HISTORY"]


@pytest.mark.asyncio
async def test_tools_across_successive_turns_still_chain(two_tools):
    """One call per turn, over two turns: unchanged behaviour."""
    alpha, beta, order = two_tools
    model = ScriptedLLM(
        [
            [_use("alpha", "t1", x=1)],
            [_use("beta", "t2", y=2)],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta])
    deltas = [d async for d in bot.query("go")]

    assert order == ["alpha", "beta"]
    assert model.call_count == 3
    assert _kinds(deltas) == [
        "USE:alpha",
        "RES:t1",
        "USE:beta",
        "RES:t2",
        "TXT:done",
        "HISTORY",
    ]


@pytest.mark.asyncio
async def test_three_parallel_calls(two_tools):
    alpha, beta, order = two_tools

    async def gamma(z: int) -> str:
        """Gamma tool"""
        order.append("gamma")
        return "C"

    model = ScriptedLLM(
        [
            [
                _use("alpha", "t1", x=1),
                _use("beta", "t2", y=2),
                _use("gamma", "t3", z=3),
            ],
            [_text("done")],
        ]
    )
    bot = Bot(prompt="t", model=model, functions=[alpha, beta, gamma])
    await bot("go").string()

    assert order == ["alpha", "beta", "gamma"]
    assert model.call_count == 2


def test_anthropic_message_shape_for_parallel_calls():
    """[use, use, result, result] must collapse to the shape providers expect."""
    from mus.llm.anthropic import deltas_to_messages
    from mus.llm.types import Query, ToolResult, ensure_tool_value

    history = [
        Query("go"),
        _use("alpha", "t1", x=1),
        _use("beta", "t2", y=2),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="t1", content=ensure_tool_value("A"))
            )
        ),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="t2", content=ensure_tool_value("B"))
            )
        ),
    ]
    messages = deltas_to_messages(history)

    msgs = t.cast(list, messages)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    # One assistant turn holding both tool_use blocks...
    assistant_blocks = t.cast(list, msgs[1]["content"])
    assert [b["type"] for b in assistant_blocks] == ["tool_use", "tool_use"]
    # ...answered by one user turn holding both tool_result blocks.
    result_blocks = t.cast(list, msgs[2]["content"])
    assert [b["type"] for b in result_blocks] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in result_blocks] == ["t1", "t2"]


# --- Google: parallel calls to the same function must stay distinguishable ---
#
# Gemini does not populate function_call.id, so the adapter falls back to the
# function *name*. When one turn calls the same function twice -- the common
# parallel case, "weather in Paris and Tokyo" -- both calls, and both results,
# end up sharing an id.
#
# That was survivable while a tool result was yielded immediately after its use:
# consumers paired them by adjacency. Grouping the calls removes that, and
# tool_invocation_id is what replaces it -- so on Gemini the replacement pairing
# key is currently unusable.

from unittest.mock import AsyncMock, Mock  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

from mus.llm.google import GoogleGenAILLM, deltas_to_contents  # noqa: E402
from mus.llm.types import Query, ToolResult, ensure_tool_value  # noqa: E402


@pytest.fixture
def google_llm():
    client = Mock(spec=genai.Client)
    client.aio = Mock()
    client.aio.models = Mock()
    return GoogleGenAILLM("gemini-2.5-flash", client)


async def _to_async(seq):
    for item in seq:
        yield item


def _gemini_call_part(name: str, args: dict):
    """A Gemini function_call part. Note id is None -- Gemini omits it."""
    call = Mock()
    call.id = None
    call.name = name
    call.args = args

    part = Mock()
    part.text = None
    part.thought_signature = None
    part.function_call = call
    return part


def _gemini_response(parts):
    candidate = Mock()
    candidate.content = Mock()
    candidate.content.parts = parts
    candidate.finish_reason = genai_types.FinishReason.STOP

    resp = Mock()
    resp.candidates = [candidate]
    resp.usage_metadata = None
    resp.prompt_feedback = Mock()
    resp.prompt_feedback.block_reason = None
    return resp


@pytest.mark.asyncio
async def test_gemini_parallel_calls_to_one_function_get_distinct_ids(google_llm):
    """DRIVES: two calls to the same function must not collide on id."""

    async def get_weather(city: str) -> str:
        """Get the weather"""
        return "sunny"

    google_llm.client.aio.models.generate_content_stream = AsyncMock(
        return_value=_to_async(
            [
                _gemini_response(
                    [
                        _gemini_call_part("get_weather", {"city": "Paris"}),
                        _gemini_call_part("get_weather", {"city": "Tokyo"}),
                    ]
                )
            ]
        )
    )

    deltas = [
        d
        async for d in google_llm.stream(
            prompt="p", model="gemini-2.5-flash", history=[],
            functions=[to_schema(get_weather)],
        )
    ]
    uses = [d.content for d in deltas if isinstance(d.content, DeltaToolUse)]
    assert len(uses) == 2
    assert uses[0].data.input["city"] == "Paris"
    assert uses[1].data.input["city"] == "Tokyo"
    # Without distinct ids there is no way to say which result answers which call.
    assert uses[0].data.id != uses[1].data.id


@pytest.mark.asyncio
async def test_gemini_parallel_calls_get_distinct_invocation_ids(google_llm):
    """DRIVES: the id mus hands consumers for pairing must differ per call."""

    async def get_weather(city: str) -> str:
        """Get the weather"""
        return "sunny"

    google_llm.client.aio.models.generate_content_stream = AsyncMock(
        return_value=_to_async(
            [
                _gemini_response(
                    [
                        _gemini_call_part("get_weather", {"city": "Paris"}),
                        _gemini_call_part("get_weather", {"city": "Tokyo"}),
                    ]
                )
            ]
        )
    )

    bot = Bot(prompt="p", model=google_llm, functions=[get_weather])
    deltas = [d async for d in bot.query("weather in Paris and Tokyo?")]

    use_invs = [
        d.tool_invocation_id for d in deltas if isinstance(d.content, DeltaToolUse)
    ]
    assert len(use_invs) == 2
    assert use_invs[0] != use_invs[1]


def test_gemini_function_response_still_carries_the_real_name():
    """GUARD: a synthetic id must not leak onto the wire as the function name."""
    history = [
        Query("go"),
        _use("get_weather", "get_weather:0", city="Paris"),
        _use("get_weather", "get_weather:1", city="Tokyo"),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="get_weather:0", content=ensure_tool_value("rain"))
            )
        ),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="get_weather:1", content=ensure_tool_value("clear"))
            )
        ),
    ]
    contents = deltas_to_contents(history)
    names = [
        p.function_response.name
        for c in contents
        for p in (c.parts or [])
        if getattr(p, "function_response", None)
    ]
    assert names == ["get_weather", "get_weather"]


def test_gemini_distinct_results_are_not_collapsed_into_one():
    """DRIVES: results keyed by a colliding id lose one of the two values."""
    history = [
        Query("go"),
        _use("get_weather", "get_weather", city="Paris"),
        _use("get_weather", "get_weather", city="Tokyo"),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="get_weather", content=ensure_tool_value("rain"))
            )
        ),
        Delta(
            content=DeltaToolResult(
                data=ToolResult(id="get_weather", content=ensure_tool_value("clear"))
            )
        ),
    ]
    contents = deltas_to_contents(history)
    blob = str(contents)
    # Both distinct answers must survive the conversion.
    assert "rain" in blob
    assert "clear" in blob


@pytest.mark.asyncio
async def test_gemini_ids_are_unique_across_turns_not_just_within_one(google_llm):
    """DRIVES: minted ids must not repeat between turns.

    deltas_to_contents keys `unsigned_ids` and `collapsed_result_text` by tool id
    over the *whole* history. A counter that restarts each turn makes turn 3's
    first call collide with turn 1's, so one turn's collapsed result text can be
    attributed to another turn's call.
    """

    async def get_weather(city: str) -> str:
        """Get the weather"""
        return "sunny"

    seen_ids = []
    for city in ("Paris", "Tokyo"):
        google_llm.client.aio.models.generate_content_stream = AsyncMock(
            return_value=_to_async(
                [_gemini_response([_gemini_call_part("get_weather", {"city": city})])]
            )
        )
        async for d in google_llm.stream(
            prompt="p", model="gemini-2.5-flash", history=[],
            functions=[to_schema(get_weather)],
        ):
            if isinstance(d.content, DeltaToolUse):
                seen_ids.append(d.content.data.id)

    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]


# --- Google: parallel calls and thought signatures -------------------------
#
# Gemini 3 rejects a functionCall replayed without the thought_signature it was
# issued with, so deltas_to_contents collapses signature-less calls into plain
# text to keep the request valid.
#
# But Gemini stamps the signature on the *first* call of a parallel batch only.
# Its unsigned siblings belong to the same model turn and are covered by that
# signature -- collapsing them means the model never sees them answered as
# function calls, so it re-requests them, turn after turn.


def _signed_use(name, tool_id, stream_id, signature, **args) -> Delta:
    return Delta(
        content=DeltaToolUse(data=ToolUse(id=tool_id, name=name, input=args)),
        stream_id=stream_id,
        metadata={"thought_signature": signature} if signature else {},
    )


def _result(tool_id, value, stream_id) -> Delta:
    return Delta(
        content=DeltaToolResult(
            data=ToolResult(id=tool_id, content=ensure_tool_value(value))
        ),
        stream_id=stream_id,
    )


def _part_kinds(contents):
    kinds = []
    for c in contents:
        for p in c.parts or []:
            if getattr(p, "function_call", None):
                kinds.append(f"call:{p.function_call.name}")
            elif getattr(p, "function_response", None):
                kinds.append(f"response:{p.function_response.name}")
            elif getattr(p, "text", None):
                kinds.append("text")
    return kinds


def test_unsigned_sibling_of_a_signed_call_is_not_collapsed():
    """DRIVES: only the first call of a parallel batch carries a signature."""
    history = [
        Query("weather in Paris and Tokyo?"),
        _signed_use("get_weather", "gw:a", "s1", b"sig-xyz", city="Paris"),
        _signed_use("get_weather", "gw:b", "s1", None, city="Tokyo"),
        _result("gw:a", "17C rain", "s1"),
        _result("gw:b", "26C clear", "s1"),
    ]
    kinds = _part_kinds(deltas_to_contents(history))

    # Both calls must go out as real function calls, each answered.
    assert kinds.count("call:get_weather") == 2
    assert kinds.count("response:get_weather") == 2
    # ...and neither collapsed into a text note.
    assert "text" not in kinds[1:]


def test_a_wholly_unsigned_turn_is_still_collapsed():
    """GUARD: the original protection must survive.

    A turn with no signature anywhere cannot be replayed as function calls --
    that is the 400 the collapse exists to prevent.
    """
    history = [
        Query("go"),
        _signed_use("signed_tool", "st:a", "s1", b"sig-xyz", x=1),
        _result("st:a", "ok", "s1"),
        # A later turn with no signature at all.
        _signed_use("get_weather", "gw:b", "s2", None, city="Tokyo"),
        _result("gw:b", "26C clear", "s2"),
    ]
    kinds = _part_kinds(deltas_to_contents(history))

    assert kinds.count("call:get_weather") == 0
    assert kinds.count("response:get_weather") == 0
    assert "text" in kinds


def test_function_responses_use_a_role_every_gemini_family_accepts():
    """DRIVES: role="tool" is rejected outright by Gemini 3 models.

    gemini-3.5-flash-lite answers a function response sent with role="tool"
    with 400 "Role 'tool' is not supported. Please use a valid role: SYSTEM,
    SYSTEM_1, USER, ASSISTANT, DEVELOPER, CONTEXT, USER_CONTEXT, MODEL, USER",
    which breaks every tool flow on that family. gemini-2.5-flash accepts
    "tool" and "user" alike, so "user" is the one that works everywhere.
    """
    history = [
        Query("weather in Paris?"),
        _signed_use("get_weather", "gw:a", "s1", b"sig-xyz", city="Paris"),
        _result("gw:a", "17C rain", "s1"),
    ]
    contents = deltas_to_contents(history)

    response_roles = [
        c.role
        for c in contents
        for p in (c.parts or [])
        if getattr(p, "function_response", None)
    ]
    assert response_roles == ["user"]


def test_a_turn_whose_first_call_is_unsigned_still_collapses():
    """DRIVES: Gemini requires the signature on the *first* call of a turn.

    Measured against gemini-3.5-flash-lite by replaying a real signed batch with
    signatures selectively stripped:

        signatures as issued (first signed)  -> OK
        only FIRST signed                    -> OK
        only LAST signed                     -> 400 missing thought_signature
        all stripped                         -> 400 missing thought_signature

    So "the turn contains a signed call somewhere" is too weak a test: a turn
    whose first call lost its signature is not replayable, even if a later one
    kept theirs.
    """
    history = [
        Query("weather in Paris and Tokyo?"),
        _signed_use("get_weather", "gw:a", "s1", None, city="Paris"),
        _signed_use("get_weather", "gw:b", "s1", b"sig-xyz", city="Tokyo"),
        _result("gw:a", "17C rain", "s1"),
        _result("gw:b", "26C clear", "s1"),
    ]
    kinds = _part_kinds(deltas_to_contents(history))

    # Not replayable: no functionCall parts may survive from this turn.
    assert kinds.count("call:get_weather") == 0
    assert "text" in kinds
