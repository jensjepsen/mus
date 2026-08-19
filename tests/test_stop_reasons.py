"""Tests for normalized stop reasons and the unplanned-stop exception.

Every provider reports *why* generation ended; mus normalises those into a small
shared vocabulary (``StopReason``). A stop the caller asked for -- end_turn,
stop_sequence, tool_use -- is reported and nothing more. Anything else is
"unplanned" and raises ``LLMStoppedException``, because silently returning a
truncated answer (or silently dropping a truncated tool call) is data loss.

There is deliberately no recovery hook here: a half-emitted tool call cannot be
continued, so the exception instead carries the state needed to recover at the
call site (``history``, ``partial_text``, ``pending_tool_call``).
"""

import typing as t
from unittest.mock import Mock, AsyncMock

import pytest

from mus import (
    Bot,
    Delta,
    DeltaText,
    DeltaToolUse,
    DeltaToolResult,
    IterableResult,
    Query,
    StopReason,
)
from mus.llm.types import LLM, StopReasonKind, ToolUse
from mus.llm.exceptions import LLMStoppedException

from mus.llm.anthropic import _map_anthropic_stop_reason
from mus.llm.openai import _map_openai_stop_reason, OpenAILLM
from mus.llm.bedrock import _map_bedrock_stop_reason
from mus.llm.google import _map_google_stop_reason
from mus.llm.mistral import _map_mistral_stop_reason, MistralLLM
from mus.llm.google import GoogleGenAILLM, _raw_finish_reason
from mus.llm.anthropic import AnthropicLLM
from mus.functions import to_schema

from mistralai.client.sdk import Mistral
from google import genai
from google.genai import types as genai_types
import anthropic
import openai


# --- Normalisation tables -------------------------------------------------


@pytest.mark.parametrize(
    "mapper,raw,expected",
    [
        # Planned
        (_map_anthropic_stop_reason, "end_turn", "end_turn"),
        (_map_anthropic_stop_reason, "stop_sequence", "stop_sequence"),
        (_map_anthropic_stop_reason, "tool_use", "tool_use"),
        (_map_openai_stop_reason, "stop", "end_turn"),
        (_map_openai_stop_reason, "tool_calls", "tool_use"),
        (_map_openai_stop_reason, "function_call", "tool_use"),
        (_map_bedrock_stop_reason, "end_turn", "end_turn"),
        (_map_bedrock_stop_reason, "stop_sequence", "stop_sequence"),
        (_map_google_stop_reason, "STOP", "end_turn"),
        (_map_mistral_stop_reason, "stop", "end_turn"),
        (_map_mistral_stop_reason, "tool_calls", "tool_use"),
        # Truncation
        (_map_anthropic_stop_reason, "max_tokens", "max_tokens"),
        (_map_openai_stop_reason, "length", "max_tokens"),
        (_map_bedrock_stop_reason, "max_tokens", "max_tokens"),
        (_map_bedrock_stop_reason, "model_context_window_exceeded", "max_tokens"),
        (_map_google_stop_reason, "MAX_TOKENS", "max_tokens"),
        (_map_mistral_stop_reason, "length", "max_tokens"),
        (_map_mistral_stop_reason, "model_length", "max_tokens"),
        # Filtering / refusal
        (_map_anthropic_stop_reason, "refusal", "content_filter"),
        (_map_openai_stop_reason, "content_filter", "content_filter"),
        (_map_bedrock_stop_reason, "content_filtered", "content_filter"),
        (_map_bedrock_stop_reason, "guardrail_intervened", "content_filter"),
        (_map_google_stop_reason, "SAFETY", "content_filter"),
        (_map_google_stop_reason, "PROHIBITED_CONTENT", "content_filter"),
        # Provider-reported malformed calls
        (_map_google_stop_reason, "MALFORMED_FUNCTION_CALL", "malformed_tool_call"),
        (_map_google_stop_reason, "UNEXPECTED_TOOL_CALL", "malformed_tool_call"),
        # Other
        (_map_anthropic_stop_reason, "pause_turn", "pause_turn"),
        (_map_mistral_stop_reason, "error", "error"),
        (_map_google_stop_reason, "OTHER", "error"),
    ],
)
def test_normalises_provider_vocabularies(mapper, raw, expected):
    stop = mapper(raw)
    assert stop is not None
    assert stop.kind == expected
    # The provider's own word always survives normalisation.
    assert stop.raw == raw


@pytest.mark.parametrize(
    "mapper",
    [
        _map_anthropic_stop_reason,
        _map_openai_stop_reason,
        _map_bedrock_stop_reason,
        _map_google_stop_reason,
        _map_mistral_stop_reason,
    ],
)
def test_unrecognised_value_is_unknown_and_unplanned(mapper):
    # A provider adding a new reason must not be silently treated as a normal
    # end of turn -- but the raw value stays available to the caller.
    stop = mapper("brand_new_provider_reason")
    assert stop is not None
    assert stop.kind == "unknown"
    assert stop.raw == "brand_new_provider_reason"
    assert stop.is_planned is False


@pytest.mark.parametrize(
    "mapper",
    [
        _map_anthropic_stop_reason,
        _map_openai_stop_reason,
        _map_bedrock_stop_reason,
        _map_google_stop_reason,
        _map_mistral_stop_reason,
    ],
)
@pytest.mark.parametrize("raw", [None, ""])
def test_unreported_reason_is_not_a_stop(mapper, raw):
    # "Didn't say" must not be escalated to "stopped unexpectedly", or a
    # provider omitting the field would break every call.
    assert mapper(raw) is None


@pytest.mark.parametrize(
    "mapper,raw",
    [
        (_map_anthropic_stop_reason, "max_tokens"),
        (_map_openai_stop_reason, "length"),
        (_map_bedrock_stop_reason, "max_tokens"),
        (_map_mistral_stop_reason, "length"),
    ],
)
def test_pending_tools_on_unplanned_stop_is_malformed(mapper, raw):
    stop = mapper(raw, pending_tools=True)
    assert stop is not None
    assert stop.kind == "malformed_tool_call"
    # The underlying reason is still recoverable from raw.
    assert stop.raw == raw


@pytest.mark.parametrize(
    "mapper,raw",
    [
        (_map_anthropic_stop_reason, "tool_use"),
        (_map_openai_stop_reason, "tool_calls"),
        (_map_bedrock_stop_reason, "tool_use"),
        (_map_mistral_stop_reason, "tool_calls"),
    ],
)
def test_pending_tools_on_planned_stop_is_untouched(mapper, raw):
    # A normal tool call has tools "pending" too -- that must stay planned.
    stop = mapper(raw, pending_tools=True)
    assert stop is not None
    assert stop.kind == "tool_use"


def test_anthropic_carries_matched_stop_sequence():
    stop = _map_anthropic_stop_reason("stop_sequence", stop_sequence="```")
    assert stop is not None
    assert stop.stop_sequence == "```"


def test_is_planned_partitions_the_vocabulary():
    planned: list[StopReasonKind] = ["end_turn", "stop_sequence", "tool_use"]
    unplanned: list[StopReasonKind] = [
        "max_tokens",
        "content_filter",
        "malformed_tool_call",
        "pause_turn",
        "error",
        "unknown",
    ]
    for kind in planned:
        assert StopReason(kind=kind, raw=kind).is_planned is True
    for kind in unplanned:
        assert StopReason(kind=kind, raw=kind).is_planned is False


# --- Bot-level behaviour --------------------------------------------------


class StreamArgs(t.TypedDict, total=False):
    pass


class ScriptedLLM(LLM[StreamArgs, str, None]):
    """Yields a scripted list of deltas per call, recording histories issued."""

    provider = "scripted"

    def __init__(self, script: list[list[Delta]]):
        self.script = script
        self.call_count = 0
        self.histories: list[object] = []

    async def stream(self, **kwargs):
        self.histories.append(list(kwargs.get("history") or []))
        idx = self.call_count
        self.call_count += 1
        deltas = self.script[idx] if idx < len(self.script) else self.script[-1]
        for d in deltas:
            yield d


def _stop(kind, raw=None) -> Delta:
    return Delta(
        content=DeltaText(data=""),
        stop_reason=StopReason(kind=kind, raw=raw if raw is not None else kind),
    )


@pytest.mark.asyncio
async def test_planned_stop_does_not_raise_and_is_reported():
    model = ScriptedLLM([[Delta(content=DeltaText(data="all done")), _stop("end_turn")]])
    bot = Bot(prompt="test", model=model)

    result = bot("hi")
    text = await result.string()

    assert text == "all done"
    assert result.stop_reason is not None
    assert result.stop_reason.kind == "end_turn"


@pytest.mark.asyncio
async def test_stop_sequence_is_planned_and_distinguishable():
    model = ScriptedLLM(
        [
            [
                Delta(content=DeltaText(data="partial")),
                Delta(
                    content=DeltaText(data=""),
                    stop_reason=StopReason(
                        kind="stop_sequence", raw="stop_sequence", stop_sequence="```"
                    ),
                ),
            ]
        ]
    )
    result = Bot(prompt="test", model=model)("hi")
    await result.string()

    assert result.stop_reason is not None
    assert result.stop_reason.kind == "stop_sequence"
    assert result.stop_reason.stop_sequence == "```"


@pytest.mark.asyncio
async def test_truncation_raises_with_partial_text_and_history():
    model = ScriptedLLM(
        [[Delta(content=DeltaText(data="half an ess")), _stop("max_tokens", "length")]]
    )
    bot = Bot(prompt="test", model=model)

    with pytest.raises(LLMStoppedException) as exc_info:
        await bot("write an essay").string()

    exc = exc_info.value
    assert exc.stop_reason.kind == "max_tokens"
    # The provider's own word is preserved for callers that need it.
    assert exc.stop_reason.raw == "length"
    assert exc.partial_text == "half an ess"
    assert exc.pending_tool_call is False
    assert exc.provider == "scripted"
    # History is non-empty so the caller can re-issue from it.
    assert exc.history


@pytest.mark.asyncio
async def test_content_filter_raises():
    model = ScriptedLLM([[_stop("content_filter", "refusal")]])
    with pytest.raises(LLMStoppedException) as exc_info:
        await Bot(prompt="test", model=model)("hi").string()
    assert exc_info.value.stop_reason.kind == "content_filter"


@pytest.mark.asyncio
async def test_unknown_reason_raises_and_preserves_raw():
    model = ScriptedLLM([[_stop("unknown", "some_new_reason")]])
    with pytest.raises(LLMStoppedException) as exc_info:
        await Bot(prompt="test", model=model)("hi").string()
    assert exc_info.value.stop_reason.raw == "some_new_reason"


@pytest.mark.asyncio
async def test_no_reported_stop_reason_never_raises():
    # Providers/stubs that report nothing must keep working unchanged.
    model = ScriptedLLM([[Delta(content=DeltaText(data="hello"))]])
    result = Bot(prompt="test", model=model)("hi")
    assert await result.string() == "hello"
    assert result.stop_reason is None


@pytest.mark.asyncio
async def test_truncated_tool_call_reports_pending_tool_call():
    model = ScriptedLLM(
        [
            [
                Delta(content=DeltaText(data="let me look that up")),
                _stop("malformed_tool_call", "max_tokens"),
            ]
        ]
    )

    with pytest.raises(LLMStoppedException) as exc_info:
        await Bot(prompt="test", model=model)("search for x").string()

    exc = exc_info.value
    # The signal that this turn must be re-issued rather than continued.
    assert exc.pending_tool_call is True
    assert exc.stop_reason.raw == "max_tokens"


@pytest.mark.asyncio
async def test_stop_in_nested_turn_carries_the_whole_turn():
    """A stop three levels into a tool loop must not lose the earlier work.

    The exception propagates out of the generator before the closing
    DeltaHistory is yielded, so without ``history`` on the exception the caller
    would lose the tool calls that already succeeded.
    """

    async def look_up(query: str) -> str:
        """Look something up"""
        return "the answer is 42"

    model = ScriptedLLM(
        [
            # First turn: a complete, successful tool call.
            [
                Delta(
                    content=DeltaToolUse(
                        data=ToolUse(id="t1", name="look_up", input={"query": "x"})
                    )
                ),
            ],
            # Second turn (recursive): truncated.
            [
                Delta(content=DeltaText(data="based on that,")),
                _stop("max_tokens", "length"),
            ],
        ]
    )

    bot = Bot(prompt="test", model=model, functions=[look_up])

    with pytest.raises(LLMStoppedException) as exc_info:
        await bot("look up x").string()

    history = exc_info.value.history
    contents = [h.content for h in history if isinstance(h, Delta)]
    tool_uses = [c for c in contents if isinstance(c, DeltaToolUse)]
    tool_results = [c for c in contents if isinstance(c, DeltaToolResult)]
    # The completed first turn survives on the exception.
    assert len(tool_uses) == 1
    assert tool_uses[0].data.name == "look_up"
    assert len(tool_results) == 1
    # partial_text is the truncated turn's own output.
    assert exc_info.value.partial_text == "based on that,"


@pytest.mark.asyncio
async def test_tool_use_does_not_mask_the_terminal_reason():
    """IterableResult keeps the inner turn's reason, not the outer "tool_use".

    A DeltaToolUse makes Bot.query recurse inline, so for providers that report
    usage after finish_reason the outer turn's trailing tool_use delta arrives
    *after* the inner turn's real terminal reason. Last-wins would clobber it.
    """

    async def look_up(query: str) -> str:
        """Look something up"""
        return "42"

    model = ScriptedLLM(
        [
            [
                Delta(
                    content=DeltaToolUse(
                        data=ToolUse(id="t1", name="look_up", input={"query": "x"})
                    )
                ),
                # Trailing terminal delta, arriving after the recursion.
                _stop("tool_use"),
            ],
            [Delta(content=DeltaText(data="done")), _stop("end_turn")],
        ]
    )

    result = Bot(prompt="test", model=model, functions=[look_up])("look up x")
    await result.string()

    assert result.stop_reason is not None
    assert result.stop_reason.kind == "end_turn"


@pytest.mark.asyncio
async def test_recovery_at_the_call_site():
    """The documented idiom: catch, extend the history, re-issue."""
    model = ScriptedLLM(
        [
            [Delta(content=DeltaText(data="part one")), _stop("max_tokens", "length")],
            [Delta(content=DeltaText(data=" part two")), _stop("end_turn")],
        ]
    )
    bot = Bot(prompt="test", model=model)

    try:
        text = await bot("write it").string()
    except LLMStoppedException as e:
        assert e.stop_reason.kind == "max_tokens"
        assert not e.pending_tool_call
        text = e.partial_text + await IterableResult(
            bot.query(history=e.history + [Query("You were cut off; continue.")])
        ).string()

    assert text == "part one part two"


# --- Provider-level integration -------------------------------------------


async def to_async_response(seq):
    for item in seq:
        yield item


@pytest.fixture
def mistral_llm():
    client = Mock(spec=Mistral)
    client.chat = Mock()
    client.chat.stream_async = AsyncMock()
    client.chat.complete_async = AsyncMock()
    return MistralLLM("mistral-medium", client)


@pytest.fixture
def google_llm():
    client = Mock(spec=genai.Client)
    client.aio = Mock()
    client.aio.models = Mock()
    return GoogleGenAILLM("gemini-1.5-pro", client)


@pytest.fixture
def openai_llm():
    client = AsyncMock(spec=openai.AsyncClient)
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return OpenAILLM("gpt-4o-mini", client)


@pytest.fixture
def anthropic_llm():
    client = Mock(spec=anthropic.AsyncAnthropic)
    client.messages = Mock()
    return AnthropicLLM("claude-sonnet-4-5", client)


def _mistral_chunk(*, content=None, tool_calls=None, finish_reason=None):
    chunk = Mock()
    chunk.data = Mock()
    chunk.data.usage = None
    chunk.data.choices = [Mock()]
    chunk.data.choices[0].delta = Mock()
    chunk.data.choices[0].delta.content = content
    chunk.data.choices[0].delta.tool_calls = tool_calls
    chunk.data.choices[0].finish_reason = finish_reason
    return chunk


@pytest.mark.asyncio
async def test_mistral_truncated_tool_call_is_not_invoked(mistral_llm):
    """Regression: Mistral used to yield tool calls as chunks arrived.

    A stream cut off mid-arguments therefore went through repair_json and the
    tool was invoked with fabricated input. Tool calls are now buffered and only
    flushed on a planned stop.
    """

    def dummy_func(param: str) -> str:
        """Dummy function for testing"""
        return "result"

    function = Mock()
    function.name = "dummy_func"
    # Truncated mid-arguments.
    function.arguments = '{"param": "val'

    chunks = [
        _mistral_chunk(tool_calls=[Mock(id="tool_1", function=function)]),
        _mistral_chunk(finish_reason="length"),
    ]
    mistral_llm.client.chat.stream_async.return_value = to_async_response(chunks)

    results = [
        d
        async for d in mistral_llm.stream(
            prompt="Test prompt",
            model="test-model",
            history=[],
            functions=[to_schema(dummy_func)],
        )
    ]

    # No tool use is handed on...
    assert not [d for d in results if isinstance(d.content, DeltaToolUse)]
    # ...and the stop is reported as a truncated call.
    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert len(stops) == 1
    assert stops[0].kind == "malformed_tool_call"
    assert stops[0].raw == "length"


@pytest.mark.asyncio
async def test_mistral_complete_tool_call_still_flushes(mistral_llm):
    def dummy_func(param: str) -> str:
        """Dummy function for testing"""
        return "result"

    function = Mock()
    function.name = "dummy_func"
    function.arguments = '{"param": "value"}'

    chunks = [
        _mistral_chunk(tool_calls=[Mock(id="tool_1", function=function)]),
        _mistral_chunk(finish_reason="tool_calls"),
    ]
    mistral_llm.client.chat.stream_async.return_value = to_async_response(chunks)

    results = [
        d
        async for d in mistral_llm.stream(
            prompt="Test prompt",
            model="test-model",
            history=[],
            functions=[to_schema(dummy_func)],
        )
    ]

    tool_uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.input == {"param": "value"}


# --- Google adapter: real enums, not mock stand-ins ------------------------


def _google_chunk(finish_reason=None, *, text="hi", block_reason=None, candidates=True):
    """A response chunk carrying a REAL FinishReason/BlockedReason enum.

    google.genai's FinishReason subclasses str, so a Mock stand-in cannot stress
    the extraction path the way the real member does.
    """
    part = Mock()
    part.text = text
    part.function_call = None
    part.thought_signature = None

    candidate = Mock()
    candidate.content = Mock()
    candidate.content.parts = [part]
    candidate.finish_reason = finish_reason

    resp = Mock()
    resp.candidates = [candidate] if candidates else []
    resp.usage_metadata = None
    resp.prompt_feedback = Mock()
    resp.prompt_feedback.block_reason = block_reason
    return resp


@pytest.mark.parametrize(
    "finish_reason,expected_kind,expected_raw",
    [
        (genai_types.FinishReason.STOP, "end_turn", "STOP"),
        (genai_types.FinishReason.MAX_TOKENS, "max_tokens", "MAX_TOKENS"),
        (genai_types.FinishReason.SAFETY, "content_filter", "SAFETY"),
        (
            genai_types.FinishReason.MALFORMED_FUNCTION_CALL,
            "malformed_tool_call",
            "MALFORMED_FUNCTION_CALL",
        ),
    ],
)
@pytest.mark.asyncio
async def test_google_stream_reports_stop_reason(
    google_llm, finish_reason, expected_kind, expected_raw
):
    google_llm.client.aio.models.generate_content_stream = AsyncMock(
        return_value=to_async_response([_google_chunk(finish_reason)])
    )

    results = [
        d
        async for d in google_llm.stream(
            prompt="Test prompt", model="gemini-1.5-pro", history=[]
        )
    ]

    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert len(stops) == 1
    assert stops[0].kind == expected_kind
    # raw must be a plain string, not the enum member: FinishReason subclasses
    # str, so passing the member through would leave an enum here (and str() on
    # it yields "FinishReason.STOP").
    assert stops[0].raw == expected_raw
    assert type(stops[0].raw) is str


@pytest.mark.asyncio
async def test_google_blocked_prompt_is_content_filter(google_llm):
    # A prompt blocked outright comes back with no candidates at all, so the
    # reason lives on prompt_feedback.
    google_llm.client.aio.models.generate_content_stream = AsyncMock(
        return_value=to_async_response(
            [
                _google_chunk(
                    block_reason=genai_types.BlockedReason.SAFETY, candidates=False
                )
            ]
        )
    )

    results = [
        d
        async for d in google_llm.stream(
            prompt="Test prompt", model="gemini-1.5-pro", history=[]
        )
    ]

    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert len(stops) == 1
    assert stops[0].kind == "content_filter"
    assert stops[0].raw == "SAFETY"


@pytest.mark.asyncio
async def test_google_no_finish_reason_reports_nothing(google_llm):
    google_llm.client.aio.models.generate_content_stream = AsyncMock(
        return_value=to_async_response([_google_chunk(None)])
    )

    results = [
        d
        async for d in google_llm.stream(
            prompt="Test prompt", model="gemini-1.5-pro", history=[]
        )
    ]

    assert all(d.stop_reason is None for d in results)


# --- Anthropic adapter: message_stop carries the reason -------------------


class _FakeAnthropicStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e

        return gen()


def _anthropic_message_stop(stop_reason, *, stop_sequence=None):
    event = Mock()
    event.type = "message_stop"
    event.message = Mock()
    event.message.stop_reason = stop_reason
    event.message.stop_sequence = stop_sequence
    event.message.usage = Mock()
    event.message.usage.input_tokens = 10
    event.message.usage.output_tokens = 5
    event.message.usage.cache_read_input_tokens = 0
    event.message.usage.cache_creation_input_tokens = 0
    return event


def _anthropic_text(text):
    event = Mock()
    event.type = "text"
    event.text = text
    return event


def _anthropic_tool_block(tool_id="t1", name="dummy_func"):
    event = Mock()
    event.type = "content_block_stop"
    event.content_block = Mock()
    event.content_block.type = "tool_use"
    event.content_block.id = tool_id
    event.content_block.name = name
    event.content_block.input = {"param": "value"}
    return event


@pytest.mark.parametrize(
    "raw,expected_kind",
    [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("refusal", "content_filter"),
        ("stop_sequence", "stop_sequence"),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_stream_reports_stop_reason(anthropic_llm, raw, expected_kind):
    anthropic_llm.client.messages.stream = Mock(
        return_value=_FakeAnthropicStream(
            [_anthropic_text("hello"), _anthropic_message_stop(raw)]
        )
    )

    results = [
        d
        async for d in anthropic_llm.stream(
            prompt="Test prompt", model="claude", history=[]
        )
    ]

    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert len(stops) == 1
    assert stops[0].kind == expected_kind
    assert stops[0].raw == raw


@pytest.mark.asyncio
async def test_anthropic_carries_stop_sequence_through_the_stream(anthropic_llm):
    anthropic_llm.client.messages.stream = Mock(
        return_value=_FakeAnthropicStream(
            [_anthropic_message_stop("stop_sequence", stop_sequence="```")]
        )
    )

    results = [
        d
        async for d in anthropic_llm.stream(
            prompt="Test prompt", model="claude", history=[]
        )
    ]

    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert stops[0].stop_sequence == "```"


@pytest.mark.asyncio
async def test_anthropic_truncated_tool_call_is_not_emitted(anthropic_llm):
    """Tool blocks collected but cut off by max_tokens must not be handed on.

    Anthropic only flushes tool blocks on a "tool_use" stop, so before this they
    were silently dropped and the turn simply ended.
    """
    anthropic_llm.client.messages.stream = Mock(
        return_value=_FakeAnthropicStream(
            [
                _anthropic_text("let me look that up"),
                _anthropic_tool_block(),
                _anthropic_message_stop("max_tokens"),
            ]
        )
    )

    results = [
        d
        async for d in anthropic_llm.stream(
            prompt="Test prompt", model="claude", history=[]
        )
    ]

    assert not [d for d in results if isinstance(d.content, DeltaToolUse)]
    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert stops[0].kind == "malformed_tool_call"
    assert stops[0].raw == "max_tokens"


@pytest.mark.asyncio
async def test_anthropic_complete_tool_call_still_emitted(anthropic_llm):
    anthropic_llm.client.messages.stream = Mock(
        return_value=_FakeAnthropicStream(
            [_anthropic_tool_block(), _anthropic_message_stop("tool_use")]
        )
    )

    results = [
        d
        async for d in anthropic_llm.stream(
            prompt="Test prompt", model="claude", history=[]
        )
    ]

    tool_uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.name == "dummy_func"


def test_google_raw_finish_reason_rejects_unusable_values():
    # Neither a string nor a named enum member: return None rather than
    # stringifying, so a garbled value can't normalise to "unknown" and raise.
    assert _raw_finish_reason(object()) is None
    assert _raw_finish_reason(None) is None
    assert _raw_finish_reason("") is None


# --- OpenAI-compatible gateways: native_finish_reason ----------------------
#
# OpenRouter and similar gateways normalise the upstream finish reason, and will
# report a planned "tool_calls" for a call the upstream actually cut off at the
# token limit. Verified live: finish_reason='tool_calls' arriving alongside
# native_finish_reason='length', with the tool arguments truncated to '{"city":"New'.
# Without consulting the native value the truncated call is flushed as complete.


@pytest.mark.parametrize(
    "raw,native,pending,expected_kind,expected_raw",
    [
        # The masked case: gateway says planned, upstream says truncated.
        ("tool_calls", "length", True, "malformed_tool_call", "length"),
        ("stop", "length", False, "max_tokens", "length"),
        # Non-OpenAI upstreams reached through the same gateway.
        ("tool_calls", "max_tokens", True, "malformed_tool_call", "max_tokens"),
        ("stop", "MAX_TOKENS", False, "max_tokens", "MAX_TOKENS"),
        ("stop", "model_length", False, "max_tokens", "model_length"),
        ("stop", "SAFETY", False, "content_filter", "SAFETY"),
        # Agreement, and plain OpenAI (no native field): unchanged.
        ("stop", "stop", False, "end_turn", "stop"),
        ("length", None, False, "max_tokens", "length"),
        ("tool_calls", None, True, "tool_use", "tool_calls"),
    ],
)
def test_native_finish_reason_wins_when_unplanned(
    raw, native, pending, expected_kind, expected_raw
):
    stop = _map_openai_stop_reason(raw, native=native, pending_tools=pending)
    assert stop is not None
    assert stop.kind == expected_kind
    assert stop.raw == expected_raw


def test_native_finish_reason_never_downgrades_an_unplanned_stop():
    # Normalisation only ever hides a truncation, so a planned native value must
    # not talk an unplanned primary reason back down to planned.
    stop = _map_openai_stop_reason("length", native="stop")
    assert stop is not None
    assert stop.kind == "max_tokens"


@pytest.mark.asyncio
async def test_openai_gateway_masked_truncation_is_caught(openai_llm):
    """End-to-end through the adapter: the truncated call must not be flushed."""

    def dummy_func(city: str, country: str) -> str:
        """Dummy function for testing"""
        return "result"

    function = Mock()
    function.name = "dummy_func"
    function.arguments = '{"city":"New'  # truncated, as observed live

    call_chunk = Mock()
    call_chunk.choices = [Mock()]
    call_chunk.choices[0].delta = Mock()
    call_chunk.choices[0].delta.content = None
    call_chunk.choices[0].delta.tool_calls = [
        Mock(id="tool_1", type="function", function=function)
    ]
    call_chunk.choices[0].finish_reason = None
    call_chunk.usage = None

    stop_chunk = Mock()
    stop_chunk.choices = [Mock()]
    stop_chunk.choices[0].delta = Mock()
    stop_chunk.choices[0].delta.content = None
    stop_chunk.choices[0].delta.tool_calls = None
    # The gateway reports a planned reason; the upstream truth rides alongside.
    stop_chunk.choices[0].finish_reason = "tool_calls"
    stop_chunk.choices[0].native_finish_reason = "length"
    stop_chunk.usage = None

    openai_llm.client.chat.completions.create = AsyncMock(
        return_value=to_async_response([call_chunk, stop_chunk])
    )

    results = [
        d
        async for d in openai_llm.stream(
            prompt="Test prompt", model="gpt-4o-mini", history=[],
            functions=[to_schema(dummy_func)],
        )
    ]

    assert not [d for d in results if isinstance(d.content, DeltaToolUse)]
    stops = [d.stop_reason for d in results if d.stop_reason is not None]
    assert stops[-1].kind == "malformed_tool_call"
    assert stops[-1].raw == "length"
