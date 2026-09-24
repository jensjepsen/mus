import logging
import typing as t
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from mus import Bot, OpenAIResponsesLLM
from mus.converters.delta import delta_converter
from mus.functions import to_schema
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMContextLengthExceededException,
    LLMRateLimitException,
    LLMServerException,
    LLMToolParseException,
)
from mus.llm.openai_responses import (
    REASONING_METADATA_KEY,
    _map_response_stop_reason,
    deltas_to_input,
    func_to_tool,
    query_to_input,
)
from mus.llm.types import (
    Assistant,
    CachePoint,
    Delta,
    DeltaText,
    DeltaToolInputUpdate,
    DeltaToolResult,
    DeltaToolUse,
    File,
    Query,
    ToolResult,
    ToolUse,
    ToolValue,
)

MODEL = "gpt-5-mini"


async def to_async_response(seq: t.Sequence[t.Any]) -> t.AsyncGenerator[t.Any, None]:
    for item in seq:
        yield item


def ev(type_: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **fields)


def usage(input_tokens=10, output_tokens=5, cached=0) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=input_tokens + output_tokens,
    )


def response(status="completed", output=(), incomplete=None, error=None) -> Response:
    return Response.model_construct(
        id="resp_1",
        status=status,
        output=list(output),
        usage=usage(),
        incomplete_details=IncompleteDetails(reason=incomplete) if incomplete else None,
        error=error,
    )


def fcall(call_id="call_1", name="look_up", arguments='{"q": "x"}', item_id="fc_1"):
    return ResponseFunctionToolCall(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
    )


def reasoning(item_id="rs_1", encrypted="ENC", summary=()):
    return ResponseReasoningItem(
        type="reasoning",
        id=item_id,
        summary=[Summary(type="summary_text", text=s) for s in summary],
        encrypted_content=encrypted,
    )


def call_events(call: ResponseFunctionToolCall, index: int, fragments: t.Sequence[str]):
    return [
        ev("response.output_item.added", item=call, output_index=index),
        *[
            ev(
                "response.function_call_arguments.delta",
                item_id=call.id,
                output_index=index,
                delta=f,
            )
            for f in fragments
        ],
        ev("response.output_item.done", item=call, output_index=index),
    ]


@pytest.fixture
def client():
    client = AsyncMock(spec=openai.AsyncClient)
    client.responses = AsyncMock()
    client.responses.create = AsyncMock()
    return client


@pytest.fixture
def llm(client):
    return OpenAIResponsesLLM(MODEL, client)


async def look_up(q: str) -> str:
    """Look something up"""
    return f"result for {q}"


async def collect(llm, **kwargs) -> t.List[Delta]:
    kwargs.setdefault("prompt", "sys")
    kwargs.setdefault("history", [])
    return [d async for d in llm.stream(**kwargs)]


# --- request mapping --------------------------------------------------------


def test_func_to_tool_is_flat():
    tool = func_to_tool(to_schema(look_up))
    assert tool["type"] == "function"
    assert tool["name"] == "look_up"
    assert tool["strict"] is False
    assert "function" not in tool
    assert tool["parameters"]["properties"]["q"]["type"] == "string"


def test_query_to_input():
    img = File(b64type="image/png", content="AAAA")
    items = query_to_input(Query(["hi", CachePoint(), img, Assistant("hello")]))
    assert items == [
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "auto",
                }
            ],
        },
        {"role": "assistant", "content": "hello"},
    ]


def _tool_use_delta(call_id, metadata=None):
    return Delta(
        content=DeltaToolUse(data=ToolUse(id=call_id, name="look_up", input={"q": "x"})),
        metadata=metadata,
    )


def _tool_result_delta(call_id, val):
    return Delta(
        content=DeltaToolResult(data=ToolResult(id=call_id, content=ToolValue(val)))
    )


def test_deltas_to_input_tool_round_trip_and_reasoning_replay():
    meta = {
        REASONING_METADATA_KEY: {
            "model": MODEL,
            "items": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}],
        }
    }
    history = [
        Query("q"),
        Delta(content=DeltaText(data="thinking...", subtype="reasoning")),
        _tool_use_delta("call_1", meta),
        _tool_use_delta("call_2"),
        _tool_result_delta("call_1", "a"),
        _tool_result_delta("call_2", "b"),
        Delta(content=DeltaText(data="done")),
    ]
    items = deltas_to_input(history, model=MODEL)
    assert items == [
        {"role": "user", "content": "q"},
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
        {"type": "function_call", "call_id": "call_1", "name": "look_up", "arguments": '{"q": "x"}'},
        {"type": "function_call", "call_id": "call_2", "name": "look_up", "arguments": '{"q": "x"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "a"},
        {"type": "function_call_output", "call_id": "call_2", "output": "b"},
        {"role": "assistant", "content": "done"},
    ]


def test_deltas_to_input_joins_streamed_text_chunks():
    """Within a turn the Bot passes history unmerged, one delta per chunk."""
    history = [
        Query("q"),
        Delta(content=DeltaText(data="I'll ")),
        Delta(content=DeltaText(data="thinking", subtype="reasoning")),
        Delta(content=DeltaText(data="look ")),
        Delta(content=DeltaText(data="")),
        Delta(content=DeltaText(data="that up.")),
        _tool_use_delta("call_1"),
        _tool_result_delta("call_1", "a"),
        Delta(content=DeltaText(data="It ")),
        Delta(content=DeltaText(data="is a.")),
        Query("thanks"),
        Delta(content=DeltaText(data="You're welcome.")),
    ]
    items = deltas_to_input(history, model=MODEL)
    assert [i.get("type", i.get("role")) for i in items] == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
        "assistant",
        "user",
        "assistant",
    ]
    assert items[1]["content"] == "I'll look that up."
    assert items[4]["content"] == "It is a."
    assert items[6]["content"] == "You're welcome."


def test_deltas_to_input_drops_reasoning_from_another_model():
    meta = {REASONING_METADATA_KEY: {"model": "o3", "items": [{"type": "reasoning", "id": "rs_1"}]}}
    items = deltas_to_input([_tool_use_delta("call_1", meta)], model=MODEL)
    assert [i["type"] for i in items] == ["function_call"]


def test_deltas_to_input_tool_images_follow_the_run_of_outputs():
    img = File(b64type="image/png", content="AAAA")
    items = deltas_to_input(
        [
            _tool_use_delta("call_1"),
            _tool_use_delta("call_2"),
            _tool_result_delta("call_1", img),
            _tool_result_delta("call_2", "text"),
        ]
    )
    kinds = [i.get("type", i.get("role")) for i in items]
    assert kinds == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "user",
    ]
    assert items[2]["output"].startswith("[Tool returned image(s)")
    assert items[4]["content"][1]["type"] == "input_image"


# --- stop reasons -----------------------------------------------------------


@pytest.mark.parametrize(
    "status,reason,tools,refused,kind",
    [
        ("completed", None, False, False, "end_turn"),
        ("completed", None, True, False, "tool_use"),
        ("completed", None, False, True, "content_filter"),
        ("incomplete", "max_output_tokens", False, False, "max_tokens"),
        ("incomplete", "max_output_tokens", True, False, "malformed_tool_call"),
        ("incomplete", "content_filter", False, False, "content_filter"),
        ("failed", None, False, False, "error"),
        ("something_new", None, False, False, "unknown"),
    ],
)
def test_map_response_stop_reason(status, reason, tools, refused, kind):
    stop = _map_response_stop_reason(status, reason, has_tool_calls=tools, refused=refused)
    assert stop is not None and stop.kind == kind


def test_map_response_stop_reason_none():
    assert _map_response_stop_reason(None) is None


# --- streaming --------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_text(llm, client):
    client.responses.create.return_value = to_async_response(
        [
            ev("response.output_text.delta", delta="Hello"),
            ev("response.output_text.delta", delta=" world"),
            ev("response.completed", response=response()),
        ]
    )
    results = await collect(llm, max_tokens=100, temperature=0.0)

    assert "".join(d.content.data for d in results if isinstance(d.content, DeltaText)) == "Hello world"
    assert results[-1].stop_reason.kind == "end_turn"
    assert results[-1].usage.input_tokens == 10

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == "sys"
    assert kwargs["store"] is False
    assert kwargs["stream"] is True
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["max_output_tokens"] == 100
    # 0.0 is a real temperature, not "unset".
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_stream_cached_tokens(llm, client):
    r = response()
    r.usage = usage(cached=7)
    client.responses.create.return_value = to_async_response(
        [ev("response.completed", response=r)]
    )
    results = await collect(llm)
    assert results[-1].usage.cache_read_input_tokens == 7


@pytest.mark.asyncio
async def test_stream_parallel_tool_calls_with_reasoning(llm, client):
    first = fcall("call_1", arguments='{"q": "a"}', item_id="fc_1")
    second = fcall("call_2", arguments='{"q": "b"}', item_id="fc_2")
    client.responses.create.return_value = to_async_response(
        [
            ev("response.reasoning_summary_text.delta", delta="Plan"),
            ev("response.output_item.done", item=reasoning(summary=["Plan"]), output_index=0),
            *call_events(first, 1, ['{"q": ', '"a"}']),
            *call_events(second, 2, ['{"q": "b"}']),
            ev("response.completed", response=response()),
        ]
    )
    results = await collect(llm, functions=[to_schema(look_up)])

    reasoning_text = [d for d in results if isinstance(d.content, DeltaText) and d.content.subtype == "reasoning"]
    assert reasoning_text[0].content.data == "Plan"

    updates = [d.content for d in results if isinstance(d.content, DeltaToolInputUpdate)]
    assert [(u.id, u.name, u.data) for u in updates] == [
        ("call_1", "look_up", '{"q": '),
        ("call_1", "look_up", '"a"}'),
        ("call_2", "look_up", '{"q": "b"}'),
    ]

    uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert [u.content.data.id for u in uses] == ["call_1", "call_2"]
    assert [u.content.data.input for u in uses] == [{"q": "a"}, {"q": "b"}]
    # The reasoning belongs to the call it preceded, not to both.
    replay = uses[0].metadata[REASONING_METADATA_KEY]
    assert replay["model"] == MODEL
    assert replay["items"][0]["encrypted_content"] == "ENC"
    assert uses[1].metadata is None

    assert results[-1].stop_reason.kind == "tool_use"


@pytest.mark.asyncio
async def test_stream_reasoning_without_encrypted_content_is_not_replayed(llm, client):
    client.responses.create.return_value = to_async_response(
        [
            ev("response.output_item.done", item=reasoning(encrypted=None), output_index=0),
            *call_events(fcall(), 1, ['{"q": "x"}']),
            ev("response.completed", response=response()),
        ]
    )
    results = await collect(llm, functions=[to_schema(look_up)])
    uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert uses[0].metadata is None


@pytest.mark.asyncio
async def test_stream_truncated_tool_call_is_not_flushed(llm, client):
    call = fcall(arguments='{"q": "ne')
    client.responses.create.return_value = to_async_response(
        [
            ev("response.output_item.added", item=call, output_index=0),
            ev("response.function_call_arguments.delta", item_id=call.id, output_index=0, delta='{"q": "ne'),
            ev("response.incomplete", response=response("incomplete", incomplete="max_output_tokens")),
        ]
    )
    results = await collect(llm, functions=[to_schema(look_up)])
    assert not [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert results[-1].stop_reason.kind == "malformed_tool_call"
    assert results[-1].stop_reason.raw == "max_output_tokens"


@pytest.mark.asyncio
async def test_stream_refusal_is_content_filter(llm, client):
    client.responses.create.return_value = to_async_response(
        [
            ev("response.refusal.delta", delta="I can't help with that"),
            ev("response.completed", response=response()),
        ]
    )
    results = await collect(llm)
    assert results[-1].stop_reason.kind == "content_filter"
    assert results[-1].stop_reason.raw == "refusal"


@pytest.mark.asyncio
async def test_stream_error_event_context_length(llm, client):
    client.responses.create.return_value = to_async_response(
        [ev("error", code="context_length_exceeded", message="too long", param=None)]
    )
    with pytest.raises(LLMContextLengthExceededException):
        await collect(llm)


async def _raising_stream(exc: Exception):
    yield ev("response.created", response=response("in_progress"))
    raise exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,message,expected",
    [
        (
            "context_length_exceeded",
            "Your input exceeds the context window of this model.",
            LLMContextLengthExceededException,
        ),
        # Recognised by message when the code is missing.
        (None, "Your input exceeds the context window of this model.", LLMContextLengthExceededException),
        ("rate_limit_exceeded", "slow down", LLMRateLimitException),
        ("server_error", "oops", LLMServerException),
    ],
)
async def test_stream_sdk_raised_error_event(llm, client, code, message, expected):
    """The SDK raises an in-stream ``error`` event as a bare APIError with no
    HTTP status; its code must still pick the exception."""
    request = Mock(spec=httpx.Request)
    sdk_error = openai.APIError(message, request, body={"code": code, "message": message})
    client.responses.create.return_value = _raising_stream(sdk_error)
    with pytest.raises(expected) as exc_info:
        await collect(llm)
    assert exc_info.value.__cause__ is sdk_error


@pytest.mark.asyncio
async def test_stream_failed_response_rate_limit(llm, client):
    failed = response(
        "failed",
        error=ResponseError.model_construct(code="rate_limit_exceeded", message="slow down"),
    )
    client.responses.create.return_value = to_async_response(
        [ev("response.failed", response=failed)]
    )
    with pytest.raises(LLMRateLimitException):
        await collect(llm)


@pytest.mark.asyncio
async def test_stream_ending_without_terminal_event(llm, client):
    client.responses.create.return_value = to_async_response(
        [ev("response.output_text.delta", delta="Hel")]
    )
    with pytest.raises(LLMConnectionException):
        await collect(llm)


@pytest.mark.asyncio
async def test_stream_malformed_tool_json(llm, client):
    client.responses.create.return_value = to_async_response(
        [
            *call_events(fcall(arguments="not json at all"), 0, ["not json at all"]),
            ev("response.completed", response=response()),
        ]
    )
    with pytest.raises(LLMToolParseException):
        await collect(llm, functions=[to_schema(look_up)])


@pytest.mark.asyncio
async def test_stream_repairs_tool_json(llm, client):
    client.responses.create.return_value = to_async_response(
        [
            *call_events(fcall(arguments='{"q": "x",}'), 0, ['{"q": "x",}']),
            ev("response.completed", response=response()),
        ]
    )
    results = await collect(llm, functions=[to_schema(look_up)])
    uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert uses[0].content.data.input == {"q": "x"}


# --- non-streaming ----------------------------------------------------------


@pytest.mark.asyncio
async def test_non_stream(llm, client):
    message = ResponseOutputMessage(
        type="message",
        id="msg_1",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="Let me check", annotations=[])],
    )
    client.responses.create.return_value = response(
        output=[reasoning(summary=["Plan"]), message, fcall()]
    )
    results = await collect(llm, no_stream=True, functions=[to_schema(look_up)])

    texts = [(d.content.subtype, d.content.data) for d in results if isinstance(d.content, DeltaText) and d.content.data]
    assert texts == [("reasoning", "Plan"), ("text", "Let me check")]
    uses = [d for d in results if isinstance(d.content, DeltaToolUse)]
    assert uses[0].content.data.id == "call_1"
    assert uses[0].metadata[REASONING_METADATA_KEY]["items"][0]["id"] == "rs_1"
    assert results[-1].stop_reason.kind == "tool_use"
    assert client.responses.create.call_args.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_non_stream_refusal(llm, client):
    message = ResponseOutputMessage(
        type="message",
        id="msg_1",
        role="assistant",
        status="completed",
        content=[ResponseOutputRefusal(type="refusal", refusal="no")],
    )
    client.responses.create.return_value = response(output=[message])
    results = await collect(llm, no_stream=True)
    assert results[-1].stop_reason.kind == "content_filter"


# --- request options and errors --------------------------------------------


@pytest.mark.asyncio
async def test_stop_sequences_are_ignored_with_a_warning(llm, client, caplog):
    client.responses.create.return_value = to_async_response(
        [ev("response.completed", response=response())]
    )
    with caplog.at_level(logging.WARNING, logger="mus.llm.openai_responses"):
        await collect(llm, stop_sequences=["END"])
    assert "stop sequences" in caplog.text
    assert "stop" not in client.responses.create.call_args.kwargs


@pytest.mark.asyncio
async def test_include_reasoning_can_be_turned_off(client):
    llm = OpenAIResponsesLLM(MODEL, client, include_reasoning=False)
    client.responses.create.return_value = to_async_response(
        [ev("response.completed", response=response())]
    )
    await collect(llm)
    assert not isinstance(client.responses.create.call_args.kwargs["include"], list)


@pytest.mark.asyncio
async def test_caller_include_is_kept(llm, client):
    client.responses.create.return_value = to_async_response(
        [ev("response.completed", response=response())]
    )
    await collect(llm, kwargs={"include": ["message.output_text.logprobs"]})
    assert client.responses.create.call_args.kwargs["include"] == [
        "message.output_text.logprobs",
        "reasoning.encrypted_content",
    ]


@pytest.mark.asyncio
async def test_create_error_is_mapped(llm, client):
    resp = Mock(spec=httpx.Response)
    resp.status_code = 401
    resp.headers = httpx.Headers({})
    resp.request = Mock(spec=httpx.Request)
    client.responses.create.side_effect = openai.AuthenticationError(
        "bad key", response=resp, body=None
    )
    with pytest.raises(LLMAuthenticationException):
        await collect(llm)


# --- persistence and end to end --------------------------------------------


def test_reasoning_metadata_survives_serialization():
    delta = _tool_use_delta(
        "call_1",
        {REASONING_METADATA_KEY: {"model": MODEL, "items": [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}]}},
    )
    restored = delta_converter.structure(delta_converter.unstructure(delta), Delta)
    assert deltas_to_input([restored], model=MODEL)[0]["encrypted_content"] == "ENC"


@pytest.mark.asyncio
async def test_bot_replays_reasoning_after_tool_call(llm, client):
    client.responses.create.side_effect = [
        to_async_response(
            [
                ev("response.output_item.done", item=reasoning(), output_index=0),
                *call_events(fcall(), 1, ['{"q": "x"}']),
                ev("response.completed", response=response()),
            ]
        ),
        to_async_response(
            [
                ev("response.output_text.delta", delta="The answer"),
                ev("response.completed", response=response()),
            ]
        ),
    ]
    bot = Bot(prompt="sys", model=llm, functions=[look_up])
    assert await bot("look up x").string() is not None

    follow_up = client.responses.create.call_args_list[1].kwargs["input"]
    kinds = [i.get("type", i.get("role")) for i in follow_up]
    assert kinds == ["user", "reasoning", "function_call", "function_call_output"]
    assert follow_up[1]["encrypted_content"] == "ENC"
    assert follow_up[3]["output"] == "result for x"
