import pytest
from unittest.mock import Mock, AsyncMock
import json
import typing as t

import httpx
import openai

from mus.llm.openai import OpenAILLM, _map_openai_exception
from mus.llm.types import (
    Delta,
    DeltaText,
    DeltaToolUse,
    DeltaToolInputUpdate,
    Query,
    ToolUse,
    Usage,
)
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMRateLimitException,
    LLMConnectionException,
    LLMTimeoutException,
    LLMBadRequestException,
    LLMServerException,
    LLMNotFoundException,
    LLMToolParseException,
    LLMException,
)
from mus.functions import to_schema


ASYNC_T = t.TypeVar("ASYNC_T")


async def to_async_response(seq: t.Sequence[ASYNC_T]) -> t.AsyncGenerator[ASYNC_T, None]:
    """Convert a sequence to an async generator."""
    for item in seq:
        yield item


@pytest.fixture
def mock_openai_client():
    client = AsyncMock(spec=openai.AsyncClient)
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def openai_llm(mock_openai_client):
    return OpenAILLM("gpt-4", mock_openai_client)


def _make_httpx_response(status_code, headers=None):
    """Helper to create a mock httpx.Response."""
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    resp.request = Mock(spec=httpx.Request)
    return resp


def _make_openai_status_error(cls, status_code, headers=None):
    """Helper to create an OpenAI APIStatusError subclass."""
    resp = _make_httpx_response(status_code, headers)
    return cls(
        message=f"Error {status_code}",
        response=resp,
        body=None,
    )


# --- Direct mapping function tests ---

def test_map_openai_auth_error():
    exc = _make_openai_status_error(openai.AuthenticationError, 401, headers={"x-request-id": "req-001"})
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 401
    assert mapped.request_id == "req-001"


def test_map_openai_rate_limit_with_retry_after():
    exc = _make_openai_status_error(openai.RateLimitError, 429, headers={"retry-after": "25", "x-request-id": "req-002"})
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 429
    assert mapped.retry_after == 25.0
    assert mapped.request_id == "req-002"


def test_map_openai_rate_limit_no_retry_after():
    exc = _make_openai_status_error(openai.RateLimitError, 429)
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.retry_after is None


def test_map_openai_timeout_error():
    exc = openai.APITimeoutError(request=Mock(spec=httpx.Request))
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMTimeoutException)
    assert mapped.provider == "openai"


def test_map_openai_connection_error():
    exc = openai.APIConnectionError(request=Mock(spec=httpx.Request))
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMConnectionException)
    assert mapped.provider == "openai"


def test_map_openai_not_found_error():
    exc = _make_openai_status_error(openai.NotFoundError, 404)
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMNotFoundException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 404


def test_map_openai_bad_request_error():
    exc = _make_openai_status_error(openai.BadRequestError, 400)
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 400


def test_map_openai_unprocessable_entity_error():
    exc = _make_openai_status_error(openai.UnprocessableEntityError, 422)
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 422


def test_map_openai_internal_server_error():
    exc = _make_openai_status_error(openai.InternalServerError, 500)
    mapped = _map_openai_exception(exc)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "openai"
    assert mapped.status_code == 500


# --- Basic streaming test ---

@pytest.mark.asyncio
async def test_openai_stream_basic(openai_llm, mock_openai_client):
    """Basic streaming test to verify OpenAILLM works."""
    mock_chunk = Mock()
    mock_chunk.choices = [Mock()]
    mock_chunk.choices[0].delta = Mock()
    mock_chunk.choices[0].delta.content = "Hello world"
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.choices[0].finish_reason = None
    mock_chunk.usage = None

    mock_openai_client.chat.completions.create.return_value = to_async_response([mock_chunk])

    results = []
    async for delta in openai_llm.stream(prompt="p", history=[], model="m"):
        results.append(delta)

    assert len(results) == 1
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Hello world"


# --- Exception handling tests ---

@pytest.mark.asyncio
async def test_openai_auth_error(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(openai.AuthenticationError, 401)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_rate_limit_error_with_retry_after(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(
        openai.RateLimitError, 429,
        headers={"retry-after": "30"},
    )
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_connection_error(openai_llm, mock_openai_client):
    exc = openai.APIConnectionError(request=Mock(spec=httpx.Request))
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMConnectionException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_timeout_error(openai_llm, mock_openai_client):
    exc = openai.APITimeoutError(request=Mock(spec=httpx.Request))
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMTimeoutException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_server_error(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(openai.InternalServerError, 500)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_openai_not_found_error(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(openai.NotFoundError, 404)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMNotFoundException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_openai_bad_request_error(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(openai.BadRequestError, 400)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_openai_unprocessable_entity_error(openai_llm, mock_openai_client):
    exc = _make_openai_status_error(openai.UnprocessableEntityError, 422)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_openai_stream_iteration_error(openai_llm, mock_openai_client):
    """SDK exception during stream iteration is mapped correctly."""
    exc = _make_openai_status_error(openai.InternalServerError, 500)

    async def failing_stream():
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "partial"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = None
        mock_chunk.usage = None
        yield mock_chunk
        raise exc

    mock_openai_client.chat.completions.create.return_value = failing_stream()

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_tool_parse_error_streaming(openai_llm, mock_openai_client):
    """Malformed tool JSON during streaming raises LLMToolParseException."""
    # Chunk that starts a tool call
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.content = None
    tc = Mock()
    tc.id = "call_1"
    tc.function = Mock()
    tc.function.name = "my_tool"
    tc.function.arguments = "{invalid json"
    chunk1.choices[0].delta.tool_calls = [tc]
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    # Chunk that ends with tool_calls finish reason
    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.content = None
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "tool_calls"
    chunk2.usage = None

    mock_openai_client.chat.completions.create.return_value = to_async_response([chunk1, chunk2])

    with pytest.raises(LLMToolParseException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_openai_tool_parse_error_non_streaming(openai_llm, mock_openai_client):
    """Malformed tool JSON in non-streaming mode raises LLMToolParseException."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_tc = Mock()
    mock_tc.id = "call_1"
    mock_tc.type = "function"
    mock_tc.function = Mock()
    mock_tc.function.name = "my_tool"
    mock_tc.function.arguments = "{bad json"
    mock_response.choices[0].message.tool_calls = [mock_tc]
    mock_response.usage = None

    mock_openai_client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMToolParseException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m", no_stream=True):
            pass
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_openai_exception_chaining(openai_llm, mock_openai_client):
    """Verify __cause__ is the original SDK exception."""
    exc = _make_openai_status_error(openai.AuthenticationError, 401)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_openai_rate_limit_without_retry_after(openai_llm, mock_openai_client):
    """Rate limit error without retry-after header still works."""
    exc = _make_openai_status_error(openai.RateLimitError, 429)
    mock_openai_client.chat.completions.create.side_effect = exc

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in openai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.retry_after is None


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_json,expected_args", [
    ('{"name": "test", "value": 42,}', {"name": "test", "value": 42}),
    ("{'name': 'test', 'value': 42}", {"name": "test", "value": 42}),
    ('{"name": "test", "value": 42', {"name": "test", "value": 42}),
    ('{name: "test", value: 42}', {"name": "test", "value": 42}),
])
async def test_openai_json_repair_streaming(openai_llm, mock_openai_client, malformed_json, expected_args):
    """Malformed but repairable tool JSON is repaired during streaming."""
    chunk1 = Mock()
    chunk1.choices = [Mock()]
    chunk1.choices[0].delta = Mock()
    chunk1.choices[0].delta.content = None
    tc = Mock()
    tc.id = "call_1"
    tc.function = Mock()
    tc.function.name = "my_tool"
    tc.function.arguments = malformed_json
    chunk1.choices[0].delta.tool_calls = [tc]
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    chunk2 = Mock()
    chunk2.choices = [Mock()]
    chunk2.choices[0].delta = Mock()
    chunk2.choices[0].delta.content = None
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "tool_calls"
    chunk2.usage = None

    mock_openai_client.chat.completions.create.return_value = to_async_response([chunk1, chunk2])

    deltas = []
    async for d in openai_llm.stream(prompt="p", history=[], model="m"):
        deltas.append(d)

    tool_uses = [d for d in deltas if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.input == expected_args


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_json,expected_args", [
    ('{"name": "test", "value": 42,}', {"name": "test", "value": 42}),
    ("{'name': 'test', 'value': 42}", {"name": "test", "value": 42}),
    ('{"name": "test", "value": 42', {"name": "test", "value": 42}),
    ('{name: "test", value: 42}', {"name": "test", "value": 42}),
])
async def test_openai_json_repair_non_streaming(openai_llm, mock_openai_client, malformed_json, expected_args):
    """Malformed but repairable tool JSON is repaired in non-streaming mode."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_tc = Mock()
    mock_tc.id = "call_1"
    mock_tc.type = "function"
    mock_tc.function = Mock()
    mock_tc.function.name = "my_tool"
    mock_tc.function.arguments = malformed_json
    mock_response.choices[0].message.tool_calls = [mock_tc]
    mock_response.usage = None

    mock_openai_client.chat.completions.create.return_value = mock_response

    deltas = []
    async for d in openai_llm.stream(prompt="p", history=[], model="m", no_stream=True):
        deltas.append(d)

    tool_uses = [d for d in deltas if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.input == expected_args


# --- Cached token tests ---

@pytest.mark.asyncio
async def test_openai_stream_cached_tokens(openai_llm, mock_openai_client):
    """Streaming: cached tokens from prompt_tokens_details are reported."""
    mock_chunk = Mock()
    mock_chunk.choices = [Mock()]
    mock_chunk.choices[0].delta = Mock()
    mock_chunk.choices[0].delta.content = "Hello"
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.choices[0].finish_reason = None
    mock_chunk.usage = None

    usage_chunk = Mock()
    usage_chunk.choices = []
    usage_chunk.usage = Mock()
    usage_chunk.usage.prompt_tokens = 100
    usage_chunk.usage.completion_tokens = 20
    usage_chunk.usage.prompt_tokens_details = Mock()
    usage_chunk.usage.prompt_tokens_details.cached_tokens = 80

    mock_openai_client.chat.completions.create.return_value = to_async_response(
        [mock_chunk, usage_chunk]
    )

    results = []
    async for delta in openai_llm.stream(prompt="p", history=[], model="m"):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.input_tokens == 100
    assert usage_deltas[0].usage.output_tokens == 20
    assert usage_deltas[0].usage.cache_read_input_tokens == 80


@pytest.mark.asyncio
async def test_openai_stream_no_prompt_tokens_details(openai_llm, mock_openai_client):
    """Streaming: missing prompt_tokens_details defaults to 0 cached tokens."""
    usage_chunk = Mock()
    usage_chunk.choices = []
    usage_chunk.usage = Mock()
    usage_chunk.usage.prompt_tokens = 50
    usage_chunk.usage.completion_tokens = 10
    usage_chunk.usage.prompt_tokens_details = None

    mock_openai_client.chat.completions.create.return_value = to_async_response(
        [usage_chunk]
    )

    results = []
    async for delta in openai_llm.stream(prompt="p", history=[], model="m"):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_openai_non_stream_cached_tokens(openai_llm, mock_openai_client):
    """Non-streaming: cached tokens from prompt_tokens_details are reported."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Hello"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.prompt_tokens_details = Mock()
    mock_response.usage.prompt_tokens_details.cached_tokens = 60

    mock_openai_client.chat.completions.create.return_value = mock_response

    results = []
    async for delta in openai_llm.stream(prompt="p", history=[], model="m", no_stream=True):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.input_tokens == 100
    assert usage_deltas[0].usage.output_tokens == 20
    assert usage_deltas[0].usage.cache_read_input_tokens == 60
