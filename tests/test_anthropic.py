from dataclasses import dataclass
from typing import Annotated, List, Optional
import pytest
from unittest.mock import Mock
from contextlib import asynccontextmanager

from mus.llm.anthropic import func_to_tool, functions_for_llm, _map_anthropic_exception, deltas_to_messages, add_history_cache_point
from mus import AnthropicLLM
from mus.llm.types import Query, CachePoint
from mus.functions import to_schema
from anthropic import types as at

def sample_function(param1: str, param2: int) -> str:
    """This is a sample function."""
    return f"{param1}: {param2}"

@dataclass
class SampleDataclass:
    """This is a sample dataclass."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"] = None


# Helper function to check if a dict matches ToolParam structure
def is_tool_param(obj):
    return isinstance(obj, dict) and all(key in obj for key in ['name', 'description', 'input_schema'])



@asynccontextmanager
async def return_val():
    """A context manager that yields a value."""
    async def iterator():
        yield at.Message(
            role="assistant",
            content=[
                at.TextBlock(type="text", text="This is a test response."),
            ],
            id="test-id",
            model="a-model-id",
            type="message",
            usage=at.Usage(
                input_tokens=10,
                output_tokens=5,
            ),
        )
    yield iterator()


@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def bedrock_llm(mock_client):
    return AnthropicLLM("a-model-id", mock_client)


@pytest.mark.asyncio
async def test_anthropic_stream(mock_client):
    async def dummy_tool(param1: str, param2: int) -> str:
        """A dummy tool for testing."""
        return f"Dummy result with {param1} and {param2}"
    expected_tool = to_schema(dummy_tool)
    mock_client.messages.stream = Mock(return_value=return_val())
    llm = AnthropicLLM("a-model-id", mock_client)
    async for response in llm.stream(
        prompt="Test prompt",
        history=[],
        functions=[expected_tool],
        no_stream=False,
        function_choice="auto",
        kwargs={},
        top_k=5,
        top_p=0.9,
        stop_sequences=["\n", "END"],
        temperature=0.7,
        ):
        assert response is None
    
    mock_client.messages.stream.assert_called_once_with(
        max_tokens=4096,
        model="a-model-id",
        messages=[],
        top_k=5,
        top_p=0.9,
        stop_sequences=["\n", "END"],
        temperature=0.7,
        tools=[{
            "name": expected_tool["name"],
            "description": expected_tool["description"],
            "input_schema": expected_tool["schema"]
        }],
        tool_choice={"type": "auto"},
        system="Test prompt",
    )


def test_func_to_tool():
    tool = func_to_tool(to_schema(sample_function))
    assert is_tool_param(tool)
    assert tool["name"] == "sample_function"
    assert tool["description"] == "This is a sample function."
    assert "param1" in tool["input_schema"]["properties"]
    assert "param2" in tool["input_schema"]["properties"]

def test_functions_for_llm():
    functions = [to_schema(sample_function), to_schema(SampleDataclass)]
    tools = functions_for_llm(functions)
    assert len(tools) == 2
    assert all(is_tool_param(tool) for tool in tools)
    assert tools[0]["name"] == "sample_function"
    assert tools[1]["name"] == "SampleDataclass"

def test_functions_for_llm_empty_input():
    tools = functions_for_llm(None)
    assert len(tools) == 0


# --- Exception handling tests ---

import httpx
import anthropic
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMRateLimitException,
    LLMConnectionException,
    LLMTimeoutException,
    LLMBadRequestException,
    LLMServerException,
    LLMNotFoundException,
    LLMException,
)


def _make_httpx_response(status_code, headers=None):
    """Helper to create a mock httpx.Response."""
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.request = Mock(spec=httpx.Request)
    return resp


def _make_anthropic_status_error(cls, status_code, headers=None, request_id="req-abc"):
    """Helper to create an Anthropic APIStatusError subclass."""
    resp = _make_httpx_response(status_code, headers)
    resp.headers.setdefault("x-request-id", request_id)
    err = cls.__new__(cls)
    # Manually set attributes like the real constructor does
    err.response = resp
    err.status_code = status_code
    err.request_id = request_id
    err.body = None
    err.message = f"Error {status_code}"
    Exception.__init__(err, f"Error {status_code}")
    return err


# --- Direct mapping function tests ---

def test_map_anthropic_auth_error():
    exc = _make_anthropic_status_error(anthropic.AuthenticationError, 401)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 401
    assert mapped.request_id == "req-abc"


def test_map_anthropic_permission_denied():
    exc = _make_anthropic_status_error(anthropic.PermissionDeniedError, 403)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 403


def test_map_anthropic_rate_limit_with_retry_after():
    exc = _make_anthropic_status_error(
        anthropic.RateLimitError, 429,
        headers={"retry-after": "45", "request-id": "req-abc"},
    )
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 429
    assert mapped.retry_after == 45.0


def test_map_anthropic_rate_limit_no_retry_after():
    exc = _make_anthropic_status_error(anthropic.RateLimitError, 429)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.retry_after is None


def test_map_anthropic_timeout_error():
    exc = anthropic.APITimeoutError(request=Mock(spec=httpx.Request))
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMTimeoutException)
    assert mapped.provider == "anthropic"


def test_map_anthropic_connection_error():
    exc = anthropic.APIConnectionError(request=Mock(spec=httpx.Request))
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMConnectionException)
    assert mapped.provider == "anthropic"


def test_map_anthropic_not_found():
    exc = _make_anthropic_status_error(anthropic.NotFoundError, 404)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMNotFoundException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 404


def test_map_anthropic_bad_request():
    exc = _make_anthropic_status_error(anthropic.BadRequestError, 400)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 400


def test_map_anthropic_unprocessable_entity():
    exc = _make_anthropic_status_error(anthropic.UnprocessableEntityError, 422)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 422


def test_map_anthropic_internal_server_error():
    exc = _make_anthropic_status_error(anthropic.InternalServerError, 500)
    mapped = _map_anthropic_exception(exc)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "anthropic"
    assert mapped.status_code == 500


@pytest.mark.asyncio
async def test_anthropic_auth_error(mock_client):
    exc = _make_anthropic_status_error(anthropic.AuthenticationError, 401)
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_anthropic_rate_limit_error_with_retry_after(mock_client):
    exc = _make_anthropic_status_error(
        anthropic.RateLimitError, 429,
        headers={"retry-after": "30", "x-request-id": "req-abc"},
    )
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_anthropic_connection_error(mock_client):
    exc = anthropic.APIConnectionError(request=Mock(spec=httpx.Request))
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMConnectionException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_anthropic_server_error(mock_client):
    exc = _make_anthropic_status_error(anthropic.InternalServerError, 500)
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_anthropic_not_found_error(mock_client):
    exc = _make_anthropic_status_error(anthropic.NotFoundError, 404)
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMNotFoundException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_anthropic_bad_request_error(mock_client):
    exc = _make_anthropic_status_error(anthropic.BadRequestError, 400)
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_anthropic_stream_iteration_error(mock_client):
    """SDK exception during stream iteration is mapped correctly."""
    exc = _make_anthropic_status_error(anthropic.InternalServerError, 500)

    @asynccontextmanager
    async def failing_stream():
        async def iterator():
            raise exc
            yield  # make it a generator
        yield iterator()

    mock_client.messages.stream = Mock(return_value=failing_stream())
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_anthropic_exception_chaining(mock_client):
    """Verify __cause__ is the original SDK exception."""
    exc = _make_anthropic_status_error(anthropic.AuthenticationError, 401)
    mock_client.messages.stream = Mock(side_effect=exc)
    llm = AnthropicLLM("a-model-id", mock_client)

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


def test_cache_point_tags_preceding_block():
    messages = deltas_to_messages([Query(["big context", CachePoint(), "the question"])])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # The cache point splits the otherwise-merged text into two blocks; the
    # block before the cache point carries cache_control, the one after does not.
    assert len(content) == 2
    assert content[0]["text"] == "big context"
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["text"] == "the question"
    assert "cache_control" not in content[1]


def test_cache_point_trailing_tags_last_block():
    messages = deltas_to_messages([Query(["only", CachePoint()])])
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["text"] == "only"
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_point_ttl_maps_to_cache_control():
    messages = deltas_to_messages([Query(["doc", CachePoint(ttl="1h"), "q"])])
    content = messages[0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_cache_point_leading_is_noop():
    messages = deltas_to_messages([Query([CachePoint(), "hello"])])
    assert len(messages) == 1
    content = messages[0]["content"]
    assert all("cache_control" not in block for block in content)
    assert any(block.get("text") == "hello" for block in content)


def test_no_cache_point_still_merges_text():
    # Regression: without a cache point, adjacent same-role text blocks merge into one.
    messages = deltas_to_messages([Query(["a", "b"])])
    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["text"] == "ab"


def test_add_history_cache_point_tags_last_block():
    messages = deltas_to_messages([Query(["hello world"])])
    add_history_cache_point(messages, {"type": "ephemeral"})
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_add_history_cache_point_empty_is_noop():
    messages = []
    add_history_cache_point(messages, {"type": "ephemeral"})
    assert messages == []


@pytest.mark.asyncio
async def test_stream_applies_cache_history(mock_client):
    mock_client.messages.stream = Mock(return_value=return_val())
    llm = AnthropicLLM("a-model-id", mock_client)
    async for _ in llm.stream(
        model="a-model-id",
        history=[Query(["remember this"])],
        cache={"cache_history": True},
    ):
        pass
    messages = mock_client.messages.stream.call_args.kwargs["messages"]
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_stream_without_cache_history_has_no_breakpoint(mock_client):
    mock_client.messages.stream = Mock(return_value=return_val())
    llm = AnthropicLLM("a-model-id", mock_client)
    async for _ in llm.stream(
        model="a-model-id",
        history=[Query(["remember this"])],
    ):
        pass
    messages = mock_client.messages.stream.call_args.kwargs["messages"]
    assert all(
        "cache_control" not in block
        for message in messages
        for block in message["content"]
    )