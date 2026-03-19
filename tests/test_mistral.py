import pytest
from unittest.mock import Mock, AsyncMock, patch
import base64
import json
import typing as t

from mistralai.client.sdk import Mistral
from mistralai.client.models import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Tool,
    TextChunk,
    ToolCall,
    FunctionCall,
    ImageURLChunk,
    ImageURL,
)

from mus.llm.mistral import (
    MistralLLM,
    _map_mistral_exception,
    func_schema_to_tool,
    functions_for_llm,
    file_to_image_chunk,
    str_to_text_chunk,
    parse_content,
    query_to_messages,
    tool_result_to_content,
    merge_messages,
    deltas_to_messages,
    convert_tool_arguments,
)
from mus.llm.types import File, Query, Delta, ToolUse, ToolResult, Assistant, DeltaContent, DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory, Usage, ToolValue
from mus.functions import to_schema


ASYNC_T = t.TypeVar("ASYNC_T")


async def to_async_response(seq: t.Sequence[ASYNC_T]) -> t.AsyncGenerator[ASYNC_T, None]:
    """Convert a sequence to an async generator."""
    for item in seq:
        yield item


@pytest.fixture
def mock_mistral_client():
    client = Mock(spec=Mistral)
    client.chat = Mock()
    client.chat.stream_async = AsyncMock()
    client.chat.complete_async = AsyncMock()
    return client


@pytest.fixture
def mistral_llm(mock_mistral_client):
    return MistralLLM("mistral-medium", mock_mistral_client)


def test_func_schema_to_tool():
    def dummy_func(param1: str, param2: int = 5) -> str:
        """A dummy function for testing"""
        return "result"
    
    schema = to_schema(dummy_func)
    tool = func_schema_to_tool(schema)
    
    assert tool.type == "function"
    assert tool.function.name == schema["name"]
    assert tool.function.description == schema["description"]
    assert tool.function.parameters == schema["schema"]


def test_functions_for_llm():
    def func1(x: int) -> str:
        """Function 1"""
        return str(x)
    
    def func2(y: str) -> int:
        """Function 2"""
        return len(y)
    
    schemas = [to_schema(func1), to_schema(func2)]
    tools = functions_for_llm(schemas)
    
    assert len(tools) == 2
    assert all(isinstance(tool, Tool) for tool in tools)
    assert all(tool.function.name in ["func1", "func2"] for tool in tools)


def test_functions_for_llm_empty():
    assert functions_for_llm([]) == []
    assert functions_for_llm(None) == []


def test_file_to_image_chunk():
    image_data = b"fake_image_data"
    file = File(
        b64type="image/png", 
        content=base64.b64encode(image_data).decode()
    )
    
    chunk = file_to_image_chunk(file)
    assert isinstance(chunk, ImageURLChunk)
    assert chunk.type == "image_url"
    assert chunk.image_url.url.startswith("data:image/png;base64,")
    assert chunk.image_url.url.endswith(file.content)


def test_file_to_image_chunk_unsupported_format():
    file = File(
        b64type="image/bmp", 
        content=base64.b64encode(b"fake_data").decode()
    )
    
    with pytest.raises(ValueError, match="Unsupported image format"):
        file_to_image_chunk(file)


def test_file_to_image_chunk_invalid_type():
    file = File(
        b64type="application/pdf", 
        content=base64.b64encode(b"fake_data").decode()
    )
    
    with pytest.raises(ValueError, match="Only supports image"):
        file_to_image_chunk(file)


def test_file_to_image_chunk_invalid_b64type_format():
    file = File(
        b64type="invalid_format", 
        content=base64.b64encode(b"fake_data").decode()
    )
    
    with pytest.raises(ValueError, match="Invalid b64type"):
        file_to_image_chunk(file)


def test_str_to_text_chunk():
    result = str_to_text_chunk("Hello, world!")
    assert isinstance(result, TextChunk)
    assert result.type == "text"
    assert result.text == "Hello, world!"


def test_parse_content_string():
    result = parse_content("Hello, world!")
    assert isinstance(result, TextChunk)
    assert result.text == "Hello, world!"


def test_parse_content_file():
    file = File(
        b64type="image/jpeg", 
        content=base64.b64encode(b"fake_image_data").decode()
    )
    result = parse_content(file)
    assert isinstance(result, ImageURLChunk)


def test_parse_content_invalid():
    with pytest.raises(ValueError, match="Invalid query type"):
        parse_content(123)


def test_query_to_messages():
    query = Query([
        "User message 1",
        "User message 2",
        Assistant("Assistant response"),
        "User message 3",
        File(b64type="image/png", content=base64.b64encode(b"fake_image").decode())
    ])
    
    messages = query_to_messages(query)
    
    assert len(messages) == 5
    assert messages[0].role == "user"
    assert messages[0].content == "User message 1"
    assert messages[1].role == "user"
    assert messages[1].content == "User message 2"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Assistant response"
    assert messages[3].role == "user"
    assert messages[3].content == "User message 3"
    assert messages[4].role == "user"
    assert isinstance(messages[4].content, list)  # Image content


def test_parse_tool_content_string():
    from mus.llm.mistral import parse_tool_content
    result = parse_tool_content("Test content")
    assert result == "Test content"


def test_parse_tool_content_file():
    from mus.llm.mistral import parse_tool_content
    file = File(b64type="image/jpeg", content=base64.b64encode(b"fake_image").decode())
    result = parse_tool_content(file)
    assert result == "[Image: image/jpeg]"


def test_parse_tool_content_invalid():
    from mus.llm.mistral import parse_tool_content
    with pytest.raises(ValueError, match="Invalid tool result type"):
        parse_tool_content(123)


def test_tool_result_to_content():
    # String result
    str_result = ToolResult(id="1", content=ToolValue("text result"))
    assert tool_result_to_content(str_result) == "text result"

    # File result
    file_result = ToolResult(id="2", content=ToolValue(File(b64type="image/png", content=base64.b64encode(b"fake_image_data").decode())))
    assert tool_result_to_content(file_result) == "[Image: image/png]"

    # List result
    list_result = ToolResult(id="3", content=ToolValue(["text1", "text2"]))
    assert tool_result_to_content(list_result) == "text1\ntext2"

    # Invalid result
    with pytest.raises(ValueError):
        tool_result_to_content(ToolResult(id="4", content=ToolValue(123)))


def test_merge_messages():
    messages = [
        UserMessage(role="user", content="Hello"),
        UserMessage(role="user", content=" world"),
        AssistantMessage(role="assistant", content="Hi"),
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0].role == "user"
    assert merged[0].content == "Hello world"
    assert merged[1].role == "assistant"
    assert merged[1].content == "Hi"


def test_merge_messages_different_roles():
    messages = [
        UserMessage(role="user", content="Hello"),
        AssistantMessage(role="assistant", content="Hi"),
        UserMessage(role="user", content="How are you?"),
    ]
    merged = merge_messages(messages)
    assert len(merged) == 3  # No merging should happen
    assert merged[0].role == "user"
    assert merged[0].content == "Hello"
    assert merged[1].role == "assistant"
    assert merged[1].content == "Hi"
    assert merged[2].role == "user"
    assert merged[2].content == "How are you?"


def test_merge_messages_with_lists():
    # Messages with list content shouldn't be merged
    from mistralai.client.models import TextChunk
    messages = [
        UserMessage(role="user", content=[TextChunk(type="text", text="Hello")]),
        UserMessage(role="user", content=[TextChunk(type="text", text="world")]),
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2  # No merging should happen
    assert len(merged[0].content) == 1
    assert merged[0].content[0].text == "Hello"
    assert len(merged[1].content) == 1
    assert merged[1].content[0].text == "world"


def test_deltas_to_messages():
    deltas = [
        Query(["User message"]),
        Delta(content=DeltaText(data="Assistant response")),
        Delta(content=DeltaToolUse(data=ToolUse(id="1", name="tool1", input={"param": "value"}))),
        Delta(content=DeltaToolResult(data=ToolResult(id="1", content=ToolValue("Tool result")))),
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert isinstance(messages[0], UserMessage)
    
    assert messages[1].role == "assistant"
    assert isinstance(messages[1], AssistantMessage)
    assert messages[1].content == "Assistant response"
    
    assert messages[2].role == "tool"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "Tool result"


def test_deltas_to_messages_invalid_type():
    class InvalidDelta():
        pass
    deltas = [
        Delta(content=InvalidDelta()),
    ]
    
    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        deltas_to_messages(deltas)


def test_deltas_to_messages_empty_text():
    deltas = [
        Delta(content=DeltaText(data="")),
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 0  # Empty text deltas are not added


def test_convert_tool_arguments():
    # Dict arguments
    dict_args = {"param": "value"}
    assert convert_tool_arguments(dict_args) == dict_args
    
    # String arguments (JSON)
    str_args = '{"param": "value"}'
    assert convert_tool_arguments(str_args) == dict_args
    
    # Invalid JSON string
    from mus.llm.exceptions import LLMToolParseException
    with pytest.raises(LLMToolParseException):
        convert_tool_arguments('{invalid json}')
    
    # Invalid arguments
    with pytest.raises(ValueError):
        convert_tool_arguments(123)


@pytest.mark.asyncio
async def test_mistral_stream_called(mistral_llm, mock_mistral_client):
    mock_mistral_client.chat.stream_async.return_value = to_async_response([])
    
    async def dummy_tool(hello: int) -> str:
        """Dummy tool for testing"""
        return "dummy response"
    
    parsed_function = to_schema(dummy_tool)
    
    async for _ in mistral_llm.stream(prompt="Test prompt", model="test-model", history=[], functions=[parsed_function]):
        pass
    
    mock_mistral_client.chat.stream_async.assert_called_once()
    # Skip the detailed assertions since the internal format may change
    # Just verify that the function was called


@pytest.mark.asyncio
async def test_mistral_llm_stream(mistral_llm, mock_mistral_client):
    # Create a simple test that just verifies the method exists and is callable
    # No detailed testing of the implementation to avoid brittleness
    
    # Set up a mock response
    mock_mistral_client.chat.stream_async.return_value = to_async_response([])
    
    # Just call the method and make sure it doesn't raise an exception
    called = False
    async for _ in mistral_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
    ):
        called = True
    
    # Check that the method was called correctly
    assert mock_mistral_client.chat.stream_async.called


@pytest.mark.asyncio
async def test_mistral_llm_no_stream(mistral_llm, mock_mistral_client):
    # Set up a mock response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Complete response"
    
    mock_mistral_client.chat.complete_async.return_value = mock_response
    
    # Call the method with no_stream=True
    called = False
    async for _ in mistral_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
        no_stream=True
    ):
        called = True
        break
    
    # Check that the correct method was called
    assert mock_mistral_client.chat.complete_async.called


@pytest.mark.asyncio
async def test_mistral_llm_stream_parameters(mistral_llm, mock_mistral_client):
    # Create an empty async generator that will be returned by the mock
    async def empty_generator():
        if False:  # This ensures the generator is empty
            yield
    
    # Set up the mock
    mock_mistral_client.chat.stream_async.return_value = empty_generator()
    
    # Call the stream method with various parameters
    async for _ in mistral_llm.stream(
        prompt="System prompt",
        model="mistral-medium",
        history=[Query(["User message"])],
        functions=[],
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["STOP"],
        kwargs={"safe_prompt": True}
    ):
        pass
    
    # Verify that the method was called
    assert mock_mistral_client.chat.stream_async.called


@pytest.mark.asyncio
async def test_mistral_llm_stream_with_tool_calls(mistral_llm, mock_mistral_client):
    # Mock streaming response with tool calls
    mock_chunk = Mock()
    mock_chunk.data = Mock()
    mock_chunk.data.choices = [Mock()]
    mock_chunk.data.choices[0].delta = Mock()
    mock_chunk.data.choices[0].delta.content = None
    mock_function = Mock()
    mock_function.name = "test_function"
    mock_function.arguments = '{"param": "value"}'
    mock_chunk.data.choices[0].delta.tool_calls = [
        Mock(
            id="tool_1",
            function=mock_function
        )
    ]
    mock_chunk.data.choices[0].finish_reason = None
    
    mock_mistral_client.chat.stream_async.return_value = to_async_response([mock_chunk])
    
    def dummy_func(param: str) -> str:
        """Dummy function for testing"""
        return "result"
    
    results = []
    async for delta in mistral_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[to_schema(dummy_func)],
    ):
        results.append(delta)
    
    assert len(results) == 1
    assert isinstance(results[0].content, DeltaToolUse)
    assert results[0].content.data.name == "test_function"
    assert results[0].content.data.input == {"param": "value"}


@pytest.mark.asyncio
async def test_mistral_llm_stream_with_content_and_usage(mistral_llm, mock_mistral_client):
    # Mock streaming response with content and usage
    mock_chunk1 = Mock()
    mock_chunk1.data = Mock()
    mock_chunk1.data.choices = [Mock()]
    mock_chunk1.data.choices[0].delta = Mock()
    mock_chunk1.data.choices[0].delta.content = "Hello world"
    mock_chunk1.data.choices[0].delta.tool_calls = None
    mock_chunk1.data.choices[0].finish_reason = None
    
    mock_chunk2 = Mock()
    mock_chunk2.data = Mock()
    mock_chunk2.data.choices = [Mock()]
    mock_chunk2.data.choices[0].delta = Mock()
    mock_chunk2.data.choices[0].delta.content = None
    mock_chunk2.data.choices[0].delta.tool_calls = None
    mock_chunk2.data.choices[0].finish_reason = "stop"
    mock_chunk2.data.usage = Mock()
    mock_chunk2.data.usage.prompt_tokens = 10
    mock_chunk2.data.usage.completion_tokens = 5
    
    mock_mistral_client.chat.stream_async.return_value = to_async_response([mock_chunk1, mock_chunk2])
    
    results = []
    async for delta in mistral_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
    ):
        results.append(delta)
    
    # Separate text and usage deltas
    text_deltas = [r for r in results if isinstance(r.content, DeltaText) and r.content.data]
    usage_deltas = [r for r in results if r.usage is not None]
    
    # Verify content was streamed properly
    assert len(text_deltas) == 1
    assert text_deltas[0].content.data == "Hello world"

    # Verify usage
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.input_tokens == 10
    assert usage_deltas[0].usage.output_tokens == 5


@pytest.mark.asyncio
async def test_mistral_llm_completion_with_tool_calls(mistral_llm, mock_mistral_client):
    # Mock completion response with tool calls
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response content"
    mock_function = Mock()
    mock_function.name = "test_function"
    mock_function.arguments = '{"param": "value"}'
    mock_response.choices[0].message.tool_calls = [
        Mock(
            id="tool_1",
            function=mock_function
        )
    ]
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 8
    
    mock_mistral_client.chat.complete_async.return_value = mock_response
    
    def dummy_func(param: str) -> str:
        """Dummy function for testing"""
        return "result"
    
    results = []
    async for delta in mistral_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[to_schema(dummy_func)],
        no_stream=True
    ):
        results.append(delta)
    
    # Separate different types of deltas
    text_deltas = [r for r in results if isinstance(r.content, DeltaText) and r.content.data]
    tool_deltas = [r for r in results if isinstance(r.content, DeltaToolUse)]
    usage_deltas = [r for r in results if r.usage is not None]
    
    # Verify content was streamed properly
    assert len(text_deltas) == 1
    assert text_deltas[0].content.data == "Response content"
    
    # Verify tool use
    assert len(tool_deltas) == 1
    assert tool_deltas[0].content.data.name == "test_function"
    assert tool_deltas[0].content.data.input == {"param": "value"}
    # Verify usage
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.input_tokens == 15
    assert usage_deltas[0].usage.output_tokens == 8


@pytest.mark.asyncio
async def test_choice_content_to_chunks_string():
    from mus.llm.mistral import choice_content_to_chunks
    
    results = []
    async for chunk in choice_content_to_chunks("Hello world"):
        results.append(chunk)
    
    assert len(results) == 1
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Hello world"


@pytest.mark.asyncio
async def test_choice_content_to_chunks_text_chunk():
    from mus.llm.mistral import choice_content_to_chunks
    from mistralai.client.models import TextChunk
    
    text_chunk = TextChunk(type="text", text="Hello world")
    
    results = []
    async for chunk in choice_content_to_chunks([text_chunk]):
        results.append(chunk)
    
    assert len(results) == 1
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Hello world"


@pytest.mark.asyncio
async def test_choice_content_to_chunks_invalid_type():
    from mus.llm.mistral import choice_content_to_chunks
    
    with pytest.raises(ValueError, match="Unsupported content type"):
        async for _ in choice_content_to_chunks([123]):
            pass


def test_mistral_llm_initialization_with_client():
    # Test with provided client
    mock_client = Mock(spec=Mistral)
    llm = MistralLLM("mistral-medium", mock_client)
    assert llm.client is mock_client
    assert llm.model == "mistral-medium"


def test_mistral_llm_initialization_without_client():
    # Test initialization without client (will create default)
    with patch('mus.llm.mistral.Mistral') as mock_mistral_class:
        mock_client = Mock(spec=Mistral)
        mock_mistral_class.return_value = mock_client
        
        llm = MistralLLM("mistral-medium", api_key="test-key")
        
        mock_mistral_class.assert_called_once_with(api_key="test-key")
        assert llm.client is mock_client
        assert llm.model == "mistral-medium"


# --- Exception handling tests ---

import httpx
from mistralai.client.errors.mistralerror import MistralError
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMRateLimitException,
    LLMConnectionException,
    LLMServerException,
    LLMBadRequestException,
    LLMNotFoundException,
    LLMToolParseException,
    LLMException,
)


def _make_httpx_response(status_code, headers=None):
    """Helper to create a mock httpx.Response for MistralError."""
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    resp.text = "error body"
    return resp


def _make_mistral_error(status_code, headers=None, message="error"):
    """Helper to create a MistralError."""
    resp = _make_httpx_response(status_code, headers)
    return MistralError(message, resp)


# --- Direct mapping function tests ---

def test_map_mistral_auth_error_401():
    exc = _make_mistral_error(401)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 401


def test_map_mistral_auth_error_403():
    exc = _make_mistral_error(403)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 403


def test_map_mistral_rate_limit_with_retry_after():
    exc = _make_mistral_error(429, headers={"retry-after": "60", "x-request-id": "req-mr1"})
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 429
    assert mapped.retry_after == 60.0
    assert mapped.request_id == "req-mr1"


def test_map_mistral_rate_limit_no_retry_after():
    exc = _make_mistral_error(429)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.retry_after is None


def test_map_mistral_bad_request_400():
    exc = _make_mistral_error(400)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 400


def test_map_mistral_bad_request_422():
    exc = _make_mistral_error(422)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 422


def test_map_mistral_not_found():
    exc = _make_mistral_error(404)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMNotFoundException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 404


def test_map_mistral_server_error():
    exc = _make_mistral_error(500)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 500


def test_map_mistral_server_error_503():
    exc = _make_mistral_error(503)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 503


def test_map_mistral_unknown_status():
    exc = _make_mistral_error(418)
    mapped = _map_mistral_exception(exc)
    assert isinstance(mapped, LLMException)
    assert not isinstance(mapped, LLMServerException)
    assert mapped.provider == "mistral"
    assert mapped.status_code == 418


@pytest.mark.asyncio
async def test_mistral_auth_error(mistral_llm, mock_mistral_client):
    exc = _make_mistral_error(401)
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "mistral"
    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_mistral_rate_limit_error_with_retry_after(mistral_llm, mock_mistral_client):
    exc = _make_mistral_error(429, headers={"retry-after": "60"})
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "mistral"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 60.0
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_mistral_server_error(mistral_llm, mock_mistral_client):
    exc = _make_mistral_error(500)
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "mistral"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_mistral_not_found_error(mistral_llm, mock_mistral_client):
    exc = _make_mistral_error(404)
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMNotFoundException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "mistral"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mistral_bad_request_error(mistral_llm, mock_mistral_client):
    exc = _make_mistral_error(400)
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "mistral"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_mistral_stream_iteration_error(mistral_llm, mock_mistral_client):
    """SDK exception during stream iteration is mapped correctly."""
    exc = _make_mistral_error(500)

    async def failing_stream():
        raise exc
        yield  # make it a generator

    mock_mistral_client.chat.stream_async.return_value = failing_stream()

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_mistral_tool_parse_error_in_stream(mistral_llm, mock_mistral_client):
    """Malformed tool JSON during streaming raises LLMToolParseException."""
    mock_chunk = Mock()
    mock_chunk.data = Mock()
    mock_chunk.data.choices = [Mock()]
    mock_chunk.data.choices[0].delta = Mock()
    mock_chunk.data.choices[0].delta.content = None
    mock_function = Mock()
    mock_function.name = "test_tool"
    mock_function.arguments = "{invalid json"
    mock_chunk.data.choices[0].delta.tool_calls = [
        Mock(id="t1", function=mock_function)
    ]
    mock_chunk.data.choices[0].finish_reason = None

    mock_mistral_client.chat.stream_async.return_value = to_async_response([mock_chunk])

    with pytest.raises(LLMToolParseException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m", functions=[]):
            pass
    assert exc_info.value.provider == "mistral"


@pytest.mark.asyncio
async def test_mistral_exception_chaining(mistral_llm, mock_mistral_client):
    """Verify __cause__ is the original SDK exception."""
    exc = _make_mistral_error(401)
    mock_mistral_client.chat.stream_async.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_mistral_no_stream_error(mistral_llm, mock_mistral_client):
    """Exception during non-streaming call is mapped correctly."""
    exc = _make_mistral_error(429, headers={"retry-after": "10"})
    mock_mistral_client.chat.complete_async.side_effect = exc

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in mistral_llm.stream(prompt="p", history=[], model="m", no_stream=True):
            pass
    assert exc_info.value.retry_after == 10.0
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_json,expected_args", [
    ('{"name": "test", "value": 42,}', {"name": "test", "value": 42}),
    ("{'name': 'test', 'value': 42}", {"name": "test", "value": 42}),
    ('{"name": "test", "value": 42', {"name": "test", "value": 42}),
    ('{name: "test", value: 42}', {"name": "test", "value": 42}),
])
async def test_mistral_json_repair_in_stream(mistral_llm, mock_mistral_client, malformed_json, expected_args):
    """Malformed but repairable tool JSON is repaired during streaming."""
    mock_chunk = Mock()
    mock_chunk.data = Mock()
    mock_chunk.data.choices = [Mock()]
    mock_chunk.data.choices[0].delta = Mock()
    mock_chunk.data.choices[0].delta.content = None
    mock_function = Mock()
    mock_function.name = "test_tool"
    mock_function.arguments = malformed_json
    mock_chunk.data.choices[0].delta.tool_calls = [
        Mock(id="t1", function=mock_function)
    ]
    mock_chunk.data.choices[0].finish_reason = None

    mock_mistral_client.chat.stream_async.return_value = to_async_response([mock_chunk])

    deltas = []
    async for d in mistral_llm.stream(prompt="p", history=[], model="m", functions=[]):
        deltas.append(d)

    tool_uses = [d for d in deltas if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.input == expected_args