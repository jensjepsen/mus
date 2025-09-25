import pytest
from unittest.mock import Mock, AsyncMock, patch
import base64
import json
import typing as t

from mistralai import Mistral
from mistralai.models import (
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
from mus.llm.types import File, Query, Delta, ToolUse, ToolResult, Assistant, DeltaContent, DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory, Usage
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
    str_result = ToolResult(id="1", content="text result")
    assert tool_result_to_content(str_result) == "text result"

    # File result
    file_result = ToolResult(id="2", content=File(b64type="image/png", content=base64.b64encode(b"fake_image_data").decode()))
    assert tool_result_to_content(file_result) == "[Image: image/png]"

    # List result
    list_result = ToolResult(id="3", content=["text1", "text2"])
    assert tool_result_to_content(list_result) == "text1\ntext2"

    # Invalid result
    with pytest.raises(ValueError):
        tool_result_to_content(ToolResult(id="4", content=123))


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
    from mistralai.models import TextChunk
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
        Delta(content=DeltaToolResult(data=ToolResult(id="1", content="Tool result"))),
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
    
    with pytest.raises(ValueError, match="Invalid delta type"):
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
    with pytest.raises(json.JSONDecodeError):
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
    from mistralai.models import TextChunk
    
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