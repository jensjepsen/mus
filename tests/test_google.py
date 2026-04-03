import pytest
from unittest.mock import Mock, AsyncMock
from unittest.mock import patch
import base64
import typing as t

from google import genai
from google.genai import types as genai_types

from mus.llm.google import (
    GoogleGenAILLM,
    _map_google_exception,
    func_schema_to_tool,
    functions_for_llm,
    file_to_part,
    parse_content,
    query_to_contents,
    tool_result_to_parts,
    deltas_to_contents,
)
from mus.llm.types import File, Query, Delta, ToolUse, ToolResult, Assistant, DeltaContent, DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory, DeltaToolInputUpdate, ToolValue
from mus.functions import to_schema


ASYNC_T = t.TypeVar("ASYNC_T")

async def to_async_response(seq: t.Sequence[ASYNC_T]) -> t.AsyncGenerator[ASYNC_T, None]:
    """Convert a sequence to an async generator."""
    for item in seq:
        yield item

@pytest.fixture
def mock_genai_client():
    client = Mock(spec=genai.Client)
    client.aio = Mock()
    client.aio.models = Mock()
    return client


@pytest.fixture
def google_genai_llm(mock_genai_client):
    return GoogleGenAILLM("gemini-1.5-pro", mock_genai_client)


def test_func_schema_to_tool():
    def dummy_func(param1: str, param2: int = 5) -> str:
        """A dummy function for testing"""
        return "result"
    
    schema = to_schema(dummy_func)
    tool = func_schema_to_tool(schema)
    
    assert isinstance(tool, genai_types.FunctionDeclaration)
    assert tool.name == schema["name"]
    assert tool.description == schema["description"]
    assert tool.parameters is not None
    assert tool.parameters.type == genai_types.Type.OBJECT
    assert tool.parameters.properties is not None
    assert "param1" in tool.parameters.properties
    assert "param2" in tool.parameters.properties


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
    assert all(isinstance(tool, genai_types.Tool) for tool in tools)
    assert all(len(tool.function_declarations or []) == 1 for tool in tools)


def test_functions_for_llm_empty():
    assert functions_for_llm([]) == []
    assert functions_for_llm(None) == [] # type: ignore # intentionally passing None


def test_file_to_part_image():
    image_data = b"fake_image_data"
    file = File(
        b64type="image/png", 
        content=base64.b64encode(image_data).decode()
    )
    
    part = file_to_part(file)
    assert isinstance(part, genai_types.Part)


def test_file_to_part_unsupported_image():
    file = File(
        b64type="image/bmp", 
        content=base64.b64encode(b"fake_data").decode()
    )
    
    with pytest.raises(ValueError, match="Unsupported image type"):
        file_to_part(file)


def test_file_to_part_invalid_type():
    file = File(
        b64type="invalid_type", 
        content=base64.b64encode(b"fake_data").decode()
    )
    
    with pytest.raises(ValueError, match="Invalid b64type"):
        file_to_part(file)


def test_file_to_part_other_types():
    file = File(
        b64type="application/pdf", 
        content=base64.b64encode(b"fake_pdf_data").decode()
    )
    
    part = file_to_part(file)
    assert isinstance(part, genai_types.Part)


def test_parse_content_string():
    result = parse_content("Hello, world!")
    assert isinstance(result, genai_types.Part)


def test_parse_content_file():
    file = File(
        b64type="image/jpeg", 
        content=base64.b64encode(b"fake_image_data").decode()
    )
    result = parse_content(file)
    assert isinstance(result, genai_types.Part)


def test_parse_content_invalid():
    with pytest.raises(ValueError, match="Invalid query type"):
        parse_content(123) # type: ignore # intentionally wrong type


def test_query_to_contents():
    query = Query([
        "User message 1",
        "User message 2",
        Assistant("Assistant response"),
        "User message 3",
        File(b64type="image/png", content=base64.b64encode(b"fake_image").decode())
    ])
    
    contents = query_to_contents(query)
    
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2  # Two user messages
    assert contents[1].role == "model"
    assert len(contents[1].parts) == 1  # Assistant response
    assert contents[2].role == "user"
    assert len(contents[2].parts) == 2  # User message + image


def test_query_to_contents_empty():
    query = Query([])
    contents = query_to_contents(query)
    assert contents == []


def test_tool_result_to_parts_string():
    tool_result = ToolResult(id="1", content=ToolValue("Simple string result"))
    parts = tool_result_to_parts(tool_result)
    
    assert len(parts) == 1
    assert isinstance(parts[0], genai_types.Part)
    assert parts[0].text == "Simple string result"


def test_tool_result_to_parts_invalid():
    tool_result = ToolResult(id="invalid", content=ToolValue(123))  # type: ignore # intentionally wrong type
    
    with pytest.raises(ValueError, match="Invalid tool result type"):
        tool_result_to_parts(tool_result)
    
    tool_result = ToolResult(id="invalid", content=ToolValue([123, "hello"]))  # type: ignore # intentionally wrong type
    with pytest.raises(ValueError, match="Invalid tool result type"):
        tool_result_to_parts(tool_result)

def test_tool_result_to_parts_file():
    file = File(
        b64type="image/png", 
        content=base64.b64encode(b"fake_image").decode()
    )
    tool_result = ToolResult(id="2", content=ToolValue(file))
    parts = tool_result_to_parts(tool_result)
    
    assert len(parts) == 1
    assert isinstance(parts[0], genai_types.Part)
    assert parts[0].as_image() is not None


def test_tool_result_to_parts_list():
    tool_result = ToolResult(id="3", content=ToolValue(["text1", "text2"]))
    parts = tool_result_to_parts(tool_result)
    
    assert len(parts) == 2
    assert all(isinstance(part, genai_types.Part) for part in parts)
    assert parts[0].text == "text1"
    assert parts[1].text == "text2"


def test_tool_result_to_parts_mixed_list():
    file = File(
        b64type="image/jpeg", 
        content=base64.b64encode(b"fake_image").decode()
    )
    tool_result = ToolResult(id="4", content=ToolValue(["text", file, "123"]))
    parts = tool_result_to_parts(tool_result)
    
    assert len(parts) == 3
    assert all(isinstance(part, genai_types.Part) for part in parts)
    assert parts[0].text == "text"
    assert parts[1].as_image() is not None
    assert parts[2].text == "123"

def test_tool_result_to_parts_with_metadata():
    """Test that metadata in ToolValue is not serialized as text in parts."""
    tool_result = ToolResult(
        id="5", 
        content=ToolValue("Result with metadata", metadata={"key": "value"}), 
    )
    parts = tool_result_to_parts(tool_result)
    
    assert len(parts) == 1
    assert isinstance(parts[0], genai_types.Part)
    assert parts[0].text == "Result with metadata"
    

def test_deltas_to_contents():
    deltas = [
        Query(["User question"]),
        Delta(content=DeltaText(data="Assistant response")),
        Delta(content=DeltaToolUse(data=ToolUse(name="search", input={"query": "test"}, id="tool1"))),
        Delta(content=DeltaToolResult(data=ToolResult(id="tool1", content=ToolValue("Searchresults")))),
    ]
    
    contents = deltas_to_contents(deltas)
    
    assert len(contents) == 4
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "model"  # tool_use
    assert contents[3].role == "tool"   # tool_result


@pytest.mark.asyncio
async def test_google_genai_stream_basic(google_genai_llm, mock_genai_client):
    # Mock streaming response
    async def resp():
        # First chunk with text
        mock_part1 = Mock()
        mock_part1.text = "Hello"
        mock_part1.function_call = None
        mock_part1.thought_signature = None

        mock_candidate1 = Mock()
        mock_candidate1.content = Mock()
        mock_candidate1.content.parts = [mock_part1]

        yield Mock(
            candidates=[mock_candidate1],
            usage_metadata=None
        )

        # Second chunk with text and usage
        mock_part2 = Mock()
        mock_part2.text = " world!"
        mock_part2.function_call = None
        mock_part2.thought_signature = None

        mock_candidate2 = Mock()
        mock_candidate2.content = Mock()
        mock_candidate2.content.parts = [mock_part2]

        yield Mock(
            candidates=[mock_candidate2],
            usage_metadata=Mock(
                prompt_token_count=10,
                candidates_token_count=5
            )
        )

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=resp())

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test prompt",
        model="gemini-1.5-pro",
        history=[],
        functions=[]
    ):
        results.append(delta)

    assert len(results) == 3  # 2 text deltas + 1 usage delta
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Hello"
    assert isinstance(results[1].content, DeltaText)
    assert results[1].content.data == " world!"
    assert results[2].usage.input_tokens == 10
    assert results[2].usage.output_tokens == 5


@pytest.mark.asyncio
async def test_google_genai_stream_with_tools(google_genai_llm, mock_genai_client):
    def search_tool(query: str) -> str:
        """Search for information"""
        return "search results"

    mock_function_call = Mock()
    mock_function_call.id = "call_123"
    mock_function_call.name = "search_tool"
    mock_function_call.args = {"query": "test"}

    mock_part = Mock()
    mock_part.text = None
    mock_part.function_call = mock_function_call
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(
            candidates=[mock_candidate],
            usage_metadata=None
        )
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test prompt",
        model="gemini-1.5-pro",
        history=[],
        functions=[to_schema(search_tool)]
    ):
        results.append(delta)

    assert len(results) == 2
    assert isinstance(results[0].content, DeltaToolInputUpdate)
    assert results[0].content.name == "search_tool"
    assert results[0].content.id == "call_123"
    assert results[0].content.data == '{"query": "test"}'

    assert isinstance(results[1].content, DeltaToolUse)
    tool_use = results[1].content.data
    assert tool_use.id == "call_123"
    assert tool_use.name == "search_tool"
    assert tool_use.input == {"query": "test"}


@pytest.mark.asyncio
async def test_google_genai_no_stream(google_genai_llm, mock_genai_client):
    mock_part = Mock()
    mock_part.text = "Complete response"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = Mock(
        candidates=[mock_candidate],
        usage_metadata=Mock(
            prompt_token_count=15,
            candidates_token_count=8
        )
    )

    mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test prompt",
        model="gemini-1.5-pro",
        history=[],
        functions=[],
        no_stream=True
    ):
        results.append(delta)

    assert len(results) == 2  # 1 text delta + 1 usage delta
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Complete response"
    assert results[1].usage.input_tokens == 15
    assert results[1].usage.output_tokens == 8


@pytest.mark.asyncio
async def test_google_genai_stream_parameters(google_genai_llm, mock_genai_client):
    mock_part = Mock()
    mock_part.text = "Response"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(candidates=[mock_candidate], usage_metadata=None)
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    async for _ in google_genai_llm.stream(
        prompt="System prompt",
        model="gemini-1.5-pro",
        history=[Query(["User message"])],
        functions=[],
        max_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["STOP"]
    ):
        pass

    # Verify the call was made with correct parameters
    call_args = mock_genai_client.aio.models.generate_content_stream.call_args
    assert call_args[1]["model"] == "gemini-1.5-pro"

    config = call_args[1]["config"]
    assert config.system_instruction == "System prompt"
    assert config.max_output_tokens == 1000
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.stop_sequences == ["STOP"]


@pytest.mark.asyncio
async def test_google_genai_stream_with_history(google_genai_llm, mock_genai_client):
    mock_part = Mock()
    mock_part.text = "Response"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(candidates=[mock_candidate], usage_metadata=None)
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    history = [
        Query(["First user message"]),
        Delta(content=DeltaText(data="First assistant response")),
        Query(["Second user message"])
    ]

    async for _ in google_genai_llm.stream(
        model="gemini-1.5-pro",
        history=history,
        functions=[]
    ):
        pass

    call_args = mock_genai_client.aio.models.generate_content_stream.call_args
    contents = call_args[1]["contents"]

    # Should have processed the history into proper contents
    assert len(contents) >= 1
    assert all(isinstance(content, genai_types.Content) for content in contents)


def test_google_genai_llm_initialization():
    # Test with provided client
    mock_client = Mock(spec=genai.Client)
    llm = GoogleGenAILLM("gemini-1.5-pro", mock_client)
    assert llm.client is mock_client
    assert llm.model == "gemini-1.5-pro"


def test_google_genai_llm_initialization_no_client():
    # Test without provided client (should create default)
    with patch('google.genai.Client') as mock_client_class:
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        llm = GoogleGenAILLM("gemini-1.5-pro")
        assert llm.client is mock_client_instance
        mock_client_class.assert_called_once()


@pytest.mark.asyncio
async def test_google_genai_empty_response(google_genai_llm, mock_genai_client):
    # Test handling of empty/None responses
    mock_response = to_async_response([
        Mock(candidates=[], usage_metadata=None)
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        model="gemini-1.5-pro",
        history=[],
        functions=[]
    ):
        results.append(delta)

    # Should handle empty responses gracefully
    assert len(results) == 0


@pytest.mark.asyncio
async def test_google_genai_function_call_without_id(google_genai_llm, mock_genai_client):
    # Test function call that doesn't have an ID (uses name as fallback)
    mock_function_call = Mock()
    mock_function_call.id = None  # Explicitly set to None to test fallback
    mock_function_call.name = "test_function"
    mock_function_call.args = {"param": "value"}

    mock_part = Mock()
    mock_part.text = None
    mock_part.function_call = mock_function_call
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(
            candidates=[mock_candidate],
            usage_metadata=None
        )
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        model="gemini-1.5-pro",
        history=[],
        functions=[]
    ):
        results.append(delta)

    assert len(results) == 2
    assert isinstance(results[0].content, DeltaToolInputUpdate)
    assert results[0].content.name == "test_function"
    assert results[0].content.id == "test_function"
    tool_use = results[1].content.data
    assert tool_use.id == "test_function"  # Should use name as fallback
    assert tool_use.name == "test_function"


@pytest.mark.asyncio
async def test_google_genai_thought_signatures(google_genai_llm, mock_genai_client):
    # Test that thought signatures are preserved in metadata
    mock_part1 = Mock()
    mock_part1.text = "Thinking about the problem..."
    mock_part1.function_call = None
    mock_part1.thought_signature = "thinking"

    mock_candidate1 = Mock()
    mock_candidate1.content = Mock()
    mock_candidate1.content.parts = [mock_part1]

    mock_part2 = Mock()
    mock_part2.text = "Here's my response"
    mock_part2.function_call = None
    mock_part2.thought_signature = None

    mock_candidate2 = Mock()
    mock_candidate2.content = Mock()
    mock_candidate2.content.parts = [mock_part2]

    mock_response = to_async_response([
        Mock(candidates=[mock_candidate1], usage_metadata=None),
        Mock(candidates=[mock_candidate2], usage_metadata=None)
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        model="gemini-1.5-pro",
        history=[],
        functions=[]
    ):
        results.append(delta)

    assert len(results) == 2

    # First delta should have thought signature in metadata
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Thinking about the problem..."
    assert results[0].metadata.get("thought_signature") == "thinking"

    # Second delta should not have thought signature
    assert isinstance(results[1].content, DeltaText)
    assert results[1].content.data == "Here's my response"
    assert "thought_signature" not in results[1].metadata or results[1].metadata.get("thought_signature") is None


@pytest.mark.asyncio
async def test_google_genai_thought_signatures_with_tools(google_genai_llm, mock_genai_client):
    # Test that thought signatures are preserved with tool calls
    mock_function_call = Mock()
    mock_function_call.id = "call_456"
    mock_function_call.name = "analyze_data"
    mock_function_call.args = {"data": "test"}

    mock_part = Mock()
    mock_part.text = None
    mock_part.function_call = mock_function_call
    mock_part.thought_signature = "tool_use"

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(candidates=[mock_candidate], usage_metadata=None)
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        model="gemini-1.5-pro",
        history=[],
        functions=[]
    ):
        results.append(delta)

    assert len(results) == 2

    # Both tool-related deltas should have thought signature
    assert isinstance(results[0].content, DeltaToolInputUpdate)
    assert results[0].metadata.get("thought_signature") == "tool_use"

    assert isinstance(results[1].content, DeltaToolUse)
    assert results[1].metadata.get("thought_signature") == "tool_use"


# --- Exception handling tests ---

from google.genai import errors as genai_errors
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMRateLimitException,
    LLMServerException,
    LLMBadRequestException,
    LLMNotFoundException,
    LLMModelException,
    LLMException,
)


def _make_google_api_error(cls, code, status=None):
    """Helper to create a Google genai APIError (or subclass)."""
    response_json = {"error": {"message": "test error", "status": status or "UNKNOWN"}}
    err = cls(code, response_json, response=None)
    return err


# --- Direct mapping function tests ---

def test_map_google_auth_error_401():
    exc = _make_google_api_error(genai_errors.ClientError, 401)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "google"
    assert mapped.status_code == 401


def test_map_google_auth_error_403():
    exc = _make_google_api_error(genai_errors.ClientError, 403)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "google"
    assert mapped.status_code == 403


def test_map_google_rate_limit():
    exc = _make_google_api_error(genai_errors.ClientError, 429)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "google"
    assert mapped.status_code == 429


def test_map_google_bad_request():
    exc = _make_google_api_error(genai_errors.ClientError, 400)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "google"
    assert mapped.status_code == 400


def test_map_google_not_found():
    exc = _make_google_api_error(genai_errors.ClientError, 404)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMNotFoundException)
    assert mapped.provider == "google"
    assert mapped.status_code == 404


def test_map_google_model_unavailable():
    exc = _make_google_api_error(genai_errors.ServerError, 503, status="UNAVAILABLE")
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMModelException)
    assert mapped.provider == "google"
    assert mapped.status_code == 503


def test_map_google_server_error():
    exc = _make_google_api_error(genai_errors.ServerError, 500)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "google"
    assert mapped.status_code == 500


def test_map_google_client_error_fallback():
    """ClientError with unrecognized status code falls through to LLMBadRequestException."""
    exc = _make_google_api_error(genai_errors.ClientError, 418)
    mapped = _map_google_exception(exc)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "google"
    assert mapped.status_code == 418


@pytest.mark.asyncio
async def test_google_auth_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ClientError, 401)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_google_rate_limit_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ClientError, 429)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_google_server_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ServerError, 500)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_google_not_found_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ClientError, 404)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMNotFoundException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_google_bad_request_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ClientError, 400)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_google_model_unavailable_error(google_genai_llm, mock_genai_client):
    exc = _make_google_api_error(genai_errors.ServerError, 503, status="UNAVAILABLE")
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMModelException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_google_stream_iteration_error(google_genai_llm, mock_genai_client):
    """SDK exception during stream iteration is mapped correctly."""
    exc = _make_google_api_error(genai_errors.ServerError, 500)

    async def failing_stream():
        yield Mock(candidates=[], usage_metadata=None)
        raise exc

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=failing_stream())

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_google_exception_chaining(google_genai_llm, mock_genai_client):
    """Verify __cause__ is the original SDK exception."""
    exc = _make_google_api_error(genai_errors.ClientError, 401)
    mock_genai_client.aio.models.generate_content_stream = AsyncMock(side_effect=exc)

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_google_no_stream_error(google_genai_llm, mock_genai_client):
    """Exception during non-streaming call is mapped correctly."""
    exc = _make_google_api_error(genai_errors.ClientError, 429)
    mock_genai_client.aio.models.generate_content = AsyncMock(side_effect=exc)

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in google_genai_llm.stream(prompt="p", history=[], model="m", no_stream=True):
            pass
    assert exc_info.value.provider == "google"
    assert exc_info.value.__cause__ is exc


# --- Cached token tests ---

@pytest.mark.asyncio
async def test_google_genai_stream_cached_tokens(google_genai_llm, mock_genai_client):
    """Streaming: cached_content_token_count is reported as cache_read_input_tokens."""
    mock_part = Mock()
    mock_part.text = "Hello"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(
            candidates=[mock_candidate],
            usage_metadata=Mock(
                prompt_token_count=100,
                candidates_token_count=20,
                cached_content_token_count=75,
            ),
        )
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test", model="gemini-1.5-pro", history=[], functions=[]
    ):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.input_tokens == 100
    assert usage_deltas[0].usage.output_tokens == 20
    assert usage_deltas[0].usage.cache_read_input_tokens == 75


@pytest.mark.asyncio
async def test_google_genai_no_stream_cached_tokens(google_genai_llm, mock_genai_client):
    """Non-streaming: cached_content_token_count is reported as cache_read_input_tokens."""
    mock_part = Mock()
    mock_part.text = "Hello"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = Mock(
        candidates=[mock_candidate],
        usage_metadata=Mock(
            prompt_token_count=100,
            candidates_token_count=20,
            cached_content_token_count=50,
        ),
    )

    mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test", model="gemini-1.5-pro", history=[], functions=[], no_stream=True
    ):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.cache_read_input_tokens == 50


@pytest.mark.asyncio
async def test_google_genai_stream_no_cached_tokens(google_genai_llm, mock_genai_client):
    """Streaming: missing cached_content_token_count defaults to 0."""
    mock_part = Mock()
    mock_part.text = "Hello"
    mock_part.function_call = None
    mock_part.thought_signature = None

    mock_candidate = Mock()
    mock_candidate.content = Mock()
    mock_candidate.content.parts = [mock_part]

    mock_response = to_async_response([
        Mock(
            candidates=[mock_candidate],
            usage_metadata=Mock(
                prompt_token_count=50,
                candidates_token_count=10,
                cached_content_token_count=None,
            ),
        )
    ])

    mock_genai_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_response)

    results = []
    async for delta in google_genai_llm.stream(
        prompt="Test", model="gemini-1.5-pro", history=[], functions=[]
    ):
        results.append(delta)

    usage_deltas = [r for r in results if r.usage is not None]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage.cache_read_input_tokens == 0