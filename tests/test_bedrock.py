import pytest
from unittest.mock import Mock, AsyncMock
from mus.llm.bedrock import (
    BedrockLLM,
    _map_bedrock_exception,
    functions_for_llm,
    file_to_image,
    str_to_text_block,
    parse_content,
    query_to_messages,
    tool_result_to_content,
    join_content,
    merge_messages,
    deltas_to_messages,
    add_history_cache_point,
)
from mus.llm.types import DeltaToolInputUpdate, File, Query, Delta, ToolUse, ToolResult, Assistant, DeltaContent, DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory, Usage, ToolValue, CachePoint
from mus.functions import to_schema
from dataclasses import dataclass
import base64
import json
import logging

@pytest.fixture
def mock_bedrock_client():
    mock = AsyncMock()
    return mock

@pytest.fixture
def bedrock_llm(mock_bedrock_client):
    return BedrockLLM("a-model-id", mock_bedrock_client)

@pytest.mark.asyncio
async def test_bedrock_stream_called(bedrock_llm, mock_bedrock_client):
    async def async_iter():
        return
        yield  # Make this a generator

    mock_bedrock_client.converse_stream.return_value = {
        "stream": async_iter()
    }
    async def dummy_tool(hello: int) -> str:
        """Dummy tool for testing"""
        return "dummy response"
    parsed_function = to_schema(dummy_tool)
    async for _ in bedrock_llm.stream(prompt="Test prompt", model="test-model", history=[], functions=[parsed_function]):
        pass
    mock_bedrock_client.converse_stream.assert_called_once_with(
        modelId="a-model-id",
        messages=[],
        system=[
            {
                "text": "Test prompt",
            }
        ],
        inferenceConfig={
            "maxTokens": 4096,
        },
        toolConfig={
            "tools": [{
                "toolSpec":{
                    "name": parsed_function['name'],
                    "description": parsed_function['description'],
                    "inputSchema": {
                        "json": parsed_function['schema']
                    }
                }
            }]
        }
    )


def test_functions_for_llm():
    def func1():
        """A docstring"""
        pass
    def func2():
        """A docstring"""
        pass
    @dataclass
    class DataClass1: pass

    result = functions_for_llm([to_schema(func1), to_schema(func2), to_schema(DataClass1)])
    assert len(result) == 3
    assert all('toolSpec' in item for item in result)

def test_file_to_image():
    file = File(b64type="image/png", content=base64.b64encode(b"fake_image_data").decode())
    result = file_to_image(file)
    assert result['image']['format'] == 'png'
    assert 'bytes' in result['image']['source']

def test_str_to_text_block():
    result = str_to_text_block("Hello, world!")
    assert result == {'text': "Hello, world!"}

def test_parse_content():
    assert parse_content("text") == {'text': "text"}
    file = File(b64type="image/jpeg", content=base64.b64encode(b"fake_image_data").decode())
    assert 'image' in parse_content(file)

    with pytest.raises(ValueError):
        parse_content(123)

def test_query_to_messages():
    query = Query([
        "User message",
        Assistant("Assistant message"),
        File(b64type="image/png", content=base64.b64encode(b"fake_image_data").decode())
    ])
    messages = list(query_to_messages(query))
    assert len(messages) == 3
    assert messages[0]['role'] == 'user'
    assert messages[1]['role'] == 'assistant'
    assert messages[2]['role'] == 'user'

def test_tool_result_to_content():
    str_result = ToolResult(id="1", content=ToolValue("text result"))
    assert tool_result_to_content(str_result) == [{'text': "text result"}]

    file_result = ToolResult(id="2", content=ToolValue(File(b64type="image/png", content=base64.b64encode(b"fake_image_data").decode())))
    assert 'image' in tool_result_to_content(file_result)[0]

    list_result = ToolResult(id="3", content=ToolValue(["text1", "text2"]))
    assert tool_result_to_content(list_result) == [{'text': "text1"}, {'text': "text2"}]

    with pytest.raises(ValueError):
        tool_result_to_content(ToolResult(id="4", content=ToolValue(123)))

def test_join_content():
    assert join_content("text1", "text2") == [{'text': "text1text2"}]
    assert len(join_content([{'image': {}}], [{'text': "text"}])) == 2

def test_merge_messages():
    messages = [
        {'role': 'user', 'content': [{'text': "Hello"}]},
        {'role': 'user', 'content': [{'text': " world"}]},
        {'role': 'assistant', 'content': [{'text': "Hi"}]},
    ]
    merged = merge_messages(messages)
    assert len(merged) == 2
    assert merged[0]['content'] == [{'text': "Hello world"}]

def test_deltas_to_messages():
    deltas = [
        Query(["User message"]),
        Delta(content=DeltaText(data="Assistant response")),
        Delta(content=DeltaToolUse(data=ToolUse(id="1", name="tool1", input={"param": "value"}))),
        Delta(content=DeltaToolResult(data=ToolResult(id="1", content=ToolValue("Tool result")))),
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 3
    assert messages[0]['role'] == 'user'
    assert 'text' in messages[0]['content'][0]

    assert messages[1]['role'] == 'assistant'
    assert 'text' in messages[1]['content'][0]
    assert 'toolUse' in messages[1]['content'][1]
    
    assert messages[2]['role'] == 'user'
    assert 'toolResult' in messages[2]['content'][0]

def test_deltas_to_messages_with_reasoning():
    deltas = [
        Query(["User question"]),
        # First delta with reasoning content
        Delta(content=DeltaText(
            subtype="reasoning",
            data="Let me think about this problem step by step."
        )),
        # Second delta with more reasoning content
        Delta(content=DeltaText(
            subtype="reasoning",
            data=" First, I need to understand the context."
        )),
        # Regular text response after reasoning
        Delta(content=DeltaText(
            data="Based on my analysis, the answer is..."
        ))
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[1]['role'] == 'assistant'
    assert len(messages[1]['content']) == 2
    assert messages[1]['content'][0]['reasoningContent']['reasoningText']['text'] == "Let me think about this problem step by step. First, I need to understand the context."
    assert messages[1]['content'][1]['text'] == "Based on my analysis, the answer is..."


def test_deltas_to_messages_reasoning_preserves_signature():
    # Round-trip: signature stored in DeltaText.metadata must be written back
    # into the reasoningContent block.
    deltas = [
        Delta(
            content=DeltaText(subtype="reasoning", data="Step-by-step reasoning"),
            metadata={"signature": "sig-abc"},
        ),
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 1
    reasoning_text = messages[0]['content'][0]['reasoningContent']['reasoningText']
    assert reasoning_text['text'] == "Step-by-step reasoning"
    assert reasoning_text['signature'] == "sig-abc"


def test_deltas_to_messages_reasoning_redacted_content():
    # redactedContent in metadata must produce a redactedContent block and
    # suppress the reasoningText block entirely.
    deltas = [
        Delta(
            content=DeltaText(subtype="reasoning", data=""),
            metadata={"redactedContent": b"redacted-bytes"},
        ),
    ]
    messages = deltas_to_messages(deltas)
    assert len(messages) == 1
    reasoning_content = messages[0]['content'][0]['reasoningContent']
    assert reasoning_content['redactedContent'] == b"redacted-bytes"
    assert 'reasoningText' not in reasoning_content
    
def test_merge_messages_with_reasoning():
    # Test merging messages with reasoning content
    messages = [
        {
            'role': 'assistant', 
            'content': [{
                'reasoningContent': {
                    'reasoningText': {
                        'text': 'First reasoning part', 
                        'signature': 'sig1'
                    }
                }
            }]
        },
        {
            'role': 'assistant', 
            'content': [{
                'reasoningContent': {
                    'reasoningText': {
                        'text': ' second reasoning part', 
                        'signature': None
                    }
                }
            }]
        },
        {
            'role': 'assistant',
            'content': [{'text': 'First text response'}]
        },
        {
            'role': 'assistant',
            'content': [{'text': ' continued text response'}]
        }
    ]
    merged = merge_messages(messages)
    assert len(merged) == 1
    assert merged[0]['role'] == 'assistant'
    
    # Check that we have two content blocks - one for reasoning and one for text
    assert len(merged[0]['content']) == 2
    
    # Check reasoning content was merged correctly
    reasoning_block = merged[0]['content'][0]
    assert reasoning_block['reasoningContent']['reasoningText']['text'] == 'First reasoning part second reasoning part'
    assert reasoning_block['reasoningContent']['reasoningText']['signature'] == 'sig1'
    
    # Check text content was merged correctly
    assert merged[0]['content'][1]['text'] == 'First text response continued text response'

@pytest.mark.asyncio
async def test_bedrock_llm_stream(bedrock_llm):
    async def async_stream_iter():
        events = [
            {'contentBlockDelta': {'delta': {'text': 'Response text'}}},
            {'contentBlockStart': {'start': {'toolUse': {'name': 'tool1', 'toolUseId': '1'}}}},
            {'contentBlockDelta': {'delta': {'toolUse': {'input': '{"param":'}}}},
            {'contentBlockDelta': {'delta': {'toolUse': {'input': ' "value"}'}}}},
            {'contentBlockStop': {}},
            {'messageStop': {'stopReason': 'tool_use'}},
            {'metadata': {'usage': {'inputTokens': 10, 'outputTokens': 20, 'cacheReadInputTokens': 3, 'cacheWriteInputTokens': 7}}},
        ]
        for event in events:
            yield event

    mock_response = {
        'stream': async_stream_iter()
    }
    bedrock_llm.client.converse_stream.return_value = mock_response

    results = [delta async for delta in bedrock_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
    )]

    assert len(results) == 5
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Response text"
    assert isinstance(results[1].content, DeltaToolInputUpdate)
    assert results[1].content.data == '{"param":'
    assert isinstance(results[2].content, DeltaToolInputUpdate)
    assert results[2].content.data == ' "value"}'
    assert isinstance(results[3].content, DeltaToolUse)
    assert results[3].content.data == ToolUse(id="1", name="tool1", input={"param": "value"})
    assert results[4].usage == Usage(input_tokens=10, output_tokens=20, cache_read_input_tokens=3, cache_written_input_tokens=7)

@pytest.mark.asyncio
async def test_bedrock_llm_stream_reasoning_metadata(bedrock_llm):
    # Stream side of the round-trip: reasoningContent must surface as a
    # reasoning DeltaText carrying signature/redactedContent in metadata.
    async def async_stream_iter():
        events = [
            {'contentBlockDelta': {'delta': {'reasoningContent': {'text': 'Let me think', 'signature': 'sig-xyz'}}}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ]
        for event in events:
            yield event

    bedrock_llm.client.converse_stream.return_value = {'stream': async_stream_iter()}

    results = [delta async for delta in bedrock_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
    )]

    assert len(results) == 1
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.subtype == "reasoning"
    assert results[0].content.data == "Let me think"
    assert results[0].metadata["signature"] == "sig-xyz"

@pytest.mark.asyncio
async def test_bedrock_llm_no_stream(bedrock_llm):
    mock_response = {
        'output': {
            'message': {
                'content': [
                    {'text': 'Response text'},
                    {'toolUse': {'toolUseId': '1', 'name': 'tool1', 'input': {'param': 'value'}}},
                ]
            }
        },
        'stopReason': 'tool_use',
        'usage': {'inputTokens': 10, 'outputTokens': 20, 'cacheReadInputTokens': 7, 'cacheWriteInputTokens': 3},
    }
    bedrock_llm.client.converse.return_value = mock_response

    results = [delta async for delta in bedrock_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[],
        no_stream=True,
    )]

    assert len(results) == 3
    assert isinstance(results[0].content, DeltaText)
    assert results[0].content.data == "Response text"
    assert isinstance(results[1].content, DeltaToolUse)
    assert results[1].content.data == ToolUse(id="1", name="tool1", input={"param": "value"})
    assert results[2].usage == Usage(input_tokens=10, output_tokens=20, cache_read_input_tokens=7, cache_written_input_tokens=3)

@pytest.mark.asyncio
async def test_bedrock_llm_cache_options(bedrock_llm):
    async def async_stream_iter():
        events = [
            {'contentBlockDelta': {'delta': {'text': 'Response text'}}},
            {'contentBlockStop': {}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {'inputTokens': 10, 'outputTokens': 20}}},
        ]
        for event in events:
            yield event

    mock_response = {
        'stream': async_stream_iter()
    }


    bedrock_llm.client.converse_stream.return_value = mock_response

    async def dummy_tool(hello: int) -> str:
        """Dummy tool for testing"""
        return "dummy response"

    results = [delta async for delta in bedrock_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[to_schema(dummy_tool)],
        cache={
            "cache_system_prompt": True,
            "cache_tools": True,
        },
    )]
    assert bedrock_llm.client.converse_stream.call_count == 1
    call_args = bedrock_llm.client.converse_stream.call_args[1]
    assert call_args["system"][0]["text"] == "Test prompt"
    assert call_args["system"][1] == {'cachePoint': {'type': 'default'}}

    assert call_args["toolConfig"]["tools"][0]["toolSpec"]["name"] == "dummy_tool"
    assert call_args["toolConfig"]["tools"][1] == {'cachePoint': {'type': 'default'}}

    # no cache
    async def async_stream_iter2():
        events = [
            {'contentBlockDelta': {'delta': {'text': 'Response text'}}},
            {'contentBlockStop': {}},
            {'messageStop': {'stopReason': 'end_turn'}},
            {'metadata': {'usage': {'inputTokens': 10, 'outputTokens': 20}}},
        ]
        for event in events:
            yield event

    bedrock_llm.client.converse_stream.return_value = {
        'stream': async_stream_iter2()
    }

    results_no_cache = [delta async for delta in bedrock_llm.stream(
        prompt="Test prompt",
        model="test-model",
        history=[],
        functions=[to_schema(dummy_tool)],
        cache=None,
    )]
    assert bedrock_llm.client.converse_stream.call_count == 2
    call_args_no_cache = bedrock_llm.client.converse_stream.call_args[1]
    assert len(call_args_no_cache["system"]) == 1, "System messages should not include cache info when cache is None"
    assert call_args_no_cache["system"][0]["text"] == "Test prompt"

    assert len(call_args_no_cache["toolConfig"]["tools"]) == 1, "Tool config should not include cache info when cache is None"
    assert call_args_no_cache["toolConfig"]["tools"][0]["toolSpec"]["name"] == "dummy_tool"


# --- Exception handling tests ---

import botocore.exceptions
from mus.llm.exceptions import (
    LLMAuthenticationException,
    LLMRateLimitException,
    LLMConnectionException,
    LLMTimeoutException,
    LLMBadRequestException,
    LLMServerException,
    LLMNotFoundException,
    LLMModelException,
    LLMToolParseException,
    LLMCachingException,
    LLMException,
)


def _make_client_error(code, message="error", status_code=400, request_id="req-123"):
    """Helper to create a botocore ClientError."""
    return botocore.exceptions.ClientError(
        error_response={
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {
                "RequestId": request_id,
                "HTTPStatusCode": status_code,
            },
        },
        operation_name="ConverseStream",
    )


# --- Direct mapping function tests ---

def test_map_bedrock_throttling_exception():
    err = _make_client_error("ThrottlingException", status_code=429, request_id="req-map-1")
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 429
    assert mapped.request_id == "req-map-1"


def test_map_bedrock_access_denied():
    err = _make_client_error("AccessDeniedException", status_code=403)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 403


def test_map_bedrock_caching_not_supported():
    # Bedrock returns this when a cache point is sent to a model that does not
    # support prompt caching. It arrives as AccessDeniedException, but the cause
    # is a caching feature mismatch, not credentials.
    err = _make_client_error(
        "AccessDeniedException",
        message=(
            "You invoked an unsupported model or your request did not allow "
            "prompt caching. See the documentation for more information."
        ),
        status_code=403,
    )
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMCachingException)
    # Still a caching error even though Bedrock used the AccessDenied code, so it
    # must not be mistaken for an auth failure.
    assert not isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 403


def test_map_bedrock_validation_exception():
    err = _make_client_error("ValidationException", status_code=400)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMBadRequestException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 400


def test_map_bedrock_resource_not_found():
    err = _make_client_error("ResourceNotFoundException", status_code=404)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMNotFoundException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 404


def test_map_bedrock_internal_server_exception():
    err = _make_client_error("InternalServerException", status_code=500)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMServerException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 500


def test_map_bedrock_model_error_exception():
    err = _make_client_error("ModelErrorException", status_code=424)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMModelException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_no_credentials():
    err = botocore.exceptions.NoCredentialsError()
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMAuthenticationException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_endpoint_connection_error():
    err = botocore.exceptions.EndpointConnectionError(endpoint_url="https://example.com")
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMConnectionException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_connect_timeout():
    err = botocore.exceptions.ConnectTimeoutError(endpoint_url="https://example.com")
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMTimeoutException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_read_timeout():
    err = botocore.exceptions.ReadTimeoutError(endpoint_url="https://example.com")
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMTimeoutException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_service_quota_exceeded():
    err = _make_client_error("ServiceQuotaExceededException", status_code=429)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "bedrock"


def test_map_bedrock_unknown_client_error():
    err = _make_client_error("SomeUnknownException", status_code=418)
    mapped = _map_bedrock_exception(err)
    assert isinstance(mapped, LLMException)
    assert not isinstance(mapped, LLMRateLimitException)
    assert mapped.provider == "bedrock"
    assert mapped.status_code == 418


# --- stream() integration tests ---

@pytest.mark.asyncio
async def test_bedrock_auth_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("AccessDeniedException", status_code=403)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.status_code == 403
    assert exc_info.value.request_id == "req-123"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_bedrock_rate_limit_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("ThrottlingException", status_code=429)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_bedrock_connection_error(bedrock_llm, mock_bedrock_client):
    exc = botocore.exceptions.EndpointConnectionError(endpoint_url="https://example.com")
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMConnectionException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_bedrock_timeout_error(bedrock_llm, mock_bedrock_client):
    exc = botocore.exceptions.ConnectTimeoutError(endpoint_url="https://example.com")
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMTimeoutException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_bedrock_server_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("InternalServerException", status_code=500)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMServerException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_bedrock_not_found_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("ResourceNotFoundException", status_code=404)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMNotFoundException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_bedrock_no_credentials_error(bedrock_llm, mock_bedrock_client):
    exc = botocore.exceptions.NoCredentialsError()
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMAuthenticationException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_bedrock_stream_iteration_error(bedrock_llm, mock_bedrock_client):
    """SDK exception during stream iteration is mapped correctly."""
    exc = _make_client_error("ThrottlingException", status_code=429)

    async def failing_stream():
        yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
        raise exc

    mock_bedrock_client.converse_stream.return_value = {"stream": failing_stream()}

    with pytest.raises(LLMRateLimitException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.__cause__ is exc


@pytest.mark.asyncio
async def test_bedrock_tool_parse_error(bedrock_llm, mock_bedrock_client):
    """Malformed tool JSON raises LLMToolParseException."""
    async def stream_with_bad_tool():
        yield {"contentBlockStart": {"start": {"toolUse": {"name": "my_tool", "toolUseId": "t1"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{invalid json"}}}}
        yield {"contentBlockStop": {}}

    mock_bedrock_client.converse_stream.return_value = {"stream": stream_with_bad_tool()}

    with pytest.raises(LLMToolParseException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"


@pytest.mark.asyncio
async def test_bedrock_model_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("ModelErrorException", status_code=424)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMModelException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"


@pytest.mark.asyncio
async def test_bedrock_bad_request_error(bedrock_llm, mock_bedrock_client):
    exc = _make_client_error("ValidationException", status_code=400)
    mock_bedrock_client.converse_stream.side_effect = exc

    with pytest.raises(LLMBadRequestException) as exc_info:
        async for _ in bedrock_llm.stream(prompt="p", history=[], model="m"):
            pass
    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_json,expected_args", [
    ('{"name": "test", "value": 42,}', {"name": "test", "value": 42}),
    ("{'name': 'test', 'value': 42}", {"name": "test", "value": 42}),
    ('{"name": "test", "value": 42', {"name": "test", "value": 42}),
    ('{name: "test", value: 42}', {"name": "test", "value": 42}),
])
async def test_bedrock_json_repair(bedrock_llm, mock_bedrock_client, malformed_json, expected_args):
    """Malformed but repairable tool JSON is repaired during streaming."""
    async def stream_with_repairable_tool():
        yield {"contentBlockStart": {"start": {"toolUse": {"name": "my_tool", "toolUseId": "t1"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": malformed_json}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}

    mock_bedrock_client.converse_stream.return_value = {"stream": stream_with_repairable_tool()}

    deltas = []
    async for d in bedrock_llm.stream(prompt="p", history=[], model="m"):
        deltas.append(d)

    tool_uses = [d for d in deltas if isinstance(d.content, DeltaToolUse)]
    assert len(tool_uses) == 1
    assert tool_uses[0].content.data.input == expected_args


def test_cache_point_inserts_block():
    messages = deltas_to_messages([Query(["big", CachePoint(), "tail"])])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert len(content) == 3
    assert content[0]["text"] == "big"
    assert content[1] == {"cachePoint": {"type": "default"}}
    assert content[2]["text"] == "tail"


def test_cache_point_leading_is_noop():
    messages = deltas_to_messages([Query([CachePoint(), "hello"])])
    assert len(messages) == 1
    content = messages[0]["content"]
    assert all("cachePoint" not in block for block in content)
    assert content[0]["text"] == "hello"


def test_no_cache_point_still_merges_text():
    messages = deltas_to_messages([Query(["a", "b"])])
    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["text"] == "ab"


def _make_caching_error():
    return _make_client_error(
        "AccessDeniedException",
        message=(
            "You invoked an unsupported model or your request did not allow "
            "prompt caching."
        ),
        status_code=403,
    )


async def _ok_stream():
    events = [
        {"contentBlockDelta": {"delta": {"text": "ok"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
    ]
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_warn_on_unsupported_cache_retries_without_cache_points(caplog):
    client = AsyncMock()
    client.converse_stream.side_effect = [
        _make_caching_error(),
        {"stream": _ok_stream()},
    ]
    llm = BedrockLLM("a-model-id", client, warn_on_unsupported_cache=True)

    with caplog.at_level(logging.WARNING):
        deltas = [
            d
            async for d in llm.stream(
                model="a-model-id",
                history=[Query(["big context", CachePoint(), "the question"])],
            )
        ]

    # Retried exactly once after the caching failure.
    assert client.converse_stream.call_count == 2
    # The retry stripped the cache point out of the request.
    retry_args = client.converse_stream.call_args_list[1].kwargs
    assert all(
        "cachePoint" not in block
        for message in retry_args["messages"]
        for block in message["content"]
    )
    # Output still flowed through, and a warning was logged.
    assert any(isinstance(d.content, DeltaText) and d.content.data for d in deltas)
    assert "caching" in caplog.text.lower()


def test_add_history_cache_point_appends_block():
    messages = deltas_to_messages([Query(["hello world"])])
    add_history_cache_point(messages)
    assert messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}


def test_add_history_cache_point_empty_is_noop():
    messages = []
    add_history_cache_point(messages)
    assert messages == []


@pytest.mark.asyncio
async def test_stream_applies_cache_history():
    client = AsyncMock()
    client.converse_stream.return_value = {"stream": _ok_stream()}
    llm = BedrockLLM("a-model-id", client)
    async for _ in llm.stream(
        model="a-model-id",
        history=[Query(["remember this"])],
        cache={"cache_history": True},
    ):
        pass
    messages = client.converse_stream.call_args.kwargs["messages"]
    assert messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}


@pytest.mark.asyncio
async def test_stream_without_cache_history_has_no_breakpoint():
    client = AsyncMock()
    client.converse_stream.return_value = {"stream": _ok_stream()}
    llm = BedrockLLM("a-model-id", client)
    async for _ in llm.stream(
        model="a-model-id",
        history=[Query(["remember this"])],
    ):
        pass
    messages = client.converse_stream.call_args.kwargs["messages"]
    assert all(
        "cachePoint" not in block
        for message in messages
        for block in message["content"]
    )


@pytest.mark.asyncio
async def test_unsupported_cache_raises_without_flag():
    client = AsyncMock()
    client.converse_stream.side_effect = _make_caching_error()
    llm = BedrockLLM("a-model-id", client)  # default: warn_on_unsupported_cache=False

    with pytest.raises(LLMCachingException):
        async for _ in llm.stream(
            model="a-model-id",
            history=[Query(["big context", CachePoint(), "the question"])],
        ):
            pass
    assert client.converse_stream.call_count == 1