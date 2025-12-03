import pytest
from unittest.mock import Mock, AsyncMock
from mus.llm.bedrock import (
    BedrockLLM,
    functions_for_llm,
    file_to_image,
    str_to_text_block,
    parse_content,
    query_to_messages,
    tool_result_to_content,
    join_content,
    merge_messages,
    deltas_to_messages,
)
from mus.llm.types import DeltaToolInputUpdate, File, Query, Delta, ToolUse, ToolResult, Assistant, DeltaContent, DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory, Usage, ToolValue
from mus.functions import to_schema
from dataclasses import dataclass
import base64

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
        'usage': {'inputTokens': 10, 'outputTokens': 20, 'cacheReadInputTokens': 7, 'cacheWrittenInputTokens': 3},
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