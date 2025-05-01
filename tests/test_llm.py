import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import anthropic.types as at
from anthropic.lib.streaming import TextEvent, ContentBlockStopEvent

from mus.llm import LLM
from mus.llm.llm import IterableResult, merge_history
from mus.llm.types import Delta, ToolUse, ToolResult, System, Query, Assistant, File
from mus.llm.anthropic import AnthropicLLM

class MockClient():
    def __init__(self):
        self.messages = MagicMock()
    
    def set_response(self, responses):
        mock_response = MagicMock()
        mock_response.__aiter__.return_value = iter(responses)
        self.messages.stream.return_value.__aenter__.return_value = mock_response

@dataclass
class TestStructure:
    field1: str
    field2: int

@pytest.fixture
def mock_client():
    return MockClient()

@pytest.fixture
def llm(mock_client):
    return LLM(prompt="Test prompt", model=AnthropicLLM(model="claude-3-5-sonnet-20241022", client=mock_client))

@pytest.fixture
def mock_model():
    class MockLLM(MagicMock):
        def set_response(self, responses):
            self.stream.return_value.__aiter__.return_value = iter(responses)
    
    return MockLLM()

@pytest.mark.asyncio
async def test_llm_query(llm, mock_client):
    mock_client.set_response([
        TextEvent(type="text", text="Hello", snapshot="Hello"),
        at.MessageStopEvent(
            type="message_stop",
            message=at.Message(
                role="assistant",
                content=[
                    {"type": "text", "text": "Hello"}
                ],
                stop_reason="end_turn",
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                id="test_tool",
                stop_sequence="0",
                type="message",
                usage=at.Usage(input_tokens=50, output_tokens=30)
            )
        )
    ])
    
    result = [msg async for msg in llm.query("Test query")]
    assert len(result) == 3
    assert isinstance(result[0], Delta)
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Hello"
    assert isinstance(result[0], Delta)
    assert result[1].usage is not None
    assert result[1].usage["input_tokens"] == 50
    assert result[1].usage["output_tokens"] == 30

@pytest.mark.asyncio
async def test_llm_query_with_tool_use(llm, mock_client):
    mock_client.set_response([
        TextEvent(type="text", text="Using tool", snapshot="Using tool"),
        ContentBlockStopEvent(
            index=0,
            type="content_block_stop",
            content_block=at.ToolUseBlock(
                type="tool_use",
                input={"param1": "test", "param2": 123},
                id="test_tool",
                name="test_tool",

            )
        ),
        at.MessageStopEvent(
            type="message_stop",
            message=at.Message(
                role="assistant",
                content=[{"type": "text", "text": "Using tool"}],
                stop_reason="tool_use",
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                id="test_tool",
                stop_sequence="0",
                type="message",
                usage=at.Usage(
                    input_tokens=10,
                    output_tokens=10,
                )
            )
        )
    ])

    called = False

    async def test_tool(**kwargs):
        """Test tool function"""
        nonlocal called
        called = True
        return "Tool result"

    result = [msg async for msg in llm.query("Test query", functions=[test_tool])]

    assert called, "Tool function was not called"
    assert len(result) == 5
    assert result[0].content["type"] == "text"
    assert result[2].content["type"] == "tool_use"
    assert result[3].content["type"] == "tool_result"
    assert isinstance(result[2].content["data"], ToolUse)
    assert isinstance(result[3].content["data"], ToolResult)

@pytest.mark.asyncio
@patch('mus.llm.LLM.query')
async def test_llm_call(mock_query, llm):
    async def return_value():
        for d in [Delta(content={"data": "Test response", "type": "text"})]:
            yield d
    mock_query.return_value = return_value()
    result = llm("Test query")

    assert isinstance(result, IterableResult)
    assert (await result.string()) == "Test response"

@pytest.mark.asyncio
@patch('mus.llm.LLM.query')
async def test_llm_fill(mock_query, llm):
    async def return_value():
        for d in [Delta(content={"data": "Processing", "type": "text"}), Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"field1": "test", "field2": 123}, id="abc")})]:
            yield d
    mock_query.return_value = return_value()
    result = await llm.fill("Test query", TestStructure)
    assert isinstance(result, TestStructure), f"Expected TestStructure, got {type(result)}"
    assert result.field1 == "test", f"Expected 'test', got {result.field1}"
    assert result.field2 == 123, f"Expected 123, got {result.field2}"

@pytest.mark.asyncio
async def test_llm_fill_strategy_prefill(mock_model):
    
        # below looks a little weird because the answer is prefilled, so it's missing the initial json
    responses = [
                Delta(
                    content={
                        "type": "text",
                        "data": """
                            "test",
                        """
                    }
                ),
                Delta(
                    content={
                        "type": "text",
                        "data": """
                            "field2": 1234
                            }
                        """
                    }
                )
            ]
    mock_model.set_response(responses)
    llm = LLM(prompt="Test prompt", model=mock_model)
    result = await llm.fill("Test query", TestStructure, strategy="prefill")
    assert isinstance(result, TestStructure)
    assert result.field1 == "test"
    assert result.field2 == 1234

@pytest.mark.asyncio
@patch('mus.llm.LLM.query')
async def test_llm_bot_decorator(mock_query, llm):
    async def return_value():
        for d in [Delta(content={"data": "Test response", "type": "text"})]:
            yield d
    mock_query.return_value = return_value()
    
    @llm.bot
    def bot(query: str):
        return query

    assert await bot("Test query").string() == "Test response"

@pytest.mark.asyncio
async def test_iterable_result():
    deltas = [
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"data": ToolUse(name="test_tool", input={}, id="abc"), "type": "tool_use"}),
        Delta(content={"data": ToolResult(content="Tool output", id="abc"), "type": "tool_result"})
    ]
    async def return_value():
        for d in deltas:
            yield d
    
    result = IterableResult(return_value())
    
    assert await result.string() == "HelloRunning tool: test_toolTool applied"

@pytest.mark.asyncio
async def test_dynamic_system_prompt():
    mock_client = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = iter([
        Delta(content={"type": "text", "data": "Hello"}),
    ])
    llm = LLM("A system prompt", model=mock_client)
    
    await llm(System("Can be overwritten") + "Test query").string()

    assert mock_client.stream.called
    assert mock_client.stream.call_args[1]['prompt'] == "Can be overwritten"
    hist = mock_client.stream.call_args[1]['history']
    assert len(hist) == 1
    assert isinstance(hist[0], Query)
    assert hist[0].val == ["Test query"]


@pytest.mark.asyncio
async def test_assistant_prefill_echo(mock_model):
    mock_model.set_response([
        Delta(content={"type": "text", "data": "Hello"}),
    ])
    llm = LLM(prompt="Test prompt", model=mock_model)
    result = [msg async for msg in llm.query("Hello" + Assistant("Test prefill", echo=True))]
    

    assert len(result) == 3
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Test prefill"
    assert result[1].content["type"] == "text"
    assert result[1].content["data"] == "Hello"

@pytest.mark.asyncio
async def test_assistant_prefill_no_echo(mock_model):
    mock_model.set_response([
        Delta(content={"type": "text", "data": "Hello"}),
    ])
    llm = LLM(prompt="Test prompt", model=mock_model)
    result = [msg async for msg in llm.query("Hello" + Assistant("Test prefill", echo=False))]

    assert len(result) == 2
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Hello"



@pytest.mark.asyncio
async def test_merge_consecutive_text_deltas():
    """Test that consecutive text deltas with subtype 'text' are merged."""
    delta1 = Delta(content={"type": "text", "data": "Hello ", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert merged[0].content["type"] == "text"
    assert merged[0].content["data"] == "Hello world!"
    assert merged[0].content["subtype"] == "text"

@pytest.mark.asyncio
async def test_dont_merge_different_subtypes():
    """Test that text deltas with different subtypes are not merged."""
    delta1 = Delta(content={"type": "text", "data": "Hello", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "Reasoning", "subtype": "reasoning"})
    delta3 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert merged[0].content["data"] == "Hello"
    assert merged[1].content["data"] == "Reasoning"
    assert merged[2].content["data"] == "world!"

@pytest.mark.asyncio
async def test_dont_merge_different_types():
    """Test that deltas of different types are not merged."""
    delta1 = Delta(content={"type": "text", "data": "Hello", "subtype": "text"})
    delta2 = Delta(content={"type": "tool_use", "data": "some_tool"})
    delta3 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert merged[0].content["data"] == "Hello"
    assert merged[1].content["type"] == "tool_use"
    assert merged[2].content["data"] == "world!"

@pytest.mark.asyncio
async def test_prune_empty_text():
    """Test that empty text deltas are pruned."""
    delta1 = Delta(content={"type": "text", "data": "Hello", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "   ", "subtype": "text"})
    delta3 = Delta(content={"type": "text", "data": "\n\t", "subtype": "text"})
    delta4 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, delta2, delta3, delta4]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert merged[0].content["data"] == """Hello   \n\tworld!"""

@pytest.mark.asyncio
async def test_preserve_non_delta_messages():
    """Test that non-Delta messages in history are preserved."""
    delta1 = Delta(content={"type": "text", "data": "Hello", "subtype": "text"})
    non_delta = {"type": "some_other_type", "data": "preserved"}
    delta2 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, non_delta, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert merged[0].content["data"] == "Hello"
    assert merged[1] == non_delta
    assert merged[2].content["data"] == "world!"

@pytest.mark.asyncio
async def test_empty_history():
    """Test that an empty history returns an empty list."""
    history = []
    
    merged = merge_history(history)
    
    assert merged == []

@pytest.mark.asyncio
async def test_single_delta():
    """Test that a history with a single delta returns that delta."""
    delta = Delta(content={"type": "text", "data": "Hello", "subtype": "text"})
    history = [delta]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert merged[0].content["data"] == "Hello"

@pytest.mark.asyncio
async def test_merge_multiple_consecutive_text_deltas():
    """Test merging multiple consecutive text deltas."""
    delta1 = Delta(content={"type": "text", "data": "Hello ", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "beautiful ", "subtype": "text"})
    delta3 = Delta(content={"type": "text", "data": "world!", "subtype": "text"})
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert merged[0].content["data"] == "Hello beautiful world!"

@pytest.mark.asyncio
async def test_complex_history():
    """Test a complex history with various delta types and subtypes."""
    delta1 = Delta(content={"type": "text", "data": "Start ", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "of text.", "subtype": "text"})
    delta3 = Delta(content={"type": "tool_use", "data": "some_tool"})
    delta4 = Delta(content={"type": "tool_result", "data": "result"})
    delta5 = Delta(content={"type": "text", "data": "More ", "subtype": "reasoning"})
    delta6 = Delta(content={"type": "text", "data": "reasoning.", "subtype": "reasoning"})
    delta7 = Delta(content={"type": "text", "data": "End ", "subtype": "text"})
    delta8 = Delta(content={"type": "text", "data": "of text.", "subtype": "text"})
    
    history = [delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8]
    
    merged = merge_history(history)
    
    assert len(merged) == 6
    assert merged[0].content["data"] == "Start of text."
    assert merged[1].content["type"] == "tool_use"
    assert merged[2].content["type"] == "tool_result"
    assert merged[3].content["data"] == "More "
    assert merged[4].content["data"] == "reasoning."
    assert merged[5].content["data"] == "End of text."

@pytest.mark.asyncio
async def test_bug_empty_history_index_error():
    """Test that the function handles empty history without IndexError."""
    # This test is to ensure the function doesn't try to access merged[-1] when merged is empty
    history = [Delta(content={"type": "text", "data": "Hello", "subtype": "text"})]
    
    # This should not raise an IndexError
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert merged[0].content["data"] == "Hello"

# New tests for query types

@pytest.mark.asyncio
async def test_history_with_file_objects():
    """Test that File objects in history are preserved."""
    delta1 = Delta(content={"type": "text", "data": "Here's an image:", "subtype": "text"})
    file_obj = File(b64type="image/png", content="dummy_base64_content")
    delta2 = Delta(content={"type": "text", "data": "What do you think?", "subtype": "text"})
    history = [delta1, file_obj, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert merged[0].content["data"] == "Here's an image:"
    assert merged[1] == file_obj
    assert merged[2].content["data"] == "What do you think?"

@pytest.mark.asyncio
async def test_history_with_query_objects():
    """Test that Query objects in history are preserved."""
    delta1 = Delta(content={"type": "text", "data": "Processing query:", "subtype": "text"})
    query_obj = Query(["item1", "item2"])
    delta2 = Delta(content={"type": "text", "data": "Query processed", "subtype": "text"})
    history = [delta1, query_obj, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert merged[0].content["data"] == "Processing query:"
    assert merged[1] == query_obj
    assert merged[2].content["data"] == "Query processed"

@pytest.mark.asyncio
async def test_complex_history_with_query_types():
    """Test a complex history with various delta types and query types."""
    delta1 = Delta(content={"type": "text", "data": "Start ", "subtype": "text"})
    delta2 = Delta(content={"type": "text", "data": "of text.", "subtype": "text"})
    
    file_obj = File(b64type="image/png", content="dummy_base64_content")
    
    delta3 = Delta(content={"type": "tool_use", "data": "some_tool"})
    delta4 = Delta(content={"type": "tool_result", "data": "result"})
    
    query_obj = Query(["query_item"])
    assistant_obj = Assistant("test_assistant")
    
    delta5 = Delta(content={"type": "text", "data": "More ", "subtype": "reasoning"})
    delta6 = Delta(content={"type": "text", "data": "reasoning.", "subtype": "reasoning"})
    delta7 = Delta(content={"type": "text", "data": "End ", "subtype": "text"})
    delta8 = Delta(content={"type": "text", "data": "of text.", "subtype": "text"})
    
    history = [
        delta1, delta2, file_obj, delta3, delta4, 
        query_obj, assistant_obj, delta5, delta6, delta7, delta8
    ]
    
    merged = merge_history(history)
    
    assert len(merged) == 9
    assert merged[0].content["data"] == "Start of text."
    assert merged[1] == file_obj
    assert merged[2].content["type"] == "tool_use"
    assert merged[3].content["type"] == "tool_result"
    assert merged[4] == query_obj
    assert merged[5] == assistant_obj
    assert merged[6].content["data"] == "More "
    assert merged[7].content["data"] == "reasoning."
    assert merged[8].content["data"] == "End of text."

@pytest.mark.asyncio
async def test_history_with_mixed_query_types():
    """Test history with a mix of simple and complex query types."""
    delta1 = Delta(content={"type": "text", "data": "Text before", "subtype": "text"})
    
    # String query
    string_query = "Simple string query"
    
    # File query
    file_obj = File(b64type="image/png", content="dummy_base64_content")
    
    assistant_obj = Assistant("test_assistant")
    
    query_list = Query(["item1", file_obj, assistant_obj])
    
    delta2 = Delta(content={"type": "text", "data": "Text after", "subtype": "text"})
    
    history = [delta1, string_query, file_obj, assistant_obj, query_list, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 6
    assert merged[0].content["data"] == "Text before"
    assert merged[1] == string_query
    assert merged[2] == file_obj
    assert merged[3] == assistant_obj
    assert merged[4] == query_list
    assert merged[5].content["data"] == "Text after"

if __name__ == "__main__":
    pytest.main()