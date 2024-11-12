import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import anthropic.types as at
from anthropic.lib.streaming import TextEvent, ContentBlockStopEvent

from mus.llm import LLM
from mus.llm.llm import IterableResult
from mus.llm.types import Delta, ToolUse, ToolResult
from mus.llm.anthropic import AnthropicLLM

@dataclass
class TestStructure:
    field1: str
    field2: int

@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def llm(mock_client):
    return LLM(prompt="Test prompt", client=AnthropicLLM(mock_client))

def test_llm_query(llm, mock_client):
    mock_response = MagicMock()
    mock_response.__iter__.return_value = [
        TextEvent(type="text", text="Hello", snapshot="Hello"),
        at.MessageStopEvent(type="message_stop", message=at.Message(role="assistant", content=[{"type": "text", "text": "Hello"}], stop_reason="end_turn", model="anthropic.claude-3-5-sonnet-20240620-v1:0", id="test_tool", stop_sequence="0", type="message", usage=at.Usage(input_tokens="0", output_tokens="0")))
    ]
    mock_client.messages.stream.return_value.__enter__.return_value = mock_response

    result = list(llm.query("Test query"))
    assert len(result) == 1
    assert isinstance(result[0], Delta)
    assert result[0].type == "text"
    assert result[0].content == "Hello"

def test_llm_query_with_tool_use(llm, mock_client):
    mock_response = MagicMock()
    mock_response.__iter__.return_value = iter([
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
                    input_tokens="0",
                    output_tokens="0",
                )
            )
        )
    ])
    mock_client.messages.stream.return_value.__enter__.return_value = mock_response

    called = False

    def test_tool(**kwargs):
        nonlocal called
        called = True
        return "Tool result"

    result = list(llm.query("Test query", functions=[test_tool]))
    assert called, "Tool function was not called"
    assert len(result) == 3
    assert result[0].type == "text"
    assert result[1].type == "tool_use"
    assert result[2].type == "tool_result"
    assert isinstance(result[1].content, ToolUse)
    assert isinstance(result[2].content, ToolResult)

@patch('bob.llm.LLM.query')
def test_llm_call(mock_query, llm):
    mock_query.return_value = iter([Delta(type="text", content="Test response")])
    result = llm("Test query")
    assert isinstance(result, IterableResult)
    assert str(result) == "Test response"

@patch('bob.llm.LLM.query')
def test_llm_structured(mock_query, llm):
    mock_query.return_value = iter([
        Delta(type="text", content="Processing"),
        Delta(type="tool_result", content=ToolResult(content=TestStructure(field1="test", field2=123)))
    ])
    result = llm.structured("Test query", TestStructure)
    assert isinstance(result, TestStructure)
    assert result.field1 == "test"
    assert result.field2 == 123

def test_iterable_result():
    bot = Mock()
    deltas = [
        Delta(type="text", content="Hello"),
        Delta(type="tool_use", content=ToolUse(name="test_tool", input={})),
        Delta(type="tool_result", content=ToolResult(content="Tool output"))
    ]
    result = IterableResult(deltas, bot)
    
    assert str(result) == "HelloRunning tool: test_toolTool result: Tool output"
    
    with pytest.raises(TypeError):
        result + 1

    assert result + " World" == "HelloRunning tool: test_toolTool result: Tool output World"

    # Test the __call__ method
    result()
    bot.assert_called_once_with(history=result.history)

if __name__ == "__main__":
    pytest.main()