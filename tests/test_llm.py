import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import anthropic.types as at
from anthropic.lib.streaming import TextEvent, ContentBlockStopEvent

from src.mus.llm import LLM
from src.mus.llm.llm import IterableResult
from src.mus.llm.types import Delta, ToolUse, ToolResult
from src.mus.llm.anthropic import AnthropicLLM

class MockClient():
    def __init__(self):
        self.messages = MagicMock()
    
    def set_response(self, responses):
        mock_response = MagicMock()
        mock_response.__iter__.return_value = iter(responses)
        self.messages.stream.return_value.__enter__.return_value = mock_response

@dataclass
class TestStructure:
    field1: str
    field2: int

@pytest.fixture
def mock_client():
    return MockClient()

@pytest.fixture
def llm(mock_client):
    return LLM(prompt="Test prompt", client=AnthropicLLM(client=mock_client))

def test_llm_query(llm, mock_client):
    mock_client.set_response([
        TextEvent(type="text", text="Hello", snapshot="Hello"),
        at.MessageStopEvent(type="message_stop", message=at.Message(role="assistant", content=[{"type": "text", "text": "Hello"}], stop_reason="end_turn", model="anthropic.claude-3-5-sonnet-20240620-v1:0", id="test_tool", stop_sequence="0", type="message", usage=at.Usage(input_tokens="0", output_tokens="0")))
    ])
    
    result = list(llm.query("Test query"))
    assert len(result) == 1
    assert isinstance(result[0], Delta)
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Hello"

def test_llm_query_with_tool_use(llm, mock_client):
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
                    input_tokens="0",
                    output_tokens="0",
                )
            )
        )
    ])

    called = False

    def test_tool(**kwargs):
        """Test tool function"""
        nonlocal called
        called = True
        return "Tool result"

    result = list(llm.query("Test query", functions=[test_tool]))
    assert called, "Tool function was not called"
    assert len(result) == 3
    assert result[0].content["type"] == "text"
    assert result[1].content["type"] == "tool_use"
    assert result[2].content["type"] == "tool_result"
    assert isinstance(result[1].content["data"], ToolUse)
    assert isinstance(result[2].content["data"], ToolResult)

@patch('src.mus.llm.LLM.query')
def test_llm_call(mock_query, llm):
    mock_query.return_value = iter([Delta(content={"data": "Test response", "type": "text"})])
    result = llm("Test query")

    assert isinstance(result, IterableResult)
    assert str(result) == "Test response"

@patch('src.mus.llm.LLM.query')
def test_llm_fill(mock_query, llm):
    mock_query.return_value = iter(list([
        Delta(content={"data": "Processing", "type": "text"}),
        Delta(content={"data": ToolResult(content=TestStructure(field1="test", field2=123), id="abc"), "type": "tool_result"})
    ]))
    result = llm.fill("Test query", TestStructure)
    assert isinstance(result, TestStructure)
    assert result.field1 == "test"
    assert result.field2 == 123

@patch('src.mus.llm.LLM.query')
def test_llm_bot_decorator(mock_query, llm):
    mock_query.return_value = iter([Delta(content={"type":"text", "data":"Test response"})])
    
    @llm.bot
    def bot(query: str):
        return query

    assert str(bot("Test query")) == "Test response"

def test_iterable_result():
    deltas = [
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"data": ToolUse(name="test_tool", input={}, id="abc"), "type": "tool_use"}),
        Delta(content={"data": ToolResult(content="Tool output", id="abc"), "type": "tool_result"})
    ]
    result = IterableResult(deltas)
    
    assert str(result) == "HelloRunning tool: test_toolTool applied"
    
    with pytest.raises(TypeError):
        _ = result + 1

    assert result + " World" == "HelloRunning tool: test_toolTool applied World"

if __name__ == "__main__":
    pytest.main()