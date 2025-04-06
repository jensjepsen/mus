import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import anthropic.types as at
from anthropic.lib.streaming import TextEvent, ContentBlockStopEvent

from mus.llm import LLM
from mus.llm.llm import IterableResult
from mus.llm.types import Delta, ToolUse, ToolResult
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
    
if __name__ == "__main__":
    pytest.main()