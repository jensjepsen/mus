import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
import json

from mus import Bot
from mus.llm.llm import IterableResult, merge_history, invoke_function
from mus.functions import parse_tools
from mus.llm.types import Delta, ToolUse, ToolResult, System, Query, Assistant, File, DeltaToolResult, DeltaText, DeltaToolUse, Usage, ToolValue

from mus import ToolNotFoundError
import typing as t

@dataclass
class TestStructure:
    field1: str
    field2: int

@pytest.fixture
def mock_model():
    class MockLLM(MagicMock):
        def set_response(self, responses):
            self.stream.return_value.__aiter__.return_value = iter(responses)
    
    return MockLLM()

@pytest.mark.asyncio
async def test_llm_query(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=50, output_tokens=30, cache_read_input_tokens=11, cache_written_input_tokens=3)),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    
    result = [msg async for msg in llm("Test query")]
    assert len(result) == 2
    assert isinstance(result[0], Delta)
    assert result[0].content.data == "Hello"
    assert isinstance(result[0], Delta)
    assert result[1].usage is not None
    assert result[1].usage.input_tokens == 50
    assert result[1].usage.output_tokens == 30
    
@pytest.mark.asyncio
async def test_llm_with_tool_use(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"param1": "test", "param2": 123}, id="test_tool"))),
        Delta(content=DeltaText(data="Tool used")),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)

    called = False

    async def test_tool(**kwargs):
        """Test tool function"""
        nonlocal called
        called = True
        return "Tool result"

    result = [msg async for msg in llm("Test query", functions=[test_tool])]
    assert called, "Tool function was not called"
    assert len(result) == 4
    assert isinstance(result[0].content, DeltaText)
    assert isinstance(result[1].content, DeltaToolUse)
    assert isinstance(result[1].content.data, ToolUse)

    assert isinstance(result[2].content.data, ToolResult)
    assert result[2].content.data.content.val == "Tool result"

    assert isinstance(result[3].content, DeltaText)
    assert result[3].content.data == "Tool used"

@pytest.mark.asyncio
async def test_llm_with_tool_use_nonexistent_function(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="nonexistent_function", input={"param1": "test", "param2": 123}, id="test_tool"))),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)

    async def test_tool(**kwargs):
        """Test tool function"""
        return "Tool result"

    with pytest.raises(ToolNotFoundError) as exc_info:
        _ = [msg async for msg in llm("Test query", functions=[test_tool])]
    assert "nonexistent_function" in str(exc_info.value)
    

@pytest.mark.asyncio
async def test_llm_with_fallback_tool_use(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="a_tool_that_does_not_exist", input={"param1": "test", "param2": 123}, id="test_tool"))),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)

    test_called = False

    async def test_tool(**kwargs):
        """Test tool function"""
        nonlocal test_called
        test_called = True
        return "Tool result"

    fallback_called = False
    async def fallback_tool(original_tool_name: str, original_input: t.Mapping[str, t.Any]):
        """Fallback tool function"""
        nonlocal fallback_called
        fallback_called = True
        assert original_tool_name == "a_tool_that_does_not_exist"
        assert original_input == {"param1": "test", "param2": 123}
        return "Fallback tool result"

    result = [msg async for msg in llm("Test query", functions=[test_tool], fallback_function=fallback_tool)]
    assert not test_called, "Tool function was not called"
    assert fallback_called, "Fallback tool function was not called"
    assert len(result) == 3
    assert isinstance(result[0].content, DeltaText)
    assert isinstance(result[1].content, DeltaToolUse)
    assert isinstance(result[1].content.data, ToolUse)

    assert isinstance(result[2].content.data, ToolResult)
    assert result[2].content.data.content.val == "Fallback tool result"

@pytest.mark.asyncio
async def test_llm_call(mock_model):
    mock_model.set_response([Delta(content=DeltaText(data="Test response"))])
    llm = Bot(prompt="Test prompt", model=mock_model)

    result = llm("Test query")

    assert isinstance(result, IterableResult)
    assert (await result.string()) == "Test response"

@pytest.mark.parametrize("strategy", ["tool_use", "prefill"])
@pytest.mark.asyncio
async def test_llm_fill(strategy, mock_model):
    if strategy == "tool_use":
        mock_model.set_response([
            Delta(content=DeltaText(data="Processing")),
            Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"field1": "test", "field2": 123}, id="abc")))
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content=DeltaText(data="""
"test",
"field2": 123
}
""")),
        ])
    llm = Bot(prompt="Test prompt", model=mock_model)
    result = await llm.fill("Test query", TestStructure, strategy=strategy)
    assert isinstance(result, TestStructure), f"Expected TestStructure, got {type(result)}"
    assert result.field1 == "test", f"Expected 'test', got {result.field1}"
    assert result.field2 == 123, f"Expected 123, got {result.field2}"

@pytest.mark.parametrize("strategy", ["tool_use", "prefill"])
@pytest.mark.asyncio
async def test_llm_fill_missing_arguments(strategy, mock_model):
    if strategy == "tool_use":
        mock_model.set_response([
            Delta(content=DeltaText(data="Processing")),
            Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"field1": "test"}, id="abc"))),
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content=DeltaText(data="""
"test"
}
```
"""
            )),
        ])
    
    llm = Bot(prompt="Test prompt", model=mock_model)
    
    with pytest.raises(ValueError) as exc_info:
        await llm.fill("Test query", TestStructure, strategy=strategy)
    
    assert "data must contain ['field2']" in str(exc_info.value)

@pytest.mark.parametrize("strategy", ["tool_use", "prefill"])
@pytest.mark.asyncio
async def test_llm_fill_wrong_argument_type(strategy, mock_model):
    if strategy == "tool_use":
        mock_model.set_response([
            Delta(content=DeltaText(data="Processing")),
            Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"field1": "hello", "field2": "not_an_int"}, id="abc"))),
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content=DeltaText(data="""
"test",
"field2": "not_an_int"
}
""")),
        ])
        
    llm = Bot(prompt="Test prompt", model=mock_model)
    
    with pytest.raises(ValueError) as exc_info:
        await llm.fill("Test query", TestStructure, strategy=strategy)
    assert "data.field2 must be int" in str(exc_info.value)

@pytest.mark.asyncio
async def test_llm_bot_decorator(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Test response"))
    ])
    llm = Bot(prompt="Test prompt", model=mock_model)
    @llm.bot
    def bot(query: str):
        return query

    assert await bot("Test query").string() == "Test response"

@pytest.mark.asyncio
async def test_iterable_result():
    deltas = [
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={}, id="abc"))),
        Delta(content=DeltaToolResult(data=ToolResult(content="Tool output", id="abc")))
    ]
    async def return_value():
        for d in deltas:
            yield d
    
    result = IterableResult(return_value())
    
    assert await result.string() == "HelloRunning tool: test_toolTool applied"

@pytest.mark.asyncio
async def test_dynamic_system_prompt(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
    ])
    
    llm = Bot("A system prompt", model=mock_model)
    
    await llm(System("Can be overwritten") + "Test query").string()

    assert mock_model.stream.called
    assert mock_model.stream.call_args[1]['prompt'] == "Can be overwritten"
    hist = mock_model.stream.call_args[1]['history']
    assert len(hist) == 1
    assert isinstance(hist[0], Query)
    assert hist[0].val == ["Test query"]


@pytest.mark.asyncio
async def test_assistant_prefill_echo(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
    ])
    llm = Bot(prompt="Test prompt", model=mock_model)
    result = [msg async for msg in llm.query("Hello" + Assistant("Test prefill", echo=True))]
    

    assert len(result) == 3
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Test prefill"
    assert isinstance(result[1].content, DeltaText)
    assert result[1].content.data == "Hello"

@pytest.mark.asyncio
async def test_assistant_prefill_no_echo(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
    ])
    llm = Bot(prompt="Test prompt", model=mock_model)
    result = [msg async for msg in llm.query("Hello" + Assistant("Test prefill", echo=False))]

    assert len(result) == 2
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Hello"



@pytest.mark.asyncio
async def test_merge_consecutive_text_deltas():
    """Test that consecutive text deltas with subtype 'text' are merged."""
    delta1 = Delta(content=DeltaText(data="Hello ", subtype="text"))
    delta2 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert isinstance(merged[0], Delta)
    assert isinstance(merged[0].content, DeltaText)
    assert merged[0].content.data == "Hello world!"
    assert merged[0].content.subtype == "text"

@pytest.mark.asyncio
async def test_dont_merge_different_subtypes():
    """Test that text deltas with different subtypes are not merged."""
    delta1 = Delta(content=DeltaText(data="Hello", subtype="text"))
    delta2 = Delta(content=DeltaText(data="Reasoning", subtype="reasoning"))
    delta3 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Hello"
    assert isinstance(merged[1], Delta)
    assert merged[1].content.data == "Reasoning"
    assert isinstance(merged[2], Delta)
    assert merged[2].content.data == "world!"

@pytest.mark.asyncio
async def test_dont_merge_different_types():
    """Test that deltas of different types are not merged."""
    delta1 = Delta(content=DeltaText(data="Hello", subtype= "text"))
    delta2 = Delta(content=DeltaToolUse(data=ToolUse(id="1", name="some_tool", input={})))
    delta3 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Hello"
    assert isinstance(merged[1], Delta)
    assert isinstance(merged[1].content, DeltaToolUse)
    assert merged[1].content.data.name == "some_tool"
    assert isinstance(merged[2], Delta)
    assert isinstance(merged[2].content, DeltaText)
    assert merged[2].content.data == "world!"

@pytest.mark.asyncio
async def test_prune_empty_text():
    """Test that empty text deltas are pruned."""
    delta1 = Delta(content=DeltaText(data="Hello", subtype="text"))
    delta2 = Delta(content=DeltaText(data="   ", subtype="text"))
    delta3 = Delta(content=DeltaText(data="\n\t", subtype="text"))
    delta4 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, delta2, delta3, delta4]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == """Hello   \n\tworld!"""

@pytest.mark.asyncio
async def test_preserve_non_delta_messages():
    """Test that non-Delta messages in history are preserved."""
    delta1 = Delta(content=DeltaText(data="Hello", subtype="text"))
    non_delta = {"type": "some_other_type", "data": "preserved"}
    delta2 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, non_delta, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Hello"

    assert merged[1] == non_delta

    assert isinstance(merged[2], Delta)
    assert merged[2].content.data == "world!"

@pytest.mark.asyncio
async def test_empty_history():
    """Test that an empty history returns an empty list."""
    history = []
    
    merged = merge_history(history)
    
    assert merged == []

@pytest.mark.asyncio
async def test_single_delta():
    """Test that a history with a single delta returns that delta."""
    delta = Delta(content=DeltaText(data="Hello", subtype="text"))
    history = [delta]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Hello"

@pytest.mark.asyncio
async def test_merge_multiple_consecutive_text_deltas():
    """Test merging multiple consecutive text deltas."""
    delta1 = Delta(content=DeltaText(data="Hello ", subtype="text"))
    delta2 = Delta(content=DeltaText(data="beautiful ", subtype="text"))
    delta3 = Delta(content=DeltaText(data="world!", subtype="text"))
    history = [delta1, delta2, delta3]
    
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert isinstance(merged[0], Delta)
    assert isinstance(merged[0].content, DeltaText)
    assert merged[0].content.data == "Hello beautiful world!"

@pytest.mark.asyncio
async def test_complex_history():
    """Test a complex history with various delta types and subtypes."""
    delta1 = Delta(content=DeltaText(data="Start ", subtype="text"))
    delta2 = Delta(content=DeltaText(data="of text.", subtype="text"))
    delta3 = Delta(content=DeltaToolUse(data=ToolUse(name="some_tool", input={}, id="1")))
    delta4 = Delta(content=DeltaToolResult(data=ToolResult(id="1", content="result")))
    delta5 = Delta(content=DeltaText(data="More ", subtype="reasoning"))
    delta6 = Delta(content=DeltaText(data="reasoning.", subtype="reasoning"))
    delta7 = Delta(content=DeltaText(data="End ", subtype="text"))
    delta8 = Delta(content=DeltaText(data="of text.", subtype="text"))

    history = [delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8]
    
    merged = merge_history(history)
    
    assert len(merged) == 6
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Start of text."
    assert isinstance(merged[1], Delta)
    assert isinstance(merged[1].content, DeltaToolUse)
    assert merged[1].content.data.name == "some_tool"
    assert isinstance(merged[2], Delta)
    assert isinstance(merged[2].content, DeltaToolResult)
    assert merged[2].content.data.content == "result"
    assert isinstance(merged[3], Delta)
    assert merged[3].content.data == "More "
    assert isinstance(merged[4], Delta)
    assert merged[4].content.data == "reasoning."
    assert isinstance(merged[5], Delta)
    assert merged[5].content.data == "End of text."
    
@pytest.mark.asyncio
async def test_bug_empty_history_index_error():
    """Test that the function handles empty history without IndexError."""
    # This test is to ensure the function doesn't try to access merged[-1] when merged is empty
    history = [Delta(content=DeltaText(data="Hello", subtype="text"))]
    
    # This should not raise an IndexError
    merged = merge_history(history)
    
    assert len(merged) == 1
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Hello"

# New tests for query types

@pytest.mark.asyncio
async def test_history_with_file_objects():
    """Test that File objects in history are preserved."""
    delta1 = Delta(content=DeltaText(data="Here's an image:", subtype="text"))
    file_obj = File(b64type="image/png", content="dummy_base64_content")
    delta2 = Delta(content=DeltaText(data="What do you think?", subtype="text"))
    history = [delta1, file_obj, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Here's an image:"
    assert merged[1] == file_obj
    assert isinstance(merged[2], Delta)
    assert merged[2].content.data == "What do you think?"

@pytest.mark.asyncio
async def test_history_with_query_objects():
    """Test that Query objects in history are preserved."""
    delta1 = Delta(content=DeltaText(data="Processing query:", subtype="text"))
    query_obj = Query(["item1", "item2"])
    delta2 = Delta(content=DeltaText(data="Query processed", subtype="text"))
    history = [delta1, query_obj, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 3
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Processing query:"
    assert merged[1] == query_obj
    assert isinstance(merged[2], Delta)
    assert merged[2].content.data == "Query processed"

@pytest.mark.asyncio
async def test_complex_history_with_query_types():
    """Test a complex history with various delta types and query types."""
    delta1 = Delta(content=DeltaText(data="Start ", subtype="text"))
    delta2 = Delta(content=DeltaText(data="of text.", subtype="text"))

    file_obj = File(b64type="image/png", content="dummy_base64_content")

    delta3 = Delta(content=DeltaToolUse(data=ToolUse(name="some_tool", input={}, id="1")))
    delta4 = Delta(content=DeltaToolResult(data=ToolResult(id="1", content="result")))

    query_obj = Query(["query_item"])
    assistant_obj = Assistant("test_assistant")

    delta5 = Delta(content=DeltaText(data="More ", subtype="reasoning"))
    delta6 = Delta(content=DeltaText(data="reasoning.", subtype="reasoning"))
    delta7 = Delta(content=DeltaText(data="End ", subtype="text"))
    delta8 = Delta(content=DeltaText(data="of text.", subtype="text"))

    history = [
        delta1, delta2, file_obj, delta3, delta4, 
        query_obj, assistant_obj, delta5, delta6, delta7, delta8
    ]
    
    merged = merge_history(history)
    
    assert len(merged) == 9
    
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Start of text."

    assert merged[1] == file_obj
    assert isinstance(merged[2], Delta)
    assert isinstance(merged[2].content, DeltaToolUse)
    
    assert isinstance(merged[3], Delta)
    assert isinstance(merged[3].content, DeltaToolResult)
    
    assert merged[4] == query_obj
    assert merged[5] == assistant_obj
    
    assert isinstance(merged[6], Delta)
    assert merged[6].content.data == "More "
    assert isinstance(merged[7], Delta)
    assert merged[7].content.data == "reasoning."
    assert isinstance(merged[8], Delta)
    assert merged[8].content.data == "End of text."

@pytest.mark.asyncio
async def test_history_with_mixed_query_types():
    """Test history with a mix of simple and complex query types."""
    delta1 = Delta(content=DeltaText(data="Text before", subtype="text"))
    
    # String query
    string_query = "Simple string query"
    
    # File query
    file_obj = File(b64type="image/png", content="dummy_base64_content")
    
    assistant_obj = Assistant("test_assistant")
    
    query_list = Query(["item1", file_obj, assistant_obj])
    
    delta2 = Delta(content=DeltaText(data="Text after", subtype="text"))
    
    history = [delta1, string_query, file_obj, assistant_obj, query_list, delta2]
    
    merged = merge_history(history)
    
    assert len(merged) == 6
    assert isinstance(merged[0], Delta)
    assert merged[0].content.data == "Text before"
    assert merged[1] == string_query
    assert merged[2] == file_obj
    assert merged[3] == assistant_obj
    assert merged[4] == query_list
    assert isinstance(merged[5], Delta)
    assert merged[5].content.data == "Text after"

@pytest.mark.asyncio
async def test_invoke_function():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    tools = parse_tools([sample_function])
    func_map = {
        tool.schema["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3, "b": 5}
    result = await invoke_function("sample_function", input_data, func_map)

    assert result == "8", f"Expected '8', got {result}"

@pytest.mark.asyncio
async def test_invoke_function_nonexistent():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    tools = parse_tools([sample_function])
    func_map = {
        tool.schema["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3, "b": 5}
    with pytest.raises(ToolNotFoundError) as exc_info:
        await invoke_function("nonexistent_function", input_data, func_map)
    
    assert "nonexistent_function" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_function_wrong_args():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    tools = parse_tools([sample_function])
    func_map = {
        tool.schema["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3}
    return_val = await invoke_function("sample_function", input_data, func_map)
    assert type(return_val) == str, f"Expected str, got {type(return_val)}"
    result = json.loads(return_val)
    assert "error" in result, f"Expected error, got {result}"
    assert "data must contain ['b']" in result["error"], f"Expected error message not found, got {result['error']}"

@pytest.mark.asyncio
async def test_invoke_function_internal_scope_wrong_args():
    def bad_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)

    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return bad_function(a) # type: ignore # this will raise an error because bad_function expects two arguments
    
    tools = parse_tools([sample_function])
    func_map = {
        tool.schema["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3, "b": 5} # these are the correct arguments, but the function will fail internally

    with pytest.raises(TypeError) as exc_info:
        await invoke_function("sample_function", input_data, func_map)


@pytest.mark.asyncio
async def test_pass_cache_options():
    mock_client = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = iter([
        Delta(content=DeltaText(data="Hello")),
    ])
    llm = Bot(prompt="Test prompt", model=mock_client, cache={
        "cache_system_prompt": True,
        "cache_tools": True
    })
    
    await llm(System("Can be overwritten") + "Test query").string()

    assert mock_client.stream.called
    assert mock_client.stream.call_args[1]['cache'] == {
        "cache_system_prompt": True,
        "cache_tools": True
    }

    # no cache

    llm_no_cache = Bot(prompt="Test prompt", model=mock_client, cache=None)
    await llm_no_cache(System("Can be overwritten") + "Test query").string()
    assert mock_client.stream.call_args[1]['cache'] is None


@pytest.mark.asyncio
async def test_cumulative_usage(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=50, output_tokens=30, cache_read_input_tokens=11, cache_written_input_tokens=3)),
        Delta(content=DeltaText(data="World")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=20, output_tokens=10, cache_read_input_tokens=5, cache_written_input_tokens=2)),
        Delta(content=DeltaText(data="!")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2, cache_written_input_tokens=1)),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    response = llm("Test query")
    result = [msg async for msg in response]
    
    assert response.usage is not None
    assert response.usage.input_tokens == 80
    assert response.usage.output_tokens == 45
    assert response.usage.cache_read_input_tokens == 18
    assert response.usage.cache_written_input_tokens == 6

@pytest.mark.asyncio
async def test_usage_with_history(mock_model):
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=50, output_tokens=30, cache_read_input_tokens=11, cache_written_input_tokens=3)),
        Delta(content=DeltaText(data="World")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=20, output_tokens=10, cache_read_input_tokens=5, cache_written_input_tokens=2)),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    response = llm("Test query")
    result = [msg async for msg in response]
    
    mock_model.set_response([
        Delta(content=DeltaText(data="!")),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2, cache_written_input_tokens=1)),
    ])
    another_response = llm("Another query", previous=response)
    another_result = [msg async for msg in another_response]
    assert another_response.usage is not None
    assert another_response.usage.input_tokens == 10
    assert another_response.usage.output_tokens == 5
    assert another_response.usage.cache_read_input_tokens == 2
    assert another_response.usage.cache_written_input_tokens == 1

@pytest.mark.asyncio
async def test_usage_with_tool_calls(mock_model):

    called = False
    async def test_tool(param1: str, param2: int) -> str:
        """A test tool that returns a string."""
        nonlocal called
        called = True
        return f"Tool called with {param1} and {param2}"

    tools = parse_tools([test_tool])

    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaToolUse(ToolUse(name="test_tool", input={"param1": "asdf", "param2": 10}, id="tool1")), usage=Usage(input_tokens=50, output_tokens=30, cache_read_input_tokens=11, cache_written_input_tokens=3)),
        Delta(content=DeltaText(data=""), usage=Usage(input_tokens=20, output_tokens=10, cache_read_input_tokens=5, cache_written_input_tokens=2)),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model, functions=tools)
    response = llm("Test query")
    result = [msg async for msg in response]

    assert len(result) == 4
    assert isinstance(result[0], Delta)
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Hello"

    assert isinstance(result[1].content.data, ToolUse)
    assert result[1].content.data.name == "test_tool"
    assert result[1].content.data.input == {"param1": "asdf", "param2": 10}

    assert isinstance(result[2].content.data, ToolResult)
    assert result[2].content.data.content.val == "Tool called with asdf and 10"

    assert called, "Tool function was not called"

    assert response.usage is not None

    assert response.usage.input_tokens == 70
    assert response.usage.output_tokens == 40
    assert response.usage.cache_read_input_tokens == 16
    assert response.usage.cache_written_input_tokens == 5

# Transform delta hook tests

@pytest.mark.asyncio
async def test_transform_delta_hook_basic(mock_model):
    """Test that transform_delta_hook is called on each delta and can modify text."""
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaText(data=" world")),
    ])

    hook_called_count = 0

    async def uppercase_hook(delta: Delta) -> Delta:
        """Transform text deltas to uppercase."""
        nonlocal hook_called_count
        hook_called_count += 1
        if isinstance(delta.content, DeltaText):
            return Delta(
                content=DeltaText(data=delta.content.data.upper(), subtype=delta.content.subtype),
                usage=delta.usage
            )
        return delta

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=uppercase_hook)
    result = [msg async for msg in llm("Test query")]

    # Hook should be called for each delta from the model (2 text deltas)
    # Note: History delta is generated after the stream, so not passed through hook
    assert hook_called_count == 2

    # Text should be transformed to uppercase
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "HELLO"
    assert isinstance(result[1].content, DeltaText)
    assert result[1].content.data == " WORLD"

@pytest.mark.asyncio
async def test_transform_delta_hook_with_tool_use(mock_model):
    """Test that transform_delta_hook works with tool use deltas."""
    # Create async generators for mock responses
    async def first_response():
        yield Delta(content=DeltaText(data="Calling tool"))
        yield Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"param": "value"}, id="tool1")))

    async def second_response():
        yield Delta(content=DeltaText(data="Done"))

    # Set up responses for initial call and recursive call after tool execution
    mock_model.stream.side_effect = [first_response(), second_response()]

    hook_called_count = 0
    delta_types_seen = []

    async def tracking_hook(delta: Delta) -> Delta:
        """Track which delta types are passed through the hook."""
        nonlocal hook_called_count
        hook_called_count += 1
        delta_types_seen.append(type(delta.content).__name__)
        return delta

    async def test_tool(param: str) -> str:
        """Test tool."""
        return f"Result: {param}"

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=tracking_hook)
    result = [msg async for msg in llm("Test query", functions=[test_tool])]

    # Hook should be called exactly for:
    # First iteration: DeltaText, DeltaToolUse, DeltaToolResult = 3
    # Second iteration (after tool result): DeltaText = 1
    # Total = 4 calls
    assert hook_called_count == 4
    assert delta_types_seen == ["DeltaText", "DeltaToolUse", "DeltaToolResult", "DeltaText"]

    # Verify result contains the expected deltas
    assert len(result) == 4  # text, tool use, tool result, final text
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Calling tool"
    assert isinstance(result[1].content, DeltaToolUse)
    assert isinstance(result[2].content, DeltaToolResult)
    assert result[2].content.data.content.val == "Result: value"
    assert isinstance(result[3].content, DeltaText)
    assert result[3].content.data == "Done"

@pytest.mark.asyncio
async def test_transform_delta_hook_modify_tool_input(mock_model):
    """Test that transform_delta_hook can modify tool inputs before execution."""
    mock_model.set_response([
        Delta(content=DeltaToolUse(data=ToolUse(name="test_tool", input={"value": 5}, id="tool1"))),
        Delta(content=DeltaText(data="Done")),
    ])

    async def double_value_hook(delta: Delta) -> Delta:
        """Double the value in tool use deltas."""
        if isinstance(delta.content, DeltaToolUse):
            modified_input = {**delta.content.data.input}
            if "value" in modified_input:
                modified_input["value"] = modified_input["value"] * 2
            return Delta(
                content=DeltaToolUse(data=ToolUse(
                    name=delta.content.data.name,
                    input=modified_input,
                    id=delta.content.data.id
                )),
                usage=delta.usage
            )
        return delta

    received_value = None

    async def test_tool(value: int) -> str:
        """Test tool that records received value."""
        nonlocal received_value
        received_value = value
        return f"Got {value}"

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=double_value_hook)
    result = [msg async for msg in llm("Test query", functions=[test_tool])]

    # The tool should receive the doubled value (10 instead of 5)
    assert received_value == 10
    assert any(
        isinstance(msg.content, DeltaToolResult) and
        msg.content.data.content.val == "Got 10"
        for msg in result
    )

@pytest.mark.asyncio
async def test_transform_delta_hook_with_usage(mock_model):
    """Test that transform_delta_hook preserves usage information."""
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello"), usage=Usage(input_tokens=50, output_tokens=30, cache_read_input_tokens=10, cache_written_input_tokens=5)),
    ])

    async def prefix_hook(delta: Delta) -> Delta:
        """Add prefix to text deltas while preserving usage."""
        if isinstance(delta.content, DeltaText):
            return Delta(
                content=DeltaText(data=f"[PREFIX] {delta.content.data}", subtype=delta.content.subtype),
                usage=delta.usage
            )
        return delta

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=prefix_hook)
    result = [msg async for msg in llm("Test query")]

    # Text should be prefixed
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "[PREFIX] Hello"

    # Usage should be preserved
    assert result[0].usage is not None
    assert result[0].usage.input_tokens == 50
    assert result[0].usage.output_tokens == 30

@pytest.mark.asyncio
async def test_transform_delta_hook_none(mock_model):
    """Test that None hook (no transformation) works correctly."""
    mock_model.set_response([
        Delta(content=DeltaText(data="Hello")),
        Delta(content=DeltaText(data=" world")),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=None)
    result = [msg async for msg in llm("Test query")]

    # Text should be unmodified
    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Hello"
    assert isinstance(result[1].content, DeltaText)
    assert result[1].content.data == " world"

@pytest.mark.asyncio
async def test_transform_delta_hook_filter_deltas(mock_model):
    """Test using transform_delta_hook to filter certain deltas."""
    mock_model.set_response([
        Delta(content=DeltaText(data="Keep this", subtype="text")),
        Delta(content=DeltaText(data="Filter this out", subtype="reasoning")),
        Delta(content=DeltaText(data="Keep this too", subtype="text")),
    ])

    async def filter_reasoning_hook(delta: Delta) -> Delta:
        """Filter out reasoning deltas by returning empty text."""
        if isinstance(delta.content, DeltaText) and delta.content.subtype == "reasoning":
            return Delta(content=DeltaText(data="", subtype=delta.content.subtype), usage=delta.usage)
        return delta

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=filter_reasoning_hook)
    result = [msg async for msg in llm("Test query")]

    # Should have text deltas and history delta
    text_deltas = [msg for msg in result if isinstance(msg.content, DeltaText)]
    assert len(text_deltas) == 3
    assert text_deltas[0].content.data == "Keep this"
    assert text_deltas[1].content.data == ""  # filtered
    assert text_deltas[2].content.data == "Keep this too"

@pytest.mark.asyncio
async def test_transform_delta_hook_in_query_kwargs(mock_model):
    """Test that transform_delta_hook can be passed as query kwarg."""
    mock_model.set_response([
        Delta(content=DeltaText(data="test")),
    ])

    hook_called = False

    async def test_hook(delta: Delta) -> Delta:
        nonlocal hook_called
        hook_called = True
        return delta

    llm = Bot(prompt="Test prompt", model=mock_model)
    # Pass hook in query call instead of Bot constructor
    result = [msg async for msg in llm("Test query", transform_delta_hook=test_hook)]

    assert hook_called

@pytest.mark.asyncio
async def test_transform_delta_hook_override(mock_model):
    """Test that query-level hook overrides bot-level hook."""
    mock_model.set_response([
        Delta(content=DeltaText(data="hello")),
    ])

    bot_hook_called = False
    query_hook_called = False

    async def bot_hook(delta: Delta) -> Delta:
        nonlocal bot_hook_called
        bot_hook_called = True
        return delta

    async def query_hook(delta: Delta) -> Delta:
        nonlocal query_hook_called
        query_hook_called = True
        if isinstance(delta.content, DeltaText):
            return Delta(content=DeltaText(data=delta.content.data.upper(), subtype=delta.content.subtype), usage=delta.usage)
        return delta

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=bot_hook)
    # Override with query-level hook
    result = [msg async for msg in llm("Test query", transform_delta_hook=query_hook)]

    # Only query hook should be called
    assert not bot_hook_called
    assert query_hook_called

    # Text should be uppercase (from query hook)
    text_deltas = [msg for msg in result if isinstance(msg.content, DeltaText)]
    assert text_deltas[0].content.data == "HELLO"

@pytest.mark.asyncio
async def test_transform_delta_hook_with_multiple_tool_calls(mock_model):
    """Test that transform_delta_hook works correctly with multiple tool calls."""

    # Create async generators for mock responses
    async def first_response():
        yield Delta(content=DeltaToolUse(data=ToolUse(name="tool1", input={"x": 1}, id="t1")))
        yield Delta(content=DeltaText(data="After first tool"))

    async def second_response():
        yield Delta(content=DeltaText(data="Final response"))

    # Set up responses for initial call and recursive call after tool execution
    mock_model.stream.side_effect = [first_response(), second_response()]

    hook_call_count = 0
    deltas_seen = []

    async def counting_hook(delta: Delta) -> Delta:
        nonlocal hook_call_count
        hook_call_count += 1
        deltas_seen.append((type(delta.content).__name__,
                           delta.content.data if hasattr(delta.content, 'data') else None))
        return delta

    async def tool1(x: int) -> str:
        """Test tool."""
        return f"Result {x}"

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=counting_hook)
    result = [msg async for msg in llm("Test query", functions=[tool1])]

    # Hook should be called for all deltas from the model stream
    # Order of execution:
    # 1. First stream yields DeltaToolUse
    # 2. Tool is invoked, DeltaToolResult created and passed through hook
    # 3. Recursive query starts (second stream)
    # 4. Second stream yields DeltaText("Final response")
    # 5. Recursive query completes, control returns to first stream
    # 6. First stream continues and yields DeltaText("After first tool")
    # Total = 4 calls to the hook
    assert hook_call_count == 4
    assert len(deltas_seen) == 4
    assert deltas_seen[0][0] == "DeltaToolUse"
    assert deltas_seen[1][0] == "DeltaToolResult"
    assert deltas_seen[2][0] == "DeltaText"
    assert deltas_seen[2][1] == "Final response"  # From recursive call
    assert deltas_seen[3][0] == "DeltaText"
    assert deltas_seen[3][1] == "After first tool"  # Continuing first stream

    # Verify the result contains all expected deltas
    # Result order matches what user sees: tool use, tool result, then text responses
    assert len(result) == 4
    assert isinstance(result[0].content, DeltaToolUse)
    assert result[0].content.data.name == "tool1"
    assert result[0].content.data.input == {"x": 1}
    assert isinstance(result[1].content, DeltaToolResult)
    assert result[1].content.data.content.val == "Result 1"
    assert isinstance(result[2].content, DeltaText)
    assert result[2].content.data == "Final response"
    assert isinstance(result[3].content, DeltaText)
    assert result[3].content.data == "After first tool"

@pytest.mark.asyncio
async def test_transform_delta_hook_with_sequential_tool_calls(mock_model):
    """Test hook with pattern: text -> tool1 -> text -> tool2 -> text."""

    # First stream: yields text then first tool call
    async def first_response():
        yield Delta(content=DeltaText(data="Starting"))
        yield Delta(content=DeltaToolUse(data=ToolUse(name="tool1", input={"x": 1}, id="t1")))

    # Second stream (after tool1): yields text then second tool call
    async def second_response():
        yield Delta(content=DeltaText(data="Between tools"))
        yield Delta(content=DeltaToolUse(data=ToolUse(name="tool2", input={"y": 2}, id="t2")))

    # Third stream (after tool2): yields final text
    async def third_response():
        yield Delta(content=DeltaText(data="Done"))

    mock_model.stream.side_effect = [first_response(), second_response(), third_response()]

    hook_call_count = 0
    deltas_seen = []

    async def tracking_hook(delta: Delta) -> Delta:
        nonlocal hook_call_count
        hook_call_count += 1
        if isinstance(delta.content, DeltaText):
            deltas_seen.append(("text", delta.content.data))
        elif isinstance(delta.content, DeltaToolUse):
            deltas_seen.append(("tool", delta.content.data.name))
        return delta

    async def tool1(x: int) -> str:
        """First test tool."""
        return f"Result1: {x}"

    async def tool2(y: int) -> str:
        """Second test tool."""
        return f"Result2: {y}"

    llm = Bot(prompt="Test prompt", model=mock_model, transform_delta_hook=tracking_hook)
    result = [msg async for msg in llm("Test query", functions=[tool1, tool2])]

    # Hook execution order due to nested recursive queries:
    # 1. First stream: "Starting" (text)
    # 2. First stream: tool1 (tool use)
    # 3. Tool1 invoked, DeltaToolResult created
    # 4. Recursive query (second stream) starts
    # 5. Second stream: "Between tools" (text)
    # 6. Second stream: tool2 (tool use)
    # 7. Tool2 invoked, DeltaToolResult created
    # 8. Recursive query (third stream) starts
    # 9. Third stream: "Done" (text)
    # Total: 7 hook calls (3 text + 2 tool use + 2 tool result)
    assert hook_call_count == 7
    # Note: The hook sees DeltaToolResult but tracking_hook only tracks text and tool types
    # so deltas_seen will only have those
    assert len(deltas_seen) == 5
    assert deltas_seen == [
        ("text", "Starting"),
        ("tool", "tool1"),
        ("text", "Between tools"),
        ("tool", "tool2"),
        ("text", "Done")
    ]

    # Result contains: text, tool1 use, tool1 result, text, tool2 use, tool2 result, text
    assert len(result) == 7

    assert isinstance(result[0].content, DeltaText)
    assert result[0].content.data == "Starting"

    assert isinstance(result[1].content, DeltaToolUse)
    assert result[1].content.data.name == "tool1"

    assert isinstance(result[2].content, DeltaToolResult)
    assert result[2].content.data.content.val == "Result1: 1"

    assert isinstance(result[3].content, DeltaText)
    assert result[3].content.data == "Between tools"

    assert isinstance(result[4].content, DeltaToolUse)
    assert result[4].content.data.name == "tool2"

    assert isinstance(result[5].content, DeltaToolResult)
    assert result[5].content.data.content.val == "Result2: 2"

    assert isinstance(result[6].content, DeltaText)
    assert result[6].content.data == "Done"