import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass
import json

from mus import Bot
from mus.llm.llm import IterableResult, merge_history, invoke_function
from mus.functions import parse_tools
from mus.llm.types import Delta, ToolUse, ToolResult, System, Query, Assistant, File

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
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 11, "cache_written_input_tokens": 3}),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    
    result = [msg async for msg in llm("Test query")]
    assert len(result) == 2
    assert isinstance(result[0], Delta)
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Hello"
    assert isinstance(result[0], Delta)
    assert result[1].usage is not None
    assert result[1].usage["input_tokens"] == 50
    assert result[1].usage["output_tokens"] == 30
    
@pytest.mark.asyncio
async def test_llm_with_tool_use(mock_model):
    mock_model.set_response([
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"param1": "test", "param2": 123}, id="test_tool")}),
        Delta(content={"type": "text", "data": "Tool used"}),
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
    assert result[0].content["type"] == "text"
    assert result[1].content["type"] == "tool_use"
    assert isinstance(result[1].content["data"], ToolUse)
    
    assert result[2].content["type"] == "tool_result"
    assert isinstance(result[2].content["data"], ToolResult)
    assert result[2].content["data"].content == "Tool result"

    assert result[3].content["type"] == "text"
    assert result[3].content["data"] == "Tool used"
    
    
    
    
@pytest.mark.asyncio
async def test_llm_call(mock_model):
    mock_model.set_response([Delta(content={"data": "Test response", "type": "text"})])
    llm = Bot(prompt="Test prompt", model=mock_model)

    result = llm("Test query")

    assert isinstance(result, IterableResult)
    assert (await result.string()) == "Test response"

@pytest.mark.parametrize("strategy", ["tool_use", "prefill"])
@pytest.mark.asyncio
async def test_llm_fill(strategy, mock_model):
    if strategy == "tool_use":
        mock_model.set_response([
            Delta(content={"data": "Processing", "type": "text"}),
            Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"field1": "test", "field2": 123}, id="abc")})
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content={"type": "text", "data": """
"test",
"field2": 123
}
"""}),
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
            Delta(content={"type": "text", "data": "Processing"}),
            Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"field1": "test"}, id="abc")}),
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content={"type": "text", "data": """
"test"
}
```
"""
                           }),
        ])
    
    llm = Bot(prompt="Test prompt", model=mock_model)
    
    with pytest.raises(ValueError) as exc_info:
        await llm.fill("Test query", TestStructure, strategy=strategy)
    
    assert "missing @ $.field2" in str(exc_info.value)

@pytest.mark.parametrize("strategy", ["tool_use", "prefill"])
@pytest.mark.asyncio
async def test_llm_fill_wrong_argument_type(strategy, mock_model):
    if strategy == "tool_use":
        mock_model.set_response([
            Delta(content={"type": "text", "data": "Processing"}),
            Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"field1": 123, "field2": "not_an_int"}, id="abc")}),
        ])
    elif strategy == "prefill":
        mock_model.set_response([
            Delta(content={"type": "text", "data": """
"test",
"field2": "not_an_int"
}
"""}),
        ])
        
    llm = Bot(prompt="Test prompt", model=mock_model)
    
    with pytest.raises(ValueError) as exc_info:
        await llm.fill("Test query", TestStructure, strategy=strategy)
    
    assert "expected int @ $.field2" in str(exc_info.value)

@pytest.mark.asyncio
async def test_llm_bot_decorator(mock_model):
    mock_model.set_response([
        Delta(content={"data": "Test response", "type": "text"})
    ])
    llm = Bot(prompt="Test prompt", model=mock_model)
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
async def test_dynamic_system_prompt(mock_model):
    mock_model.set_response([
        Delta(content={"type": "text", "data": "Hello"}),
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
        Delta(content={"type": "text", "data": "Hello"}),
    ])
    llm = Bot(prompt="Test prompt", model=mock_model)
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
    llm = Bot(prompt="Test prompt", model=mock_model)
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

@pytest.mark.asyncio
async def test_invoke_function():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    tools = parse_tools([sample_function])
    func_map = {
        tool["schema"]["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3, "b": 5}
    result = await invoke_function("sample_function", input_data, func_map)

    assert result == "8", f"Expected '8', got {result}"

@pytest.mark.asyncio
async def test_invoke_function_wrong_args():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    tools = parse_tools([sample_function])
    func_map = {
        tool["schema"]["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3}
    return_val = await invoke_function("sample_function", input_data, func_map)
    assert type(return_val) == str, f"Expected str, got {type(return_val)}"
    result = json.loads(return_val)
    assert "error" in result, f"Expected error, got {result}"
    assert "field missing @ $.b" in result["error"], f"Expected error message not found, got {result['error']}"

@pytest.mark.asyncio
async def test_invoke_function_coerce_args():
    async def sample_function(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)

    tools = parse_tools([sample_function])
    func_map = {
        tool["schema"]["name"]: tool
        for tool in tools
    }

    input_data = {"a": "3", "b": "5"}
    result = await invoke_function("sample_function", input_data, func_map)
    assert result == "8", f"Expected '8', got {result}"

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
        tool["schema"]["name"]: tool
        for tool in tools
    }

    input_data = {"a": 3, "b": 5} # these are the correct arguments, but the function will fail internally

    with pytest.raises(TypeError) as exc_info:
        await invoke_function("sample_function", input_data, func_map)

@pytest.mark.asyncio
async def test_pass_cache_options():
    mock_client = MagicMock()
    mock_client.stream.return_value.__aenter__.return_value = iter([
        Delta(content={"type": "text", "data": "Hello"}),
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
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 11, "cache_written_input_tokens": 3}),
        Delta(content={"type": "text", "data": "World"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 5, "cache_written_input_tokens": 2}),
        Delta(content={"type": "text", "data": "!"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2, "cache_written_input_tokens": 1}),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    response = llm("Test query")
    result = [msg async for msg in response]
    
    assert response.usage is not None
    assert response.usage["input_tokens"] == 80
    assert response.usage["output_tokens"] == 45
    assert response.usage["cache_read_input_tokens"] == 18
    assert response.usage["cache_written_input_tokens"] == 6

@pytest.mark.asyncio
async def test_usage_with_history(mock_model):
    mock_model.set_response([
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 11, "cache_written_input_tokens": 3}),
        Delta(content={"type": "text", "data": "World"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 5, "cache_written_input_tokens": 2}),
    ])

    llm = Bot(prompt="Test prompt", model=mock_model)
    response = llm("Test query")
    result = [msg async for msg in response]
    
    mock_model.set_response([
        Delta(content={"type": "text", "data": "!"}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2, "cache_written_input_tokens": 1}),
    ])
    another_response = llm("Another query", previous=response)
    another_result = [msg async for msg in another_response]
    assert another_response.usage is not None
    assert another_response.usage["input_tokens"] == 10
    assert another_response.usage["output_tokens"] == 5
    assert another_response.usage["cache_read_input_tokens"] == 2
    assert another_response.usage["cache_written_input_tokens"] == 1
    
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
        Delta(content={"type": "text", "data": "Hello"}),
        Delta(content={"type": "tool_use", "data": ToolUse(name="test_tool", input={"param1": "asdf", "param2": 10}, id="tool1")}, usage={"input_tokens": 50, "output_tokens": 30, "cache_read_input_tokens": 11, "cache_written_input_tokens": 3}),
        Delta(content={"type": "text", "data": ""}, usage={"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 5, "cache_written_input_tokens": 2}),
    ])
    
    llm = Bot(prompt="Test prompt", model=mock_model, functions=tools)
    response = llm("Test query")
    result = [msg async for msg in response]

    assert len(result) == 4
    assert result[0].content["type"] == "text"
    assert result[0].content["data"] == "Hello"
    assert result[1].content["type"] == "tool_use"
    assert isinstance(result[1].content["data"], ToolUse)
    assert result[1].content["data"].name == "test_tool"
    assert result[1].content["data"].input == {"param1": "asdf", "param2": 10}
    assert result[2].content["type"] == "tool_result"
    assert isinstance(result[2].content["data"], ToolResult)
    assert result[2].content["data"].content == "Tool called with asdf and 10"

    assert called, "Tool function was not called"
    
    assert response.usage is not None
    assert response.usage["input_tokens"] == 70
    assert response.usage["output_tokens"] == 40
    assert response.usage["cache_read_input_tokens"] == 16
    assert response.usage["cache_written_input_tokens"] == 5