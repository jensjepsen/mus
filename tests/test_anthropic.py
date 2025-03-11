from dataclasses import dataclass
from typing import Annotated, List, Optional

from mus.llm.anthropic import func_to_tool, dataclass_to_tool, functions_for_llm


def sample_function(param1: str, param2: int) -> str:
    """This is a sample function."""
    return f"{param1}: {param2}"

@dataclass
class SampleDataclass:
    """This is a sample dataclass."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"] = None


# Helper function to check if a dict matches ToolParam structure
def is_tool_param(obj):
    return isinstance(obj, dict) and all(key in obj for key in ['name', 'description', 'input_schema'])

def test_func_to_tool():
    tool = func_to_tool(sample_function)
    assert is_tool_param(tool)
    assert tool["name"] == "sample_function"
    assert tool["description"] == "This is a sample function."
    assert "param1" in tool["input_schema"]["properties"]
    assert "param2" in tool["input_schema"]["properties"]

def test_dataclass_to_tool():
    tool = dataclass_to_tool(SampleDataclass)
    assert is_tool_param(tool)
    assert tool["name"] == "SampleDataclass"
    assert tool["description"] == "This is a sample dataclass."
    assert "field1" in tool["input_schema"]["properties"]
    assert "field2" in tool["input_schema"]["properties"]
    assert "field3" in tool["input_schema"]["properties"]
    assert tool["input_schema"]["properties"]["field3"]["description"] == "An optional field"

def test_functions_for_llm():
    functions = [sample_function, SampleDataclass]
    tools = functions_for_llm(functions)
    assert len(tools) == 2
    assert all(is_tool_param(tool) for tool in tools)
    assert tools[0]["name"] == "sample_function"
    assert tools[1]["name"] == "SampleDataclass"

def test_functions_for_llm_empty_input():
    tools = functions_for_llm(None)
    assert len(tools) == 0