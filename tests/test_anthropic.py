from dataclasses import dataclass
from typing import Annotated, List, Optional
import pytest
from unittest.mock import Mock
from contextlib import asynccontextmanager

from mus.llm.anthropic import func_to_tool, functions_for_llm
from mus import AnthropicLLM
from mus.functions import to_schema
from anthropic import types as at

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



@asynccontextmanager
async def return_val():
    """A context manager that yields a value."""
    async def iterator():
        yield at.Message(
            role="assistant",
            content=[
                at.TextBlock(type="text", text="This is a test response."),
            ],
            id="test-id",
            model="a-model-id",
            type="message",
            usage=at.Usage(
                input_tokens=10,
                output_tokens=5,
            ),
        )
    yield iterator()


@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def bedrock_llm(mock_client):
    return AnthropicLLM("a-model-id", mock_client)


@pytest.mark.asyncio
async def test_anthropic_stream(mock_client):
    async def dummy_tool(param1: str, param2: int) -> str:
        """A dummy tool for testing."""
        return f"Dummy result with {param1} and {param2}"
    expected_tool = to_schema(dummy_tool)
    mock_client.messages.stream = Mock(return_value=return_val())
    llm = AnthropicLLM("a-model-id", mock_client)
    async for response in llm.stream(
        prompt="Test prompt",
        history=[],
        functions=[expected_tool],
        no_stream=False,
        function_choice="auto",
        kwargs={},
        top_k=5,
        top_p=0.9,
        stop_sequences=["\n", "END"],
        temperature=0.7,
        ):
        assert response is None
    
    mock_client.messages.stream.assert_called_once_with(
        max_tokens=4096,
        model="a-model-id",
        messages=[],
        top_k=5,
        top_p=0.9,
        stop_sequences=["\n", "END"],
        temperature=0.7,
        tools=[{
            "name": expected_tool["name"],
            "description": expected_tool["description"],
            "input_schema": expected_tool["schema"]
        }],
        tool_choice={"type": "auto"},
        system="Test prompt",
    )


def test_func_to_tool():
    tool = func_to_tool(to_schema(sample_function))
    assert is_tool_param(tool)
    assert tool["name"] == "sample_function"
    assert tool["description"] == "This is a sample function."
    assert "param1" in tool["input_schema"]["properties"]
    assert "param2" in tool["input_schema"]["properties"]

def test_functions_for_llm():
    functions = [to_schema(sample_function), to_schema(SampleDataclass)]
    tools = functions_for_llm(functions)
    assert len(tools) == 2
    assert all(is_tool_param(tool) for tool in tools)
    assert tools[0]["name"] == "sample_function"
    assert tools[1]["name"] == "SampleDataclass"

def test_functions_for_llm_empty_input():
    tools = functions_for_llm(None)
    assert len(tools) == 0