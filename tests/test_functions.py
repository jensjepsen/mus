from dataclasses import dataclass
from typing import List, Optional, Annotated

# Import the functions to be tested
from mus.functions import get_schema, functions_map

# Test data
def sample_function(param1: str, param2: int) -> str:
    """This is a sample function."""
    return f"{param1}: {param2}"

@dataclass
class SampleDataclass:
    """This is a sample dataclass."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"] = None

# Tests
def test_get_schema():
    schema = get_schema("TestModel", [("param1", str), ("param2", int)])
    assert schema["title"] == "TestModel"
    assert "param1" in schema["properties"]
    assert "param2" in schema["properties"]
    assert schema["properties"]["param1"]["type"] == "string"
    assert schema["properties"]["param2"]["type"] == "integer"

def test_functions_map():
    functions = [sample_function, SampleDataclass]
    func_map = functions_map(functions)
    assert len(func_map) == 2
    assert "sample_function" in func_map
    assert "SampleDataclass" in func_map
    assert func_map["sample_function"] == sample_function
    assert func_map["SampleDataclass"] == SampleDataclass



def test_functions_map_empty_input():
    func_map = functions_map(None)
    assert len(func_map) == 0

def test_get_schema_with_metadata():
    schema = get_schema("TestModel", [("param1", Annotated[str, "This is a string with metadata"]), ("param2", int)])
    assert schema["properties"]["param1"]["description"] == "This is a string with metadata"
    assert "description" not in schema["properties"]["param2"]