from dataclasses import dataclass
from typing import List, Optional, Annotated, TypedDict
import pytest

# Import the functions to be tested
from mus.functions import get_schema, functions_map, to_schema, FunctionSchema, schema_to_example

# Test data
def sample_function(param1: str, param2: int) -> str:
    """This is a sample function."""
    return f"{param1}: {param2}"

class SampleTypedDict(TypedDict):
    """This is a sample TypedDict."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"]

@dataclass
class SampleDataclass:
    """This is a sample dataclass."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"] = None

@dataclass
class SampleNestedDataclass:
    """This is a sample nested dataclass."""
    field1: str
    field2: int
    field3: SampleDataclass

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

def test_to_schema_function():
    schema = to_schema(sample_function)
    assert isinstance(schema, dict)
    assert schema["name"] == "sample_function"
    assert schema["description"] == "This is a sample function."
    json_schema = schema["schema"]
    assert json_schema["title"] == "sample_function"
    assert "param1" in json_schema["properties"]
    assert "param2" in json_schema["properties"]
    assert json_schema["properties"]["param1"]["type"] == "string"
    assert json_schema["properties"]["param2"]["type"] == "integer"

def test_to_schema_typed_dict():
    schema = to_schema(SampleTypedDict)
    assert isinstance(schema, dict)
    assert schema["name"] == "SampleTypedDict"
    assert schema["description"] == "This is a sample TypedDict."
    json_schema = schema["schema"]
    assert json_schema["title"] == "SampleTypedDict"
    assert "field1" in json_schema["properties"]
    assert "field2" in json_schema["properties"]
    assert "field3" in json_schema["properties"]
    assert json_schema["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field2"]["type"] == "integer"

def test_to_schema_dataclass():
    schema = to_schema(SampleDataclass)
    assert isinstance(schema, dict)
    assert schema["name"] == "SampleDataclass"
    assert schema["description"] == "This is a sample dataclass."
    json_schema = schema["schema"]
    assert json_schema["title"] == "SampleDataclass"
    assert "field1" in json_schema["properties"]
    assert "field2" in json_schema["properties"]
    assert "field3" in json_schema["properties"]
    assert json_schema["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field2"]["type"] == "integer"

def test_to_schema_nested_dataclass():
    schema = to_schema(SampleNestedDataclass)
    assert isinstance(schema, dict)
    assert schema["name"] == "SampleNestedDataclass"
    assert schema["description"] == "This is a sample nested dataclass."
    json_schema = schema["schema"]
    assert json_schema["title"] == "SampleNestedDataclass"
    assert "field1" in json_schema["properties"]
    assert "field2" in json_schema["properties"]
    assert "field3" in json_schema["properties"]
    assert json_schema["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field2"]["type"] == "integer"
    assert json_schema["properties"]["field3"]["type"] == "object"
    assert json_schema["properties"]["field3"]["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field3"]["properties"]["field2"]["type"] == "integer"
    
    

def test_to_schema_invalid_type():
    with pytest.raises(ValueError, match="Unsupported type: <class 'int'>"):
        to_schema(123)    

def test_to_schema_no_docstring():
    def no_docstring_function(param1: str, param2: int) -> str:
        return f"{param1}: {param2}"
    with pytest.raises(ValueError, match="Function no_docstring_function is missing a docstring"):
        to_schema(no_docstring_function)

    class NoDocstringTypedDict(TypedDict):
        field1: str
        field2: int

    with pytest.raises(ValueError, match="TypedDict NoDocstringTypedDict is missing a docstring"):
        to_schema(NoDocstringTypedDict)

def test_schema_to_example():
    schema = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"},
            "field3": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    example = schema_to_example(schema)
    assert example == {
        "field1": "<string>",
        "field2": "<integer>",
        "field3": ["<string>"]
    }

def test_schema_to_example_invalid():
    schema = {
        "a-random-key": 10

    }
    with pytest.raises(ValueError, match="Couldn't parse schema:"):
        schema_to_example(schema)

def test_schema_to_exmaple_enum():
    schema = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {
                "type": "string",
                "enum": ["value1", "value2", "value3"]
            }
        }
    }
    example = schema_to_example(schema)
    assert example == {
        "field1": "<string>",
        "field2": "<value1|value2|value3>"
    }

def test_schema_to_example_function_schema():
    class NestedTypedDict(TypedDict):
        field1: str
        field2: int

    class TypedDictExample(TypedDict):
        """This is a sample TypedDict."""
        field1: str
        field2: int
        field3: NestedTypedDict

    schema = to_schema(TypedDictExample)

    example = schema_to_example(schema)

    assert example == {
       "field1": "<string>",
       "field2": "<integer>",
        "field3": {
           "field1": "<string>",
           "field2": "<integer>"
        }
    }

    