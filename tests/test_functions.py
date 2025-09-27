from dataclasses import dataclass
from typing import List, Optional, Annotated, TypedDict, Literal
import pytest

# Import the functions to be tested
from mus.functions import get_schema, to_schema, FunctionSchema, schema_to_example, parse_tools, ToolCallable, verify_schema_inputs

# Test data
def sample_function(param1: str, param2: int) -> str:
    """This is a sample function."""
    return f"{param1}: {param2}"


class SampleTypedDict(TypedDict):
    """This is a sample TypedDict."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"]

def sample_function_advanced(param1: str, param2: SampleTypedDict, param3: Optional[List[str]] = None, param4: Optional[Literal["value1", "value2"]] = None) -> str:
    """This is a sample function with advanced parameters."""
    return f"{param1}: {param2}, {param3}, {param4}"

@dataclass
class SampleDataclass:
    """This is a sample dataclass."""
    field1: str
    field2: int
    field3: Annotated[Optional[List[str]], "An optional field"] = None


class NestedTypedDict(TypedDict):
    """This is a nested TypedDict."""
    field3: SampleTypedDict
    field4: SampleDataclass



# Tests
def test_get_schema():
    schema = get_schema("TestModel", [("param1", str), ("param2", int)])
    assert schema["title"] == "TestModel"
    assert "param1" in schema["properties"]
    assert "param2" in schema["properties"]
    assert schema["properties"]["param1"]["type"] == "string"
    assert schema["properties"]["param2"]["type"] == "integer"

def test_get_schema_with_metadata():
    schema = get_schema("TestModel", [("param1", Annotated[str, "This is a string with metadata"]), ("param2", int)])
    assert schema["properties"]["param1"]["description"] == "This is a string with metadata"
    assert "description" not in schema["properties"]["param2"]

def test_get_schema_with_optional():
    schema = get_schema("TestModel", [("param1", Optional[str]), ("param2", int)])
    assert "param1" not in schema.get("required", [])
    assert "param2" in schema.get("required", [])

def test_get_schema_with_literal():
    schema = get_schema("TestModel", [("param1", Literal["value1", "value2"]), ("param2", Literal[1,2,3]), ("param3", Literal[1, 2.0])])
    assert "param1" in schema["properties"]
    assert "param2" in schema["properties"]
    assert schema["properties"]["param1"]["type"] == "string"
    assert schema["properties"]["param1"]["enum"] == ["value1", "value2"]

    assert schema["properties"]["param2"]["type"] == "integer"
    assert schema["properties"]["param2"]["enum"] == [1, 2, 3]
    
    assert schema["properties"]["param3"]["type"] == "number"
    assert schema["properties"]["param3"]["enum"] == [1, 2.0]
    



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

def test_to_schema_function_advanced():
    schema = to_schema(sample_function_advanced)
    assert isinstance(schema, dict)
    assert schema["name"] == "sample_function_advanced"
    assert schema["description"] == "This is a sample function with advanced parameters."
    json_schema = schema["schema"]
    assert json_schema["title"] == "sample_function_advanced"
    assert "param1" in json_schema["properties"]
    assert "param2" in json_schema["properties"]
    assert "param3" in json_schema["properties"]
    assert json_schema["properties"]["param1"]["type"] == "string"
    assert json_schema["properties"]["param2"]["type"] == "object"
    assert json_schema["properties"]["param3"]["type"] == "array"
    # test required properties
    assert json_schema["required"] == ["param1", "param2"]

    assert json_schema["properties"]["param2"]["title"] == "SampleTypedDict"
    assert "field1" in json_schema["properties"]["param2"]["properties"]
    assert "field2" in json_schema["properties"]["param2"]["properties"]
    assert json_schema["properties"]["param2"]["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["param2"]["properties"]["field2"]["type"] == "integer"

    assert json_schema["properties"]["param3"]["type"] == "array"
    assert json_schema["properties"]["param3"]["items"]["type"] == "string"

    assert json_schema["properties"]["param4"]["type"] == "string"
    assert json_schema["properties"]["param4"]["enum"] == ["value1", "value2"]


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

def test_to_schema_typed_dict_nested():
    schema = to_schema(NestedTypedDict)
    assert isinstance(schema, dict)
    assert schema["name"] == "NestedTypedDict"
    assert schema["description"] == "This is a nested TypedDict."
    json_schema = schema["schema"]
    assert json_schema["title"] == "NestedTypedDict"
    assert "field3" in json_schema["properties"]
    assert "field4" in json_schema["properties"]
    assert json_schema["properties"]["field3"] == {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"},
            "field3": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["field1", "field2"],
        "title": "SampleTypedDict"
    }
    assert json_schema["properties"]["field4"] == {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"},
            "field3": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["field1", "field2"],
        "title": "SampleDataclass"
    }

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

    
def test_parse_tools():
    async def func1(param1: str, param2: int) -> str:
        """This is func1."""
        return f"{param1}: {param2}"
    async def func2(param1: str) -> str:
        """This is func2."""
        return f"Hello {param1}"
    
    prespecified_tool = ToolCallable(
        function=func2,
        schema=to_schema(func2)
    )

    tools = parse_tools([func1, prespecified_tool])
    assert len(tools) == 2

    assert tools[0].function == func1
    assert tools[0].schema["name"] == "func1"
    assert tools[0].schema["description"] == "This is func1."
    assert tools[0].schema["schema"]["title"] == "func1"
    assert "param1" in tools[0].schema["schema"]["properties"]
    assert "param2" in tools[0].schema["schema"]["properties"]
    assert tools[0].schema["schema"]["properties"]["param1"]["type"] == "string"
    assert tools[0].schema["schema"]["properties"]["param2"]["type"] == "integer"

    assert tools[1].function == func2
    assert tools[1].schema["name"] == "func2"
    assert tools[1].schema["description"] == "This is func2."
    assert tools[1].schema["schema"]["title"] == "func2"
    assert "param1" in tools[1].schema["schema"]["properties"]
    assert tools[1].schema["schema"]["properties"]["param1"]["type"] == "string"

def test_parse_tools_with_string_with_metadata():
    from mus import StringWithMetadata

    async def func1(param1: str, param2: int) -> StringWithMetadata:
        """This is func1."""
        return StringWithMetadata(f"{param1}: {param2}")

    tools = parse_tools([func1])
    assert len(tools) == 1
    assert tools[0].function == func1
    assert tools[0].schema["name"] == "func1"
    assert tools[0].schema["description"] == "This is func1."
    assert tools[0].schema["schema"]["title"] == "func1"
    assert "param1" in tools[0].schema["schema"]["properties"]
    assert "param2" in tools[0].schema["schema"]["properties"]
    assert tools[0].schema["schema"]["properties"]["param1"]["type"] == "string"
    assert tools[0].schema["schema"]["properties"]["param2"]["type"] == "integer"

def test_pydantic_type_to_schema():
    from pydantic import BaseModel, Field

    class NestedModel(BaseModel):
        field1: str
        field2: int

    class PydanticModel(BaseModel):
        """This is a sample Pydantic model."""
        field1: str
        field2: int
        field3: Optional[List[str]] = None
        field4: NestedModel

    schema = to_schema(PydanticModel)
    assert isinstance(schema, dict)
    assert schema["name"] == "PydanticModel"
    assert schema["description"] == "This is a sample Pydantic model."
    json_schema = schema["schema"]
    assert json_schema["title"] == "PydanticModel"
    assert "field1" in json_schema["properties"]
    assert "field2" in json_schema["properties"]
    assert "field3" in json_schema["properties"]
    assert "field4" in json_schema["properties"]
    assert json_schema["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field2"]["type"] == "integer"
    assert json_schema["properties"]["field3"]["type"] == "array"
    assert json_schema["properties"]["field3"]["items"]["type"] == "string"
    assert json_schema["properties"]["field4"]["type"] == "object"
    assert json_schema["properties"]["field4"]["title"] == "NestedModel"
    assert "field1" in json_schema["properties"]["field4"]["properties"]
    assert "field2" in json_schema["properties"]["field4"]["properties"]
    assert json_schema["properties"]["field4"]["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field4"]["properties"]["field2"]["type"] == "integer"


def test_dataclass_typing_mapping():
    from typing import Mapping, Any

    @dataclass
    class MappingDataclass:
        """This is a dataclass with a mapping."""
        field1: str
        field2: Mapping[str, Any]

    schema = to_schema(MappingDataclass)
    assert isinstance(schema, dict)
    assert schema["name"] == "MappingDataclass"
    assert schema["description"] == "This is a dataclass with a mapping."
    json_schema = schema["schema"]
    assert json_schema["title"] == "MappingDataclass"
    assert "field1" in json_schema["properties"]
    assert "field2" in json_schema["properties"]
    assert json_schema["properties"]["field1"]["type"] == "string"
    assert json_schema["properties"]["field2"]["type"] == "object"
    assert json_schema["properties"]["field2"]["additionalProperties"] == {}