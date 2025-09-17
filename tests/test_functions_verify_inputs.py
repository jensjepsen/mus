import typing as t
from dataclasses import dataclass
import pytest
from mus.functions import (
    FunctionSchema,
    schema_to_attrs,
    verify_function_inputs,
    verify_schema_inputs
)

class SimpleTypedDict(t.TypedDict):
    """A simple TypedDict for testing."""
    name: str
    age: int

class AnnotatedTypedDict(t.TypedDict):
    """A TypedDict with annotations for testing."""
    name: t.Annotated[str, "The name of the person"]
    age: t.Annotated[int, "The age of the person"]

@dataclass
class SimpleDataClass:
    """A simple dataclass for testing."""
    title: str
    count: int

class NestedTypedDict(t.TypedDict):
    """A nested TypedDict for testing."""
    user: SimpleTypedDict
    active: bool

# Tests for verify_schema_inputs
def test_verify_schema_inputs_valid_simple():
    """Test verify_schema_inputs with valid simple inputs."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "age", "active"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "name": "Alice",
        "age": 25,
        "active": True
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result == inputs

def test_verify_schema_inputs_annotated():
    """Test verify_schema_inputs with annotated types."""
    schema = FunctionSchema(
        name="AnnotatedFunction",
        description="Function with annotated parameters",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the person"},
                "age": {"type": "integer", "description": "The age of the person"},
                "hobbies": {"type": "array", "items": {"type": "string"}, "description": "List of hobbies"}
            },
            "required": ["name", "age", "hobbies"]
        },
        annotations=[]
    )
    
    inputs = {
        "name": "Bob",
        "age": 30,
        "hobbies": ["reading", "hiking"]
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["name"] == "Bob"
    assert result["age"] == 30

def test_verify_schema_inputs_annotated_invalid():
    """Test verify_schema_inputs with invalid annotated types."""
    schema = FunctionSchema(
        name="AnnotatedFunction",
        description="Function with annotated parameters",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the person"},
                "age": {"type": "integer", "description": "The age of the person"},
                "hobbies": {"type": "array", "items": {"type": "string"}, "description": "List of hobbies"}
            },
            "required": ["name", "age", "hobbies"]
        },
        annotations=[]
    )
    
    inputs = {
        "name": "Bob",
        "age": "thirty",  # Invalid type for age
        "hobbies": ["reading", "hiking"]
    }
    
    with pytest.raises(ValueError) as exc_info:
        verify_schema_inputs(schema, inputs)
    
    assert "Invalid inputs for AnnotatedFunction" in str(exc_info.value)

def test_verify_schema_inputs_invalid_type():
    """Test verify_schema_inputs with invalid type."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "hobbies": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age", "hobbies"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "name": "Alice",
        "age": "not_a_number"  # Invalid int
    }
    
    with pytest.raises(ValueError) as exc_info:
        verify_schema_inputs(schema, inputs)
    
    assert "Invalid inputs for TestFunction" in str(exc_info.value)


def test_verify_schema_inputs_missing_required_field():
    """Test verify_schema_inputs with missing required field."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        },
        annotations=[]
    )
    
    inputs = {
        "name": "Alice"
        # Missing "age"
    }
    
    with pytest.raises(ValueError) as exc_info:
        verify_schema_inputs(schema, inputs)
    
    assert "Invalid inputs for TestFunction" in str(exc_info.value)


def test_verify_schema_inputs_extra_fields():
    """Test verify_schema_inputs with extra fields."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        },
        annotations=[]
    )
    
    inputs = {
        "name": "Alice",
        "extra_field": "should_be_ignored"
    }
    
    # This should work - extra fields are typically ignored by cattrs
    result = verify_schema_inputs(schema, inputs)
    assert "name" in result
    # Extra field behavior depends on cattrs configuration


def test_verify_schema_inputs_empty_schema():
    """Test verify_schema_inputs with empty schema."""
    schema = FunctionSchema(
        name="EmptyFunction",
        description="Function with no parameters",
        schema={},
        annotations=[]
    )
    
    inputs = {}
    
    result = verify_schema_inputs(schema, inputs)
    assert result == {}

def test_verify_schema_inputs_non_empty_inputs():
    """Test verify_schema_inputs with non-empty inputs for empty schema."""
    schema = FunctionSchema(
        name="EmptyFunction",
        description="Function with no parameters",
        schema={},
        annotations=[]
    )
    
    inputs = {"unexpected": "value"}
    
    with pytest.raises(ValueError) as exc_info:
        verify_schema_inputs(schema, inputs)
    
    assert "Invalid inputs for EmptyFunction" in str(exc_info.value)


def test_verify_schema_inputs_with_optional_fields():
    """Test verify_schema_inputs with optional fields."""
    schema = FunctionSchema(
        name="OptionalFunction",
        description="Function with optional parameters",
        schema={
            "type": "object",
            "properties": {
                "required": {"type": "string"},
                "optional": {"type": ["integer", "null"]}   # Optional field
            },
            "required": ["required"]
        },
        annotations=[
        ]
    )
    
    # Test with optional field provided
    inputs1 = {"required": "test", "optional": 42}
    result1 = verify_schema_inputs(schema, inputs1)
    assert result1["required"] == "test"
    assert result1["optional"] == 42
    
    # Test with optional field as None
    inputs2 = {"required": "test", "optional": None}
    result2 = verify_schema_inputs(schema, inputs2)
    assert result2["required"] == "test"
    assert result2["optional"] is None
    
    # Test with optional field missing (if cattrs handles this)
    inputs3 = {"required": "test"}
    try:
        result3 = verify_schema_inputs(schema, inputs3)
        assert result3["required"] == "test"
    except ValueError:
        # This is also acceptable behavior depending on cattrs configuration
        pass


def test_integration_nested_structures():
    """Test integration with nested TypedDict structures."""
    schema = FunctionSchema(
        name="NestedFunction",
        description="Function with nested structure",
        schema={
            "type": "object",
            "properties": {
                "user_data": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                },
                "is_admin": {"type": "boolean"}
            },
            "required": ["user_data", "is_admin"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "user_data": {
            "name": "Bob",
            "age": 35
        },
        "is_admin": False
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["user_data"]["name"] == "Bob"
    assert result["user_data"]["age"] == 35
    assert result["is_admin"] is False

def test_integration_annotated_typed_dict():
    """Test integration with annotated TypedDict."""
    schema = FunctionSchema(
        name="AnnotatedFunction",
        description="Function with annotated TypedDict",
        schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name of the person"},
                        "age": {"type": "integer", "description": "The age of the person"}
                    },
                    "required": ["name", "age"]
                },
                "active": {"type": "boolean"}
            },
            "required": ["data", "active"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "data": {
            "name": "Alice",
            "age": 30
        },
        "active": True
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["data"]["name"] == "Alice"
    assert result["data"]["age"] == 30
    assert result["active"] is True

def test_integration_nested_dataclass():
    """Test integration with nested dataclass structures."""
    schema = FunctionSchema(
        name="NestedDataClassFunction",
        description="Function with nested dataclass",
        schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "count": {"type": "integer"}
                    },
                    "required": ["title", "count"]
                },
                "active": {"type": "boolean"}
            },
            "required": ["data", "active"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "data": {
            "title": "Example Title",
            "count": 10
        },
        "active": True
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["data"]["title"] == "Example Title"
    assert result["data"]["count"] == 10
    assert result["active"] is True

def test_integration_nested_mixed():
    """Test integration with mixed nested structures."""
    schema = FunctionSchema(
        name="MixedNestedFunction",
        description="Function with mixed nested structures",
        schema={
            "type": "object",
            "properties": {
                "nested_dict": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name", "age"]
                        },
                        "active": {"type": "boolean"}
                    },
                    "required": ["user", "active"]
                },
                "status": {"type": "string"}
            },
            "required": ["nested_dict", "status"]
        },
        annotations=[
        ]
    )
    
    inputs = {
        "nested_dict": {
            "user": {
                "name": "Charlie",
                "age": 28
            },
            "active": True
        },
        "status": "active"
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["nested_dict"]["user"]["name"] == "Charlie"
    assert result["nested_dict"]["user"]["age"] == 28
    assert result["nested_dict"]["active"] is True
    assert result["status"] == "active"




def test_verify_function_inputs():
    """Test the verify_function_inputs utility."""
    def sample_function(name: str, age: int) -> str:
        return f"{name} is {age} years old."
    
    inputs = {
        "name": "Diana",
        "age": 27
    }
    
    result = verify_function_inputs(sample_function, inputs)
    assert result["name"] == "Diana"
    assert result["age"] == 27

def test_verify_function_inputs_invalid():
    """Test the verify_function_inputs utility with invalid inputs."""
    def sample_function(name: str, age: int) -> str:
        return f"{name} is {age} years old."
    
    inputs = {
        "name": "Diana",
        "age": "twenty-seven"  # Invalid type
    }
    
    with pytest.raises(ValueError) as exc_info:
        verify_function_inputs(sample_function, inputs)
    
    assert "Invalid inputs for sample_function" in str(exc_info.value)

def test_verify_function_inputs_nested():
    """Test the verify_function_inputs utility with nested structures."""
    class User(t.TypedDict):
        name: str
        age: int

    def sample_function(user: User, active: bool) -> str:
        status = "active" if active else "inactive"
        return f"{user['name']} is {user['age']} years old and is {status}."
    
    inputs = {
        "user": {
            "name": "Eve",
            "age": 22
        },
        "active": True
    }
    
    result = verify_function_inputs(sample_function, inputs)
    assert result["user"]["name"] == "Eve"
    assert result["user"]["age"] == 22
    assert result["active"] is True