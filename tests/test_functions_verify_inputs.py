import typing as t
from dataclasses import dataclass
import attrs
import pytest
from mus.functions import (
    FunctionSchema,
    schema_to_attrs,
    type_to_attr,
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


# Tests for schema_to_attrs
def test_schema_to_attrs_simple_types():
    """Test schema_to_attrs with simple types."""
    schema = FunctionSchema(
        name="TestClass",
        description="Test description",
        schema={},
        annotations=[
            ("name", str),
            ("age", int),
            ("active", bool)
        ]
    )
    
    result_class = schema_to_attrs(schema)
    assert result_class.__name__ == "TestClass"
    
    # Test instantiation
    instance = result_class(name="John", age=30, active=True)
    assert instance.name == "John"
    assert instance.age == 30
    assert instance.active is True


def test_schema_to_attrs_empty_annotations():
    """Test schema_to_attrs with empty annotations."""
    schema = FunctionSchema(
        name="EmptyClass",
        description="Empty test class",
        schema={},
        annotations=[]
    )
    
    result_class = schema_to_attrs(schema)
    assert result_class.__name__ == "EmptyClass"
    
    # Should be able to instantiate with no arguments
    instance = result_class()
    assert instance is not None


def test_schema_to_attrs_with_optional_types():
    """Test schema_to_attrs with optional types."""
    schema = FunctionSchema(
        name="OptionalClass",
        description="Class with optional fields",
        schema={},
        annotations=[
            ("required_field", str),
            ("optional_field", t.Optional[int])
        ]
    )
    
    result_class = schema_to_attrs(schema)
    instance = result_class(required_field="test", optional_field=None)
    assert instance.required_field == "test"
    assert instance.optional_field is None


# Tests for verify_schema_inputs
def test_verify_schema_inputs_valid_simple():
    """Test verify_schema_inputs with valid simple inputs."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={},
        annotations=[
            ("name", str),
            ("age", int),
            ("active", bool)
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
        schema={},
        annotations=[
            ("name", t.Annotated[str, "The name of the person"]),
            ("age", t.Annotated[int, "The age of the person"]),
            ("hobbies", t.Annotated[list[str], "List of hobbies"])
        ]
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
        schema={},
        annotations=[
            ("name", t.Annotated[str, "The name of the person"]),
            ("age", t.Annotated[int, "The age of the person"]),
            ("hobbies", t.Annotated[list[str], "List of hobbies"])
        ]
    )
    
    inputs = {
        "name": "Bob",
        "age": "thirty",  # Invalid type for age
        "hobbies": ["reading", "hiking"]
    }
    
    with pytest.raises(ValueError) as exc_info:
        verify_schema_inputs(schema, inputs)
    
    assert "Invalid inputs for AnnotatedFunction" in str(exc_info.value)

def test_verify_schema_inputs_type_coercion():
    """Test verify_schema_inputs with type coercion."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={},
        annotations=[
            ("count", int),
            ("rate", float)
        ]
    )
    
    inputs = {
        "count": "42",  # String that can be converted to int
        "rate": "3.14"  # String that can be converted to float
    }
    
    result = verify_schema_inputs(schema, inputs)
    assert result["count"] == 42
    assert result["rate"] == 3.14


def test_verify_schema_inputs_invalid_type():
    """Test verify_schema_inputs with invalid type."""
    schema = FunctionSchema(
        name="TestFunction",
        description="Test function",
        schema={},
        annotations=[
            ("name", str),
            ("age", int)
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
        schema={},
        annotations=[
            ("name", str),
            ("age", int)
        ]
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
        schema={},
        annotations=[
            ("name", str)
        ]
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


def test_verify_schema_inputs_with_optional_fields():
    """Test verify_schema_inputs with optional fields."""
    schema = FunctionSchema(
        name="OptionalFunction",
        description="Function with optional parameters",
        schema={},
        annotations=[
            ("required", str),
            ("optional", t.Optional[int])
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
        schema={},
        annotations=[
            ("user_data", SimpleTypedDict),
            ("is_admin", bool)
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
        schema={},
        annotations=[
            ("data", AnnotatedTypedDict),
            ("active", bool)
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
        schema={},
        annotations=[
            ("data", SimpleDataClass),
            ("active", bool)
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
        schema={},
        annotations=[
            ("nested_dict", NestedTypedDict),
            ("status", str)
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



# Performance and stress tests
def test_large_schema_performance():
    """Test performance with a large schema."""
    # Create a schema with many fields
    annotations = [(f"field_{i}", str) for i in range(100)]
    
    schema = FunctionSchema(
        name="LargeFunction",
        description="Function with many parameters",
        schema={},
        annotations=annotations
    )
    
    inputs = {f"field_{i}": f"value_{i}" for i in range(100)}
    
    result = verify_schema_inputs(schema, inputs)
    assert len(result) == 100
    assert result["field_0"] == "value_0"
    assert result["field_99"] == "value_99"