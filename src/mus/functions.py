import typing as t
from .llm.types import ToolCallableType, FunctionSchema, FunctionSchemaNoAnnotations
from dataclasses import is_dataclass
import json
import attrs
import fastjsonschema

@attrs.define
class ToolCallable():
    function: ToolCallableType
    schema: FunctionSchema


def is_tool_callable(obj: t.Any) -> t.TypeGuard[ToolCallable]:
    """Check if an object is a ToolCallable."""
    return isinstance(obj, ToolCallable)

def tool(**metadata: t.Dict[str, t.Any]):
    def decorator(func: ToolCallableType):
        func.__metadata__ = metadata # type: ignore
        return func
    return decorator


def parse_tools(tools: t.Sequence[ToolCallableType | ToolCallable]) -> t.Sequence[ToolCallable]:
    """Parse a list of tool callables into a list of ToolCallable."""
    result = [
        func
        if is_tool_callable(func)
        else ToolCallable(
            function=func, # type: ignore # will be a ToolCallableType
            schema=to_schema(func)
        )
        for func in (tools or [])
    ]
    return result

def func_to_schema(func: ToolCallableType, ensure_docstring: bool=True) -> FunctionSchema:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"): # type: ignore
            return definition
    if not func.__doc__ and ensure_docstring:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    annotations = list(t.get_type_hints(func).items())
    if annotations and annotations[-1][0] == "return":
        annotations = annotations[:-1]  # Remove the return annotation if present
    p = FunctionSchema(
        name=func.__name__,
        description=func.__doc__ or "",
        schema=get_schema(func.__name__, annotations),
        annotations=annotations
    )
    return p

def schema_to_example(schema: t.Union[FunctionSchema, t.Dict[str, t.Any]]):
    if "schema" in schema:
        schema = schema.get("schema", {})

    if schema.get("type") == "object":
        example = {}
        for key, value in schema.get("properties", {}).items():
            example[key] = schema_to_example(value)
        return example
    elif schema.get("type") == "array":
        item_schema = schema.get("items", {})
        return [schema_to_example(item_schema)]
    elif enum := schema.get("enum"):
        return f"<{'|'.join(enum)}>"
    elif type := schema.get("type"):
        return schema.get("example", f"<{type}>")
    else:
        raise ValueError(f"Couldn't parse schema: {json.dumps(schema, indent=2)}")

def dataclass_to_schema(dataclass) -> FunctionSchema:
    annotations = list(t.get_type_hints(dataclass).items())
    p = FunctionSchema(
        name=dataclass.__name__,
        description=dataclass.__doc__,
        schema=get_schema(dataclass.__name__, annotations),
        annotations=annotations
    )
    return p    

def typedict_to_schema(typed_dict: t.Type[dict]) -> FunctionSchema:
    if not typed_dict.__doc__:
        raise ValueError(f"TypedDict {typed_dict.__name__} is missing a docstring")
    if not hasattr(typed_dict, '__annotations__'):
        raise ValueError(f"TypedDict {typed_dict.__name__} is missing annotations - did you use a dict instead of a TypedDict?")
    
    p = FunctionSchema(
        name=typed_dict.__name__,
        description=typed_dict.__doc__,
        schema=get_schema(typed_dict.__name__, list(typed_dict.__annotations__.items())),
        annotations=list(typed_dict.__annotations__.items())
    )
    return p

def to_schema(obj: t.Union[dict, object, ToolCallableType, t.Callable]) -> FunctionSchema:
    if is_dataclass(obj):
        return dataclass_to_schema(obj)
    elif isinstance(obj, type) and t.is_typeddict(obj):
        return typedict_to_schema(obj)
    elif isinstance(obj, ToolCallableType):
        return func_to_schema(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def remove_keys(obj: t.Union[t.Dict[str, t.Any], t.Any], keys: set):
    if isinstance(obj, dict):
        return {
            key: remove_keys(val, keys)
            for key, val
            in obj.items()
            if key not in keys
        }
    else:
        return obj

def get_pydantic_schema(py_type: t.Type) -> t.Optional[t.Dict[str, t.Any]]:
    if hasattr(py_type, 'model_json_schema') and callable(py_type.model_json_schema):
        return py_type.model_json_schema() # type: ignore
    else:
        return None

def python_type_to_json_schema(py_type: t.Type) -> t.Dict[str, t.Any]:
    """Convert Python type annotations to JSON Schema format."""
    
    # Handle basic types
    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is type(None):
        return {"type": "null"}
    elif is_dataclass(py_type):
        # Handle dataclasses
        return get_schema(py_type.__name__, list(py_type.__annotations__.items()))
    elif t.is_typeddict(py_type):
        # Handle TypedDicts
        return get_schema(py_type.__name__, list(py_type.__annotations__.items()))
    elif (pd_schema := get_pydantic_schema(py_type)):
        # Handle Pydantic models
        return pd_schema
    elif py_type is t.Any:
        return {}
    
    # Handle generic types
    origin = t.get_origin(py_type)
    args = t.get_args(py_type)
    
    if origin in [list, t.List]:
        if args:
            return {
                "type": "array",
                "items": python_type_to_json_schema(args[0])
            }
        else:
            return {"type": "array"}
    elif origin is t.Annotated:
        # Handle Annotated types
        if args:
            base_type = args[0]
            annotations = args[1:]
            schema = python_type_to_json_schema(base_type)
            if annotations:
                schema["description"] = " ".join(str(a) for a in annotations)
            return schema
        else:
            raise ValueError("Annotated type must have at least one argument")
    elif origin in [dict, t.Dict, t.Mapping]:
        schema: dict[str, t.Any] = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return schema
    elif origin is t.Union:
        # Handle Optional types (Union[T, None])
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1 and type(None) in args:
            # This is Optional[T]
            base_schema = python_type_to_json_schema(non_none_types[0])
            return {"anyOf": [base_schema, {"type": "null"}]}
        else:
            # General Union
            return {"anyOf": [python_type_to_json_schema(arg) for arg in args]}
    elif origin is t.Literal:
        # Handle Literal types
        if all(isinstance(arg, str) for arg in args):
            _type = "string"
        elif all(isinstance(arg, int) for arg in args):
            _type = "integer"
        elif all(isinstance(arg, (int, float)) for arg in args):
            _type = "number"
        elif all(isinstance(arg, bool) for arg in args):
            _type = "boolean"
        else:
            raise ValueError(f"Unsupported Literal type with mixed or unsupported types: {args}")
        return {
            "type": _type,
            "enum": list(args)
        }
    
    # raise error for unsupported types
    raise ValueError(f"Unsupported type when converting to schema: {py_type}")

def get_schema(name: str, fields: t.List[t.Tuple[str, t.Type]]) -> t.Dict[str, object]:
    """Generate a JSON schema from field definitions."""
    
    properties = {}
    required = []
    
    for field_name, field_type in fields:
        # Try to get description from type metadata
        
        # Convert type to JSON schema
        field_schema = python_type_to_json_schema(field_type)
        if "anyOf" in field_schema and len(field_schema["anyOf"]) == 2 and field_schema["anyOf"][1].get("type") == "null":
            # This is an Optional type, we can remove the null option
            field_schema = field_schema["anyOf"][0]
        else:
            required.append(field_name)
        properties[field_name] = field_schema
        
    
    # Build the complete schema
    schema = {
        "type": "object",
        "title": name,
        "properties": properties,
        "required": required
    }
        
    return schema

def schema_to_attrs(schema: FunctionSchema) -> t.Type:
    """Convert a FunctionSchema to an attrs class."""
    attrs_fields = {
        key: type_to_attr(value)
        for key, value in schema["annotations"]
    }
    
    return attrs.make_class(
        schema["name"],
        attrs_fields,
        auto_attribs=True,
    )

def type_to_attr(value: t.Type) -> attrs.Attribute:
    """Convert a key-value pair to an attrs attribute."""
    if t.is_typeddict(value) or is_dataclass(value):
        return attrs.field(
            type=schema_to_attrs(to_schema(value))
        )
    else:
        return attrs.field(
            type=value,
        )

def verify_schema_inputs(
    schema: FunctionSchema,
    inputs: t.Mapping[str, t.Any],
) -> t.Dict[str, t.Any]:
    """Verify that the inputs match the function's schema."""
    if not schema["schema"] and inputs:
        raise ValueError(f"Invalid inputs for {schema['name']}: Schema is empty and inputs are non-empty")
    try:
        fastjsonschema.validate(schema["schema"], inputs)
    except fastjsonschema.JsonSchemaException as e:
        raise ValueError(f"Invalid inputs for {schema['name']}: {e}") from e
    
    
    return dict(inputs)

def verify_function_inputs(
    func: t.Callable,
    inputs: t.Mapping[str, t.Any],
) -> t.Dict[str, t.Any]:
    """Verify that the inputs match the function call signature."""
    schema = func_to_schema(func, ensure_docstring=False)
    return verify_schema_inputs(schema, inputs)


def remove_annotations(schema: FunctionSchema) -> FunctionSchemaNoAnnotations:
    """Remove the annotations from a FunctionSchema."""
    
    return FunctionSchemaNoAnnotations(
        name=schema["name"],
        description=schema["description"],
        schema=schema["schema"],
    )

if __name__ == "__main__":
    # Example usage
    import dataclasses

    class ExampleNested(t.TypedDict):
        """An example nested TypedDict."""
        nested_field: str
        another_field: int
    
    @dataclasses.dataclass
    class ExampleDataclass:
        """An example dataclass."""
        x: int
        y: str
        nest: ExampleNested
    
    async def example_tool(a: int, b: str, double_nest: ExampleDataclass) -> str:
        """An example tool that takes an integer and a string."""
        return f"Received {a} and {b}"
    
    schema = func_to_schema(example_tool)
    
    example_inputs = {
        "a": 42,
        "b": "Hello",
        "double_nest": {
            "x": 1,
            "y": "Nested",
            "nest": {
                "nested_field": "Inner",
                "another_field": 100
            }
        }
    }
    verified_inputs = verify_schema_inputs(schema, example_inputs)
    print("Verified inputs:", verified_inputs)