import typing as t
import pydantic
from .llm.types import ToolCallableType
import jsonref
from dataclasses import is_dataclass
import json

class FunctionSchema(t.TypedDict):
    name: str
    description: str
    schema: t.Dict[str, t.Any]

def tool(**metadata: t.Dict[str, t.Any]):
    def decorator(func: ToolCallableType):
        func.__metadata__ = metadata # type: ignore
        return func
    return decorator

def remove_keys(obj: t.Union[t.Dict[str, t.Any], t.Any], keys: set):
    if isinstance(obj, dict):
        return {
            key: remove_keys(val, keys)
            for key, val
            in obj.items()
            if not key in keys
        }
    else:
        return obj

def func_to_schema(func: ToolCallableType) -> FunctionSchema:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"): # type: ignore
            return definition
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    p = FunctionSchema(
        name=func.__name__,
        description=func.__doc__,
        schema=get_schema(func.__name__, list(func.__annotations__.items()))
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
    p = FunctionSchema(
        name=dataclass.__name__,
        description=dataclass.__doc__,
        schema=get_schema(dataclass.__name__, list(dataclass.__annotations__.items()))
    )
    return p    

def typedict_to_schema(typed_dict: t.Type[dict]) -> FunctionSchema:
    if not typed_dict.__doc__:
        raise ValueError(f"TypedDict {typed_dict.__name__} is missing a docstring")
    if not hasattr(typed_dict, '__annotations__'):
        raise ValueError(f"TypedDict {typed_dict.__name__} is missing annotations - did you forget to use TypedDict?")
    
    p = FunctionSchema(
        name=typed_dict.__name__,
        description=typed_dict.__doc__,
        schema=get_schema(typed_dict.__name__, list(typed_dict.__annotations__.items()))
    )
    return p

def to_schema(obj: t.Union[dict, object, ToolCallableType]) -> FunctionSchema:
    if is_dataclass(obj):
        return dataclass_to_schema(obj)
    elif isinstance(obj, type) and t.is_typeddict(obj):
        return typedict_to_schema(obj)
    elif isinstance(obj, ToolCallableType):
        return func_to_schema(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")

def get_schema(name: str, fields: t.List[tuple[str, t.Type]]) -> t.Dict[str, object]:
    model_fields = {}
    for field_name, type in fields:
        try:
            description = type.__metadata__[0]
        except (AttributeError, IndexError):
            description = None
        model_fields[field_name] = (type, pydantic.Field(..., description=description))
    temp_model: t.Type[pydantic.BaseModel] = pydantic.create_model(name, **model_fields)
    schema = temp_model.model_json_schema()
    dereffed = jsonref.replace_refs(schema)
    cleaned = remove_keys(dereffed, {"$defs"})
    
    return cleaned # type: ignore

def functions_map(functions: t.List[ToolCallableType]) -> t.Dict[str, ToolCallableType]:
    return {func.__name__: func for func in (functions or [])}