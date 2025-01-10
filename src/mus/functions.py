import typing as t
import pydantic
from anthropic import types as at
from dataclasses import is_dataclass
from .llm.types import ToolCallableType
import jsonref
import json

def tool(**metadata: t.Dict[str, t.Any]):
    def decorator(func: ToolCallableType):
        func.__metadata__ = metadata
        return func
    return decorator

def get_schema(name: str, fields: t.List[tuple[str, t.Type]]):
    model_fields = {}
    for field_name, type in fields:
        try:
            description = type.__metadata__[0]
        except (AttributeError, IndexError):
            description = None
        model_fields[field_name] = (type, pydantic.Field(..., description=description))
    temp_model: t.Type[pydantic.BaseModel] = pydantic.create_model(name, **model_fields)
    schema = temp_model.model_json_schema(mode="serialization")
    schema = json.loads(json.dumps(jsonref.replace_refs(schema)))
    return schema

def functions_map(functions: t.List[ToolCallableType]) -> t.Dict[str, ToolCallableType]:
    return {func.__name__: func for func in (functions or [])}

