import typing as t
import pydantic
from .llm.types import ToolCallableType
import jsonref

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

