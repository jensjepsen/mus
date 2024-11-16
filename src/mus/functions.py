import typing as t
import pydantic
from anthropic import types as at
from dataclasses import is_dataclass
from .llm.types import ToolCallableType

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
    schema = temp_model.model_json_schema()
    return schema

def functions_map(functions: t.List[ToolCallableType]) -> t.Dict[str, ToolCallableType]:
    return {func.__name__: func for func in (functions or [])}

# TODO: below should be moved to llm.anthropic

def func_to_tool(func: ToolCallableType) -> at.ToolParam:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"):
            return definition
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    p = at.ToolParam(name=func.__name__, description=func.__doc__, input_schema=get_schema(func.__name__, list(func.__annotations__.items())))
    return p

def dataclass_to_tool(dataclass: t.Type) -> at.ToolParam:
    p = at.ToolParam(name=dataclass.__name__, description=dataclass.__doc__, input_schema=get_schema(dataclass.__name__, list(dataclass.__annotations__.items())))
    return p

def functions_for_llm(functions: t.List[ToolCallableType]) -> t.List[at.ToolParam]:
    return [
        dataclass_to_tool(func) if is_dataclass(func) else func_to_tool(func)
        for func
        in (functions or [])
    ]