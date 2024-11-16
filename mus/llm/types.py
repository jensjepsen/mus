import typing as t
from dataclasses import dataclass

from abc import ABC, abstractmethod

from ..types import DataClass

HISTORY_TYPE = t.TypeVar("HISTORY_TYPE")

class LLMClient(ABC, t.Generic[HISTORY_TYPE]):
    @abstractmethod
    def stream(self, prompt: t.Optional[str], query: t.Optional["QueryIterableType"], history: HISTORY_TYPE, functions: t.List[t.Callable], invoke_function: t.Callable, function_choice: t.Literal["auto", "any"]) -> t.Iterable["Delta"]:
        pass


@dataclass
class ToolUse:
    name: str
    input: t.Dict[str, t.Any]

@dataclass
class ToolResult:
    content: "ToolReturnValue"

@dataclass
class Delta:
    type: t.Literal["text", "tool_use", "tool_result"]
    content: t.Union[str, ToolUse, ToolResult]

    def __str__(self) -> str:
        if self.type == "text":
            return self.content
        elif self.type == "tool_use":
            return f"\nRunning tool: {self.content.name}\n"
        elif self.type == "tool_result":
            return f"\nTool result: {self.content.content}\n"

@dataclass
class File:
    b64type: t.Literal["image/png"]
    content: str

    def to_b64(self):
        return f"data:{self.b64type};base64,{self.content}"

    def __add__(self, other):
        return Query([self, other])
    
    def __radd__(self, other):
        return Query([other, self])



ToolSimpleReturnValue = t.Union[str, "File"]
ToolReturnValue = t.Union[t.Sequence[ToolSimpleReturnValue], ToolSimpleReturnValue]
class ToolCallableType(t.Protocol):
    __name__: str
    def __call__(self, *args: t.Any, **kwds: t.Any) -> ToolReturnValue:
        ...

def a() -> ToolReturnValue:
    return ["a", File(b64type="image/png", content="a")]

StructuredType = t.TypeVar("StructuredType", bound=DataClass)

QuerySimpleType = t.Union[str, File]
QueryIterableType = t.List[QuerySimpleType]
QueryType = t.Union[str, File, QueryIterableType, "Query"]

class Query:
    def __init__(self, val: t.Optional[QueryType]=None):
        self.set_val(val or [])

    def set_val(self, val: QueryType):
        if isinstance(val, str) or isinstance(val, File):
            self.val = t.cast(QueryIterableType, [val])
        elif isinstance(val, Query):
            self.val = val.val
        else:
            self.val = val

    def __add__(self, other: QueryType):
        if isinstance(other, str) or isinstance(other, File):
            return Query(self.val + [other])
        elif isinstance(other, Query):
            return Query(self.val + other.val)
        else:
            return Query(self.val + other)
    
    def __radd__(self, other: QueryType):
        if isinstance(other, str) or isinstance(other, File):
            return Query([other] + self.val)
        elif isinstance(other, Query):
            return Query(other.val + self.val)
        else:
            return Query(other + self.val)   
    


LLMDecoratedFunctionReturnType = t.TypeVar("LLMDecoratedFunctionReturnType", covariant=True)

@t.runtime_checkable
class LLMDecoratedFunctionType(t.Protocol, t.Generic[LLMDecoratedFunctionReturnType]):
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> LLMDecoratedFunctionReturnType:
        ...

LLMPromptFunctionArgs = t.ParamSpec("LLMPromptFunctionArgs")