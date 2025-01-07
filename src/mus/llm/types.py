import typing as t
from dataclasses import dataclass

from abc import ABC, abstractmethod
from pathlib import Path

import io
import base64

from ..types import DataClass

if t.TYPE_CHECKING:
    from anthropic import AnthropicBedrock, Anthropic

@t.runtime_checkable
class TypedDictLike(t.Protocol):
    def __getitem__(self, key: str, /) -> object:
        ...

LLM_CLIENTS = t.Union["Anthropic", "AnthropicBedrock", "LLMClient"]
STREAM_EXTRA_ARGS = t.TypeVar("STREAM_EXTRA_ARGS", bound=TypedDictLike)
MODEL_TYPE = t.TypeVar("MODEL_TYPE", bound=str)

class QueryStreamArgs(t.TypedDict, total=False):
    max_tokens: t.Optional[int]
    temperature: t.Optional[float]
    top_k: t.Optional[int]
    top_p: t.Optional[float]

class LLMClientStreamArgs(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE], QueryStreamArgs):
    model: t.Required[MODEL_TYPE]
    prompt: t.Optional[str]
    history: "History"
    functions: t.Optional[t.List["ToolCallableType"]]
    function_choice: t.Optional[t.Literal["auto", "any"]]
    kwargs: t.Optional[STREAM_EXTRA_ARGS]

class LLMClient(ABC, t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE]):
    @abstractmethod
    def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[STREAM_EXTRA_ARGS, MODEL_TYPE]]) -> t.AsyncGenerator["Delta", None]:
        pass

History = t.List[t.Union["Delta", "Query"]]

@dataclass
class ToolUse:
    id: str
    name: str
    input: t.Dict[str, t.Any]

@dataclass
class ToolResult:
    id: str
    content: "ToolReturnValue"

class DeltaText(t.TypedDict):
    type: t.Literal["text"]
    data: str

class DeltaToolUse(t.TypedDict):
    type: t.Literal["tool_use"]
    data: ToolUse

class DeltaToolResult(t.TypedDict):
    type: t.Literal["tool_result"]
    data: ToolResult

class DeltaHistory(t.TypedDict):
    type: t.Literal["history"]
    data: History

DeltaContent = DeltaText | DeltaToolUse | DeltaToolResult | DeltaHistory

class Usage(t.TypedDict):
    input_tokens: int
    output_tokens: int

@dataclass
class Delta:
    content: DeltaContent
    usage: t.Optional[Usage] = None

    def __str__(self) -> str:
        if self.content["type"] == "text":
            return self.content["data"]
        elif self.content["type"] == "tool_use":
            return f"\nRunning tool: {self.content["data"].name}\n"
        elif self.content["type"] == "tool_result":
            return f"\nTool result: {self.content['data']}\n"
        else:
            raise ValueError(f"Invalid delta type: {self.content['type']}")

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
    
    @staticmethod
    def image(path: t.Union[str, Path], as_format: t.Literal["png", "jpeg"]="png"):
        from PIL import Image
        if isinstance(path, str):
            path = Path(path)
        img = Image.open(path)
        # Create a BytesIO object
        buffered = io.BytesIO()
        # Save the image to the BytesIO object in PNG format
        img.save(buffered, format=as_format.upper())
        # Get the byte value of the image
        img_byte = buffered.getvalue()
        # Encode the bytes to base64
        img_base64 = base64.b64encode(img_byte)
        # Convert bytes to string
        img_base64_string = img_base64.decode()
        return File(b64type="image/png", content=img_base64_string)


ToolSimpleReturnValue = t.Union[str, "File"]
ToolReturnValue = t.Union[t.List[ToolSimpleReturnValue], ToolSimpleReturnValue]

def is_tool_return_value(val: t.Any) -> t.TypeGuard[ToolReturnValue]:
    return isinstance(val, str) or isinstance(val, File) or (isinstance(val, list) and all(is_tool_return_value(v) for v in val))

@t.runtime_checkable
class ToolCallableType(t.Protocol):
    __name__: str
    #__metadata__: t.Optional[t.Dict[str, t.Any]]
    async def __call__(self, *args: t.Any, **kwds: t.Any) -> ToolReturnValue:
        ...

StructuredType = t.TypeVar("StructuredType", bound=DataClass)

QuerySimpleType = t.Union[str, File]
QueryIterableType = t.List[QuerySimpleType]
QueryType = t.Union[str, File, QueryIterableType, "Query"]

def is_query_type(val: t.Any) -> t.TypeGuard[QueryType]:
    return isinstance(val, str) or isinstance(val, File) or isinstance(val, Query) or (isinstance(val, list) and all(is_query_type(v) for v in val))

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
    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> LLMDecoratedFunctionReturnType:
        ...

LLMPromptFunctionArgs = t.ParamSpec("LLMPromptFunctionArgs")