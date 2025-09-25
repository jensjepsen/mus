import typing as t
from dataclasses import dataclass

from abc import ABC, abstractmethod
from pathlib import Path
from textwrap import dedent
import attrs

import io
import base64

@t.runtime_checkable
class TypedDictLike(t.Protocol):
    def __getitem__(self, key: str, /) -> object:
        ...

CLIENT_TYPE = t.TypeVar("CLIENT_TYPE")
STREAM_EXTRA_ARGS = t.TypeVar("STREAM_EXTRA_ARGS", bound=TypedDictLike)
MODEL_TYPE = t.TypeVar("MODEL_TYPE", bound=str)

class FunctionSchema(t.TypedDict):
    name: str
    description: str
    schema: t.Dict[str, t.Any]
    annotations: t.Sequence[t.Tuple[str, t.Type]]

class FunctionSchemaNoAnnotations(t.TypedDict):
    name: str
    description: str
    schema: t.Dict[str, t.Any]


class CacheOptions(t.TypedDict):
    cache_system_prompt: t.Optional[bool]
    cache_tools: t.Optional[bool]

class QueryStreamArgs(t.TypedDict, total=False):
    max_tokens: t.Optional[int]
    temperature: t.Optional[float]
    top_k: t.Optional[int]
    top_p: t.Optional[float]
    stop_sequences: t.Optional[t.List[str]]

class LLMClientStreamArgs(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE], QueryStreamArgs):
    prompt: t.Optional[str]
    history: "History"
    functions: t.Optional[t.Sequence["FunctionSchemaNoAnnotations"]]
    function_choice: t.Optional[t.Literal["auto", "any"]]
    kwargs: t.Optional[STREAM_EXTRA_ARGS]
    no_stream: t.Optional[bool]
    cache: t.Optional[CacheOptions]

class LLM(ABC, t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE]):
    @abstractmethod 
    def __init__(self, model: MODEL_TYPE, client: t.Optional[CLIENT_TYPE]=None) -> None:
        ...
    
    @abstractmethod
    def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[STREAM_EXTRA_ARGS, MODEL_TYPE]]) -> t.AsyncGenerator["Delta", None]:
        ...


@dataclass
class ToolUse:
    id: str
    name: str
    input: t.Mapping[str, t.Any]

@dataclass
class ToolResult:
    id: str
    content: "ToolReturnValue"

@dataclass
class DeltaText():
    data: str
    subtype: t.Literal["text", "reasoning"] = "text"
    metadata: t.Optional[t.Dict[str, t.Any]] = None
    _type: t.Literal["text"] = "text"

@dataclass
class DeltaToolUse():
    data: ToolUse
    _type: t.Literal["tool-use"] = "tool-use"

@dataclass
class DeltaToolResult():
    data: ToolResult
    _type: t.Literal["tool-result"] = "tool-result"

@dataclass
class DeltaHistory():
    data: "History"
    _type: t.Literal["history"] = "history"

DeltaContent = t.Union[DeltaText, DeltaToolUse, DeltaToolResult, DeltaHistory]

@dataclass
class Usage():
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_written_input_tokens: int = 0


@dataclass
class Delta:
    content: DeltaContent
    usage: t.Optional[Usage] = None

    def __str__(self) -> str:
        if isinstance(self.content, DeltaText):
            return self.content.data
        elif isinstance(self.content, DeltaToolUse):
            return f"\nRunning tool: {self.content.data.name}\n"
        elif isinstance(self.content, DeltaToolResult):
            return f"\nTool result: {self.content.data}\n"
        else:
            raise ValueError(f"Invalid delta type: {type(self.content)}")

    """
    def __add__(self, other: "Delta") -> "Delta":
        if isinstance(self.content, DeltaHistory) and isinstance(other.content, DeltaHistory):
            return Delta(content={"type": "history", "data": self.content.data + other.content.data})
        elif isinstance(self.content, DeltaText) and isinstance(other.content, DeltaText):
            return Delta(content={"type": "text", "data": self.content.data + other.content.data})
        else:
            raise ValueError(f"Cannot add deltas of type {type(self.content)} and {type(other.content)}")
    """


@dataclass
class File:
    b64type: str
    content: str
    _type: t.Literal["file"] = "file"
    
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


class StringWithMetadata(str):
    """
    A string with associated metadata.
    Useful for passing additional context from tools.
    The metadata will not be shown to the LLM, but can be used to store extra information, such as presentation hints, source information, etc.
    """
    def __new__(cls, value="", metadata: t.Optional[t.Dict[str, t.Any]] = None):
        instance = super().__new__(cls, value)
        instance.metadata = metadata or {}
        return instance

    def __init__(self, value="", metadata: t.Optional[t.Dict[str, t.Any]] = None):
        super().__init__()
        self.metadata = metadata or {}

ToolSimpleReturnValue = t.Union[str, "File", StringWithMetadata]
ToolReturnValue = t.Union[t.Sequence[ToolSimpleReturnValue], ToolSimpleReturnValue]

def is_tool_return_value(val: t.Any) -> t.TypeGuard[ToolReturnValue]:
    return isinstance(val, str) or isinstance(val, File) or isinstance(val, StringWithMetadata) or (isinstance(val, list) and all(is_tool_return_value(v) for v in val))

def is_tool_simple_return_value(val: t.Any) -> t.TypeGuard[ToolSimpleReturnValue]:
    return isinstance(val, str) or isinstance(val, File) or isinstance(val, StringWithMetadata)

@t.runtime_checkable
class ToolCallableType(t.Protocol):
    __name__: str
    #__metadata__: t.Optional[t.Dict[str, t.Any]]
    async def __call__(self, *args: t.Any, **kwds: t.Any) -> ToolReturnValue:
        ...

QueryType = t.Union["QuerySimpleType", "QueryIterableType", "Query"]


def is_query_type(val: t.Any) -> t.TypeGuard[QueryType]:
    return isinstance(val, str) or isinstance(val, File) or isinstance(val, Query) or (isinstance(val, list) and all(is_query_type(v) for v in val))

@attrs.define
class System:
    val: str
    query: t.Optional[QueryType] = None
    
    def __add__(self, other: t.Union[QueryType, "System"]):
        if isinstance(other, System):
            return System(self.val + other.val)
        return System(self.val, other)

@attrs.define
class Assistant:
    val: str
    echo: bool = False  # Whether to echo the assistant's response in the query
    _type: t.Literal["assistant"] = "assistant"
    
    def __add__(self, other: QueryType):
        if isinstance(other, Assistant):
            return Assistant(self.val + other.val)
        
        return Query(self) + other

    def __radd__(self, other: QueryType):
        if isinstance(other, Assistant):
            return Assistant(other.val + self.val)
        
        return other + Query(self)

QuerySimpleType = t.Union[str, File, Assistant]
QueryIterableType = t.List[QuerySimpleType]

@attrs.define
class Query:
    val: QueryIterableType
    @classmethod
    def parse(cls, query: QueryType) -> "Query":
        if isinstance(query, str) or isinstance(query, File) or isinstance(query, Assistant):
            parsed_query = Query(val=[query])
        elif isinstance(query, list):
            parsed_query = Query(val=query)
        elif isinstance(query, Query):
            parsed_query = query
        else:
            raise ValueError(f"Invalid query type: {type(query)}")
        
        return parsed_query

    def __init__(self, val: t.Optional[QueryType]=None):
        self.set_val(val or [])

    def to_deltas(self):
        return [self]

    def set_val(self, val: QueryType):
        if isinstance(val, str) or isinstance(val, File) or isinstance(val, Assistant):
            if isinstance(val, str):
                val = dedent(val)
            self.val = t.cast(QueryIterableType, [val])
        elif isinstance(val, Query):
            self.val = val.val
        else:
            self.val = val

    def __add__(self, other: QueryType):
        if isinstance(other, str) or isinstance(other, File) or isinstance(other, Assistant):
            return Query(self.val + [other])
        elif isinstance(other, Query):
            return Query(self.val + other.val)
        else:
            return Query(self.val + other)
    
    def __radd__(self, other: QueryType):
        if isinstance(other, str) or isinstance(other, File) or isinstance(other, Assistant):
            return Query([other] + self.val)
        elif isinstance(other, Query):
            return Query(other.val + self.val)
        else:
            return Query(other + self.val)

HistoryItem = t.Union[Delta, Query]
History = t.List[HistoryItem]

LLMDecoratedFunctionReturnType = t.TypeVar("LLMDecoratedFunctionReturnType", covariant=True)

@t.runtime_checkable
class LLMDecoratedFunctionType(t.Protocol, t.Generic[LLMDecoratedFunctionReturnType]):
    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> LLMDecoratedFunctionReturnType:
        ...

LLMPromptFunctionArgs = t.ParamSpec("LLMPromptFunctionArgs")