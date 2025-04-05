import json
import logging
import typing as t
from textwrap import dedent

from .types import Delta, LLMClient, QueryType, System, LLMDecoratedFunctionType, LLMDecoratedFunctionReturnType, Query, LLMPromptFunctionArgs, ToolCallableType, is_tool_return_value, ToolResult, STREAM_EXTRA_ARGS, MODEL_TYPE, History, QueryStreamArgs, Usage, CLIENT_TYPE
from ..functions import functions_map
from ..types import DataClass

logger = logging.getLogger(__name__)

class IterableResult:
    def __init__(self, iterable: t.AsyncIterable[Delta]):
        self.iterable = iterable
        self.history: History = []
        self.has_iterated = False
        self.total = ""
        self.usage = Usage(input_tokens=0, output_tokens=0)
    
    async def __aiter__(self):
        async for msg in self.iterable:
            if msg.content["type"] == "text":
                self.total += msg.content["data"]
            elif msg.content["type"] == "tool_use":
                self.total += f"Running tool: {msg.content['data'].name}"
            elif msg.content["type"] == "tool_result":
                self.total += f"Tool applied"
            if msg.usage:
                self.usage["input_tokens"] += msg.usage["input_tokens"]
                self.usage["output_tokens"] += msg.usage["output_tokens"]
            if msg.content["type"] == "history":
                # TODO: Merge deltas here
                self.history.extend(msg.content["data"])
            else:
                yield msg
        self.has_iterated = True
    
    async def string(self):
        if not self.has_iterated:
            async for a in self:
                pass
        return self.total

class _LLMInitAndQuerySharedKwargs(QueryStreamArgs, total=False):
    functions: t.Optional[t.List[ToolCallableType]]
    function_choice: t.Optional[t.Literal["auto", "any"]]
    no_stream: t.Optional[bool]

class _LLMCallArgs(_LLMInitAndQuerySharedKwargs, total=False):
    previous: t.Optional[IterableResult]

QueryOrSystem = t.Union[QueryType, System]

class LLM(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE]):
    def __init__(self, 
        prompt: t.Optional[str]=None,
        *,
        client: LLMClient[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE],
        client_kwargs: t.Optional[STREAM_EXTRA_ARGS] = None,
        model: MODEL_TYPE,
        **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.client_kwargs = client_kwargs
        self.default_args = kwargs
        self.model = model

    
    async def query(self, query: t.Optional[QueryOrSystem]=None, /, *, history: History = [], **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]) -> t.AsyncGenerator[Delta, None]:
        kwargs = {**self.default_args, **kwargs}
        functions = kwargs.get("functions") or []
        

        func_map = functions_map(functions)
        async def invoke_function(func_name: str, input: t.Mapping[str, t.Any]):
            result = await func_map[func_name](**input)
            if not is_tool_return_value(result):
                result = json.dumps(result)

            return result
        parsed_query: t.Optional[Query] = None
        prompt = self.prompt
        if query:
            if isinstance(query, System):
                prompt = query.val
                query = query.query

            parsed_query = Query.parse(query) if query else None
            
        
        dedented_prompt = dedent(prompt) if prompt else None
        
        if parsed_query:
            history = history + parsed_query.to_deltas()
        
        async for msg in self.client.stream(
            prompt=dedented_prompt,
            history=history,
            model=self.model,
            kwargs=self.client_kwargs,
            **kwargs
        ):
            yield msg
            history = history + [msg]
            if msg.content["type"] == "tool_use":
                func_result = await invoke_function(msg.content["data"].name, msg.content["data"].input)
                fd = Delta(content={"data": ToolResult(id=msg.content["data"].id, content=func_result), "type": "tool_result"})
                yield fd
                history.append(fd)
                async for msg in self.query(history=history, **kwargs):
                    if msg.content["type"] == "history":
                        history.extend(msg.content["data"][len(history):])
                    else:
                        yield msg
                
            
        yield Delta(content={"type": "history", "data": history})

    @t.overload
    def __call__(self, query: QueryOrSystem, /, **kwargs: t.Unpack[_LLMCallArgs]) -> IterableResult:
        ...

    @t.overload
    def __call__(self, query: t.Callable[LLMPromptFunctionArgs, QueryOrSystem], /, **kwargs: t.Unpack[_LLMCallArgs]) -> t.Callable[LLMPromptFunctionArgs, IterableResult]:
        ...

    def __call__(self, query: t.Union[QueryOrSystem, t.Callable[LLMPromptFunctionArgs, QueryOrSystem]], /, **kwargs: t.Unpack[_LLMCallArgs]) -> t.Union[IterableResult, t.Callable[LLMPromptFunctionArgs, IterableResult]]:
        if callable(query):
            a = self.bot(query)
            return a 
        else:
            previous = kwargs.pop("previous", None)
            _q = self.query(query, history=previous.history if previous is not None else [], **kwargs)
            return IterableResult(_q)
    
    async def fill(self, query: QueryType, structure: t.Type[DataClass]):
        async for msg in self.query(query, functions=[structure], function_choice="any", no_stream=True): # type: ignore
            if msg.content["type"] == "tool_use":
                return structure(**(msg.content["data"].input))
        else:
            raise ValueError("No structured response found")
        
    
    def fun(self, function: LLMDecoratedFunctionType[LLMDecoratedFunctionReturnType]):
        async def decorated_function(query: QueryType) -> LLMDecoratedFunctionReturnType:
            async for msg in self.query(query, functions=[function], function_choice="any", no_stream=True): # type: ignore
                if msg.content["type"] == "tool_use":
                    return await function(**(msg.content["data"].input))
            else:
                raise ValueError("LLM did not invoke the function")
        return decorated_function

    def bot(self, function: t.Callable[LLMPromptFunctionArgs, QueryOrSystem]) -> t.Callable[LLMPromptFunctionArgs, IterableResult]:
        def decorated(*args: LLMPromptFunctionArgs.args, **kwargs: LLMPromptFunctionArgs.kwargs):
            prompt = function(*args, **kwargs)
            return self(prompt)
        return decorated