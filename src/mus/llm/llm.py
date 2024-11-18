import typing as t

from textwrap import dedent
from .types import Delta, LLMClient, QueryType, File, LLMDecoratedFunctionType, LLMDecoratedFunctionReturnType, Query, LLMPromptFunctionArgs, ToolCallableType, is_tool_return_value, LLM_CLIENTS, ToolResult, STREAM_EXTRA_ARGS, MODEL_TYPE, History
from ..functions import functions_map
from ..types import DataClass
import json

class IterableResult:
    def __init__(self, iterable: t.Iterable[Delta]):
        self.iterable = iterable
        self.history: History = []
        self.has_iterated = False
        self.total = ""

    def __iter__(self):
        def run():
           self.history = yield from self.iterable

        for msg in run():
            self.history.append(msg)
            if msg.content["type"] == "text":
                self.total += msg.content["data"]
            elif msg.content["type"] == "tool_use":
                self.total += f"Running tool: {msg.content['data'].name}"
            elif msg.content["type"] == "tool_result":
                self.total += f"Tool applied"
            yield msg
        self.has_iterated = True
    
    def __str__(self):
        if not self.has_iterated:
            _ = list(self)
        return self.total
    
    def string(self):
        return str(self)
    
    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'IterableResult' and '{type(other)}'")

class _LLMInitAndQuerySharedKwargs(t.TypedDict, total=False):
    functions: t.Optional[t.List[ToolCallableType]]
    function_choice: t.Optional[t.Literal["auto", "any"]]

class LLM(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE]):
    def __init__(self, 
        prompt: t.Optional[str]=None,
        *,
        client: LLMClient[STREAM_EXTRA_ARGS, MODEL_TYPE],
        model: MODEL_TYPE,
        client_kwargs: t.Optional[STREAM_EXTRA_ARGS] = None,
        **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.model = model
        self.client_kwargs = client_kwargs
        self.functions = kwargs.get("functions")
        self.function_choice = kwargs.get("function_choice")

    
    def query(self, *, query: t.Optional[QueryType]=None, history: History = [], max_tokens: t.Optional[int]=None, **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]):
        functions = kwargs.get("functions") or self.functions or []
        function_choice = kwargs.get("function_choice") or self.function_choice
        

        func_map = functions_map(functions)
        def invoke_function(func_name: str, input: t.Dict[str, t.Any]):
            result = func_map[func_name](**input)
            if not is_tool_return_value(result):
                result = json.dumps(result)

            return result
        parsed_query: t.Optional[Query] = None
        if query:
            if isinstance(query, str) or isinstance(query, File):
                parsed_query = Query(val=[query])
            elif isinstance(query, list):
                parsed_query = Query(val=query)
            elif isinstance(query, Query):
                parsed_query = query
            else:
                raise ValueError(f"Invalid query type: {type(query)}")
            dedented_query = [dedent(q) if isinstance(q, str) else q for q in parsed_query.val]
            parsed_query = Query(val=dedented_query)
        
        dedented_prompt = dedent(self.prompt) if self.prompt else None
        
        if parsed_query:
            history = history + [parsed_query]
        
        for msg in self.client.stream(
            prompt=dedented_prompt,
            history=history,
            functions=functions,
            function_choice=function_choice,
            kwargs=self.client_kwargs,
            model=self.model,
            max_tokens=max_tokens
        ):
            yield msg
            history = history + [msg]
            if msg.content["type"] == "tool_use":
                func_result = invoke_function(msg.content["data"].name, msg.content["data"].input)
                fd = Delta(content={"data": ToolResult(id=msg.content["data"].id, content=func_result), "type": "tool_result"})
                yield fd
                history.append(fd)
                history = yield from self.query(functions=functions, function_choice=function_choice, history=history, max_tokens=max_tokens)
                
            
        return history

    def __call__(self, query: QueryType, previous: t.Optional[IterableResult]=None):
        _q = self.query(query=query, history=previous.history if previous is not None else [])
        return IterableResult(_q)
    
    def fill(self, query: QueryType, structure: t.Type[DataClass]):
        for msg in self.query(query=query, functions=[structure], function_choice="any"):
            if msg.content["type"] == "tool_result":
                return msg.content["data"].content
        else:
            raise ValueError("No structured response found")
    
    def func(self, function: LLMDecoratedFunctionType[LLMDecoratedFunctionReturnType]):
        def decorated_function(query: t.Optional[QueryType]=None) -> LLMDecoratedFunctionReturnType:
            for msg in self.query(query=query, functions=[function], function_choice="any"):
                if msg.content["type"] == "tool_use":
                    return function(**(msg.content["data"].input))
            else:
                raise ValueError("LLM did not invoke the function")
        return decorated_function

    def bot(self, function: t.Callable[LLMPromptFunctionArgs, QueryType]) -> t.Callable[LLMPromptFunctionArgs, IterableResult]:
        def decorated(*args: LLMPromptFunctionArgs.args, **kwargs: LLMPromptFunctionArgs.kwargs):
            prompt = function(*args, **kwargs)
            return self(prompt)
        return decorated