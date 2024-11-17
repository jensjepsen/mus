import typing as t

from textwrap import dedent
from .types import Delta, LLMClient, QueryType, QueryIterableType, File, LLMDecoratedFunctionType, LLMDecoratedFunctionReturnType, Query, LLMPromptFunctionArgs, ToolCallableType, is_tool_return_value, LLM_CLIENTS, ToolResult
from ..functions import functions_map
from ..types import DataClass
import json

from anthropic import Anthropic, AnthropicBedrock

class IterableResult:
    def __init__(self, iterable: t.Iterable[Delta]):
        self.iterable = iterable
        self.history = []
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

def wrap_client(client: LLM_CLIENTS):
    if isinstance(client, Anthropic) or isinstance(client, AnthropicBedrock):
        from .anthropic import AnthropicLLM
        return AnthropicLLM(client=client)
    else:
        return client

class LLM:
    def __init__(self, prompt: t.Optional[str]=None, *, client: LLM_CLIENTS, functions: t.Optional[t.List[ToolCallableType]]=None, function_choice: t.Literal["auto", "any"] = "auto") -> None:
        self.client = wrap_client(client)
        self.prompt = prompt
        self.functions = functions
        self.function_choice = function_choice
    
    def query(self, query: t.Optional[QueryType]=None, functions: t.Optional[t.List[t.Callable]] = None, function_choice: t.Optional[t.Literal["auto", "any"]] = None, history: t.List[t.Any] = []):
        functions = functions or self.functions or []
        function_choice = function_choice or self.function_choice
        

        func_map = functions_map(functions)
        def invoke_function(func_name: str, input: t.Dict[str, t.Any]):
            result = func_map[func_name](**input)
            if not is_tool_return_value(result):
                result = json.dumps(result)

            return result
        parsed_query: t.Optional[QueryIterableType] = None
        if query:
            if isinstance(query, str) or isinstance(query, File):
                parsed_query = [query]
            elif isinstance(query, list):
                parsed_query = query
            elif isinstance(query, Query):
                parsed_query = query.val
            else:
                raise ValueError(f"Invalid query type: {type(query)}")
            dedented_query = [dedent(q) if isinstance(q, str) else q for q in parsed_query]
        else:
            dedented_query = None
        dedented_prompt = dedent(self.prompt) if self.prompt else None
        if dedented_query:
            history = history + [dedented_query]
        for msg in self.client.stream(prompt=dedented_prompt, history=history, functions=functions, invoke_function=invoke_function, function_choice=function_choice):
            yield msg
            history = history + [msg]
            if msg.content["type"] == "tool_use":
                func_result = invoke_function(msg.content["data"].name, msg.content["data"].input)
                fd = Delta(content={"data": ToolResult(id=msg.content["data"].id, content=func_result), "type": "tool_result"})
                yield fd
                history.append(fd)
                history = yield from self.query(None, functions=functions, function_choice=function_choice, history=history)
                
            
        return history

    def __call__(self, query: QueryType, previous: t.Optional[IterableResult]=None):
        _q = self.query(query, history=previous.history if previous is not None else [])
        return IterableResult(_q)
    
    def fill(self, query: QueryType, structure: t.Type[DataClass]):
        for msg in self.query(query, functions=[structure], function_choice="any"):
            if msg.content["type"] == "tool_result":
                return msg.content["data"].content
        else:
            raise ValueError("No structured response found")
    
    def func(self, function: LLMDecoratedFunctionType[LLMDecoratedFunctionReturnType]):
        def decorated_function(query: t.Optional[QueryType]=None) -> LLMDecoratedFunctionReturnType:
            for msg in self.query(query, functions=[function], function_choice="any"):
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