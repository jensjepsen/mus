import typing as t

from dataclasses import dataclass
from textwrap import dedent
from .types import Delta, LLMClient, QueryType, QueryIterableType, File, LLMDecoratedFunctionType, LLMDecoratedFunctionReturnType, Query, LLMPromptFunctionArgs
from ..functions import functions_map
from ..types import DataClass

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
            if msg.type == "text":
                self.total += msg.content
            elif msg.type == "tool_use":
                self.total += f"Running tool: {msg.content.name}"
            elif msg.type == "tool_result":
                self.total += f"Tool applied"
            yield msg
    
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
    
@dataclass
class LLM:
    prompt: t.Optional[str]
    client: LLMClient
    functions: t.Optional[t.List[t.Callable]]=None
    function_choice: t.Literal["auto", "any"] = "auto"
    
    def query(self, query: t.Optional[QueryType]=None, functions: t.Optional[t.List[t.Callable]] = None, function_choice: t.Optional[t.Literal["auto", "any"]] = None, history: t.List[t.Dict[str, t.Any]] = []):
        functions = functions or self.functions or []
        function_choice = function_choice or self.function_choice

        func_map = functions_map(functions)
        def invoke_function(func_name: str, input: t.Dict[str, t.Any]):
            return func_map[func_name](**input)
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
        history = yield from self.client.stream(dedented_prompt, dedented_query, history, functions, invoke_function, function_choice)
        return history

    def __call__(self, query: QueryType, previous: t.Optional[IterableResult]=None):
        return IterableResult(self.query(query, history=previous.history if previous is not None else []))
    
    def fill(self, query: QueryType, structure: t.Type[DataClass]) -> DataClass:
        for msg in self.query(query, functions=[structure], function_choice="any"):
            if msg.type == "tool_result":
                return msg.content.content
        else:
            raise ValueError("No structured response found")
    
    def wrap(self, function: LLMDecoratedFunctionType[LLMDecoratedFunctionReturnType]):
        def decorated_function(query: t.Optional[QueryType]=None) -> LLMDecoratedFunctionReturnType:
            for msg in self.query(query, functions=[function], function_choice="any"):
                if msg.type == "tool_result":
                    return msg.content.content
            else:
                raise ValueError("LLM did not invoke the function")
        return decorated_function

    def bot(self, function: t.Callable[LLMPromptFunctionArgs, QueryType]) -> t.Callable[LLMPromptFunctionArgs, IterableResult]:
        def decorated(*args: LLMPromptFunctionArgs.args, **kwargs: LLMPromptFunctionArgs.kwargs):
            prompt = function(*args, **kwargs)
            return self(prompt)
        return decorated