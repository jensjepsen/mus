import json
import logging
import typing as t
from textwrap import dedent
import sys

from .types import Delta, LLM, QueryType, System, LLMDecoratedFunctionType, LLMDecoratedFunctionReturnType, Query, LLMPromptFunctionArgs, ToolCallableType, is_tool_return_value, ToolResult, STREAM_EXTRA_ARGS, MODEL_TYPE, History, QueryStreamArgs, Usage, CLIENT_TYPE, Assistant, CacheOptions, FunctionSchemaNoAnnotations
from ..functions import to_schema, schema_to_example, parse_tools, ToolCallable, verify_schema_inputs
from ..types import FillableType

logger = logging.getLogger(__name__)

def merge_history(history: History) -> History:
    merged = []
    for msg in history:
        if merged and isinstance(msg, Delta) and msg.content["type"] == "text" and msg.content.get("subtype", "text") == "text":
            if isinstance(merged[-1], Delta) and merged[-1].content["type"] == "text" and merged[-1].content.get("subtype", "text") == "text":
                merged[-1].content["data"] += msg.content["data"]
            else:
                merged.append(msg)
                
        else:
            merged.append(msg)
    
    # prune empty text
    merged = [m for m in merged if not (isinstance(m, Delta) and m.content["type"] == "text" and not m.content["data"].strip())]

    return merged

class IterableResult:
    def __init__(self, iterable: t.AsyncIterable[Delta]):
        self.iterable = iterable
        self.history: History = []
        self.has_iterated = False
        self.total = ""
        self.usage = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_written_input_tokens=0)
    
    async def __aiter__(self):
        async for msg in self.iterable:
            if msg.content["type"] == "text":
                self.total += msg.content["data"]
            elif msg.content["type"] == "tool_use":
                self.total += f"Running tool: {msg.content['data'].name}"
            elif msg.content["type"] == "tool_result":
                self.total += "Tool applied"
            if msg.usage:
                self.usage["input_tokens"] += msg.usage["input_tokens"]
                self.usage["output_tokens"] += msg.usage["output_tokens"]
                self.usage["cache_read_input_tokens"] += msg.usage.get("cache_read_input_tokens", 0)
                self.usage["cache_written_input_tokens"] += msg.usage.get("cache_written_input_tokens", 0)
            if msg.content["type"] == "history":
                # TODO: Merge deltas here
                self.history.extend(merge_history(msg.content["data"]))
            else:
                yield msg
        self.has_iterated = True
    
    async def string(self):
        if not self.has_iterated:
            async for a in self:
                pass
        return self.total

class _LLMInitAndQuerySharedKwargs(QueryStreamArgs, total=False):
    functions: t.Optional[t.Sequence[ToolCallableType | ToolCallable]]
    function_choice: t.Optional[t.Literal["auto", "any"]]
    no_stream: t.Optional[bool]
    cache: t.Optional[CacheOptions]

class _LLMCallArgs(_LLMInitAndQuerySharedKwargs, total=False):
    previous: t.Optional[IterableResult]

QueryOrSystem = t.Union[QueryType, System]

def get_exception_depth():
    """Get the depth of the current exception's traceback"""
    _, _, exc_traceback = sys.exc_info()
    if exc_traceback is None:
        return 0
    
    depth = 0
    tb = exc_traceback
    while tb is not None:
        depth += 1
        tb = tb.tb_next
    return depth


async def invoke_function(func_name: str, input: t.Mapping[str, t.Any], func_map: dict[str, ToolCallable]):
    tool_callable = func_map[func_name]
    try:
        input = verify_schema_inputs(tool_callable["schema"], input)
    except ValueError as e:
        return json.dumps({
            "error": f"{str(e)}",
        })
    
    try:
        result = await tool_callable["function"](**input)
    except TypeError as e:
        depth = get_exception_depth()
        if depth == 1 and type(e).__name__ == "TypeError":
            return json.dumps({
                "error": f"Tool {func_name} was called with incorrect arguments: {input}. Please check the function signature and the input provided.",
            })
        else:
            raise e from e
        
    if not is_tool_return_value(result):
        result = json.dumps(result)

    return result


class Bot(t.Generic[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE]):
    def __init__(self, 
        prompt: t.Optional[str]=None,
        *,
        model: LLM[STREAM_EXTRA_ARGS, MODEL_TYPE, CLIENT_TYPE],
        client_kwargs: t.Optional[STREAM_EXTRA_ARGS] = None,
        **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]
    ) -> None:
        self.client = model
        self.prompt = prompt
        self.client_kwargs = client_kwargs
        self.default_args = kwargs

    
    async def query(self, query: t.Optional[QueryOrSystem]=None, /, *, history: History = [], **kwargs: t.Unpack[_LLMInitAndQuerySharedKwargs]) -> t.AsyncGenerator[Delta, None]:
        kwargs = {**self.default_args, **kwargs}
        functions = kwargs.get("functions") or []
        tools = parse_tools(functions)
            
        function_schemas = [
            FunctionSchemaNoAnnotations({
                "description": tool["schema"]["description"],
                "name": tool["schema"]["name"],
                "schema": tool["schema"]["schema"],
            }) for tool in tools]

        func_map = {
            tool["schema"]["name"]: tool
            for tool in tools
        }

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
        
        if parsed_query:
            if isinstance(last := parsed_query.val[-1], Assistant):
                # if the last part of the query is a prefill
                # assistant message, with echo true, we send that as the
                # first message, and then continue with the rest of the query
                # this is helpful when doing structured generation
                if last.echo:
                    yield Delta(content={"type": "text", "data": last.val})

        async for msg in self.client.stream(
            prompt=dedented_prompt,
            history=history,
            kwargs=self.client_kwargs,
            function_choice=kwargs.get("function_choice", None),
            functions=function_schemas,
            no_stream=kwargs.get("no_stream", None),
            max_tokens=kwargs.get("max_tokens", None),
            top_k=kwargs.get("top_k", None),
            top_p=kwargs.get("top_p", None),
            stop_sequences=kwargs.get("stop_sequences", None),
            temperature=kwargs.get("temperature", None),
            cache=kwargs.get("cache", None),
        ):
            yield msg
            
            history = history + [msg]
            if msg.content["type"] == "tool_use":
                func_result = await invoke_function(msg.content["data"].name, msg.content["data"].input, func_map)
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
        
        
    
    async def fill(
            self,
            query: QueryType,
            structure: t.Type[FillableType],
            strategy: t.Literal["tool_use", "prefill"] = "tool_use",
        ) -> FillableType:
        if strategy == "tool_use":
            as_tool = ToolCallable(
                function=structure,  # type: ignore
                schema=to_schema(structure)
            )
            async for msg in self.query(query, functions=[as_tool], function_choice="any", no_stream=True):
                if msg.content["type"] == "tool_use":
                    input = msg.content["data"].input
                    input = verify_schema_inputs(as_tool["schema"], input)
                    return structure(**input)
            else:
                raise ValueError("No structured response found")
        elif strategy == "prefill":
            schema = to_schema(structure)
            first_prop = list(schema["schema"]["properties"].keys())[0]
            query = Query.parse(query)
            agumented_query = (
                query
                + f"""
                Return a JSON object with that follows this structure:
                <example>
                    {schema_to_example(schema)}
                </example>
                """
                + Assistant("""\
                    ```
                    {
                        \"""" + first_prop + '": ', 
                    echo=True)
            )
            result = (await self(agumented_query, stop_sequences=["```"]).string()).strip()
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            try:
                input = json.loads(result)
                input = verify_schema_inputs(schema, input)
                return structure(**input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode JSON: {result}") from e

        
    
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
