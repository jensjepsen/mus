import typing as t
from anthropic import AnthropicBedrock, Anthropic
from anthropic import types as at
from dataclasses import is_dataclass
from .types import LLMClient, Delta, ToolUse, ToolResult, QueryIterableType, QuerySimpleType, File, ToolCallableType, DataClass
from ..functions import get_schema

def func_to_tool(func: ToolCallableType) -> at.ToolParam:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"):
            return definition
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    p = at.ToolParam(name=func.__name__, description=func.__doc__, input_schema=get_schema(func.__name__, list(func.__annotations__.items())))
    return p

def dataclass_to_tool(dataclass) -> at.ToolParam:
    p = at.ToolParam(name=dataclass.__name__, description=dataclass.__doc__, input_schema=get_schema(dataclass.__name__, list(dataclass.__annotations__.items())))
    return p

def functions_for_llm(functions: t.List[ToolCallableType]) -> t.List[at.ToolParam]:
    return [
        dataclass_to_tool(func) if is_dataclass(func) else func_to_tool(func)
        for func
        in (functions or [])
    ]

def parse_query(query: QuerySimpleType):
    if isinstance(query, str):
        return at.TextBlockParam({
            "type": "text",
            "text": query
        })
    elif isinstance(query, File):
        return at.ImageBlockParam({
            "type": "image",
            "source": {
                "data": query.content,
                "media_type": query.b64type,
                "type": "base64"
            }
        })
    else:
        raise ValueError(f"Invalid query type: {type(query)}")

def query_to_content(query: QueryIterableType):
    return [
        parse_query(q)
        for q in query
    ]

def extend_history(history: t.List[at.MessageParam], new_message: at.MessageParam):
    prev_history = history[0:-1]
    last_message = history[-1] if history else None
    
    if last_message and last_message["role"] == new_message["role"]:
        new_history = prev_history + [at.MessageParam(
            role=last_message["role"],
            content=last_message["content"] + new_message["content"]
        )]
    else:
        new_history = history + [new_message]
    return new_history

class AnthropicLLM(LLMClient[t.List[at.MessageParam]]):
    def __init__(self, client: t.Union[AnthropicBedrock, Anthropic]):
        self.client = client

    def stream(self, *, prompt: t.Optional[str], query: t.Optional[QueryIterableType], history: t.List[at.MessageParam], functions: t.List[t.Callable], invoke_function: t.Callable, function_choice: t.Literal["auto", "any"]) -> t.Iterable[Delta]:
        kwargs = {}
        if functions:
            kwargs["tools"] = functions_for_llm(functions)
            if function_choice:
                kwargs["tool_choice"] = {
                    "type": function_choice
                }
        if prompt:
            kwargs["system"] = prompt
        
        if query:
            history = extend_history(history, {"role": "user", "content": query_to_content(query)})
        
        with self.client.messages.stream(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_tokens=1000,
            messages=history,
            extra_headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "computer-use-2024-10-22"
            },
            **kwargs
        ) as response:
            function_blocks: t.List[at.ToolUseBlock] = []
            for event in response:
                if event.type == "text":
                    yield Delta(content={"type": "text", "data": event.text})
                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        function_blocks.append(event.content_block)
                
                elif event.type == "message_stop":
                    history = extend_history(history, {
                        "role": event.message.role,
                        "content": event.message.content
                    })
                    
                    if event.message.stop_reason == "tool_use":
                        func_message_contents = []
                        for block in function_blocks:
                            tool_use = ToolUse(name=block.name, input=block.input)
                            yield Delta(content={
                                "data": tool_use,
                                "type": "tool_use"
                            })
                            func_result = invoke_function(block.name, block.input)

                            yield Delta(content={"data": ToolResult(content=func_result), "type": "tool_result"})
                            
                            func_message_contents.append({
                                "tool_use_id": block.id,
                                "type": "tool_result",
                                "content": query_to_content(func_result)
                            })
                        func_message: at.MessageParam = {
                            "role": "user",
                            "content": func_message_contents
                        }
                        history = extend_history(history, func_message)
                        history = yield from self.stream(prompt=prompt, query=None, history=history, functions=functions, invoke_function=invoke_function, function_choice=function_choice)
        return history
