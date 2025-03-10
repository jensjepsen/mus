import typing as t
from dataclasses import is_dataclass
from .types import LLMClient, Delta, ToolUse, ToolResult, File, ToolCallableType, Query, Usage, Assistant
from ..functions import get_schema
import base64

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCallParam, ChatCompletionChunk, ChatCompletion
from openai._types import NotGiven
import json

def func_to_tool(func: ToolCallableType) -> ChatCompletionToolParam:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"):  # type: ignore
            return definition
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": get_schema(func.__name__, list(func.__annotations__.items()))
        }
    }

def dataclass_to_tool(dataclass) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": dataclass.__name__,
            "description": dataclass.__doc__,
            "parameters": get_schema(dataclass.__name__, list(dataclass.__annotations__.items()))
        }
    }

def functions_for_llm(functions: t.List[ToolCallableType]) -> t.List[ChatCompletionToolParam]:
    return [
        dataclass_to_tool(func) if is_dataclass(func) else func_to_tool(func)
        for func in (functions or [])
    ]

def file_to_image(file: File) -> str:
    try:
        type_, subtype = file.b64type.split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid b64type: {file.b64type}, must be in format type/subtype")
    if type_ != "image":
        raise ValueError(f"Only supports image/[type], not: {file.b64type}")
    elif subtype not in ["png", "jpeg"]:
        raise ValueError(f"Only supports image/png and image/jpeg, not: {file.b64type}")
    
    return f"data:{file.b64type};base64,{file.content}"

def parse_content(query: t.Union[str, File]) -> t.Union[str, t.Dict[str, str]]:
    if isinstance(query, str):
        return query
    elif isinstance(query, File):
        return {"image": file_to_image(query)}
    else:
        raise ValueError(f"Invalid query type: {type(query)}")

def query_to_messages(query: Query) -> t.List[ChatCompletionMessageParam]:
    messages = []
    for q in query.val:
        if isinstance(q, Assistant):
            messages.append({"role": "assistant", "content": q.val})
        else:
            content = parse_content(q)
            messages.append({"role": "user", "content": content})
    return messages

def parse_tool_content(c: t.Union[str, File]) -> t.Union[str, t.Dict[str, str]]:
    if isinstance(c, str):
        return c
    elif isinstance(c, File):
        return {"image": file_to_image(c)}
    else:
        raise ValueError(f"Invalid tool result type: {type(c)}")

def tool_result_to_content(tool_result: ToolResult) -> t.Union[str, t.List[t.Union[str, t.Dict[str, str]]]]:
    if isinstance(tool_result.content, list):
        return [parse_tool_content(c) for c in tool_result.content]
    else:
        return parse_tool_content(tool_result.content)

def deltas_to_messages(deltas: t.Iterable[t.Union[Query, Delta]]) -> t.List[ChatCompletionMessageParam]:
    messages = []
    for delta in deltas:
        if isinstance(delta, Delta):
            if delta.content["type"] == "text":
                messages.append({"role": "assistant", "content": delta.content["data"]})
            elif delta.content["type"] == "tool_use":
                tool_call: ChatCompletionMessageToolCallParam = {
                    "id": delta.content["data"].id,
                    "type": "function",
                    "function": {
                        "name": delta.content["data"].name,
                        "arguments": json.dumps(delta.content["data"].input)
                    }
                }
                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            elif delta.content["type"] == "tool_result":
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result_to_content(delta.content["data"])),
                    "tool_call_id": delta.content["data"].id
                })
            else:
                raise ValueError(f"Invalid delta type: {delta.content['type']}")
        else:
            messages.extend(query_to_messages(delta))
    return messages

class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]

STREAM_ARGS = StreamArgs
MODEL_TYPE = str

class OpenAILLM(LLMClient[StreamArgs, MODEL_TYPE, openai.AsyncClient]):
    def __init__(self, client: t.Optional[openai.AsyncClient]=None):
        if not client:
            client = openai.AsyncClient()
        self.client = client

    async def stream(self, *,
            prompt: t.Optional[str],
            model: str,
            history: t.List[t.Union[Delta, Query]],
            functions: t.Optional[t.List[ToolCallableType]]=None,
            function_choice: t.Optional[t.Literal["auto", "any"]]=None,
            max_tokens: t.Optional[int]=4096,
            top_k: t.Optional[int]=None,
            top_p: t.Optional[float]=None,
            temperature: t.Optional[float]=None,
            stop_sequences: t.Optional[t.List[str]]=None,
            kwargs: t.Optional[StreamArgs]=None,
            no_stream: t.Optional[bool]=None
        ):
        messages = deltas_to_messages(history)
        if prompt:
            messages.insert(0, {"role": "system", "content": prompt})

        tools = functions_for_llm(functions) if functions else NotGiven()
        
        stream = not no_stream
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if function_choice == "auto" else NotGiven(),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
            stream=stream,
            **kwargs or {}
        )

        def is_stream(response) -> t.TypeGuard[openai.AsyncStream[ChatCompletionChunk]]:
            return stream and hasattr(response, "__aiter__")
        
        def is_not_stream(response) -> t.TypeGuard[ChatCompletion]:
            return not stream
        
        if is_stream(response):
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield Delta(content={"type": "text", "data": delta.content})
                elif delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function:
                            tool_use = ToolUse(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                input=json.loads(tool_call.function.arguments)
                            )
                            yield Delta(content={"type": "tool_use", "data": tool_use})
                if chunk.usage:
                    yield Delta(content={"type": "text", "data": ""}, usage={
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens
                    })
                if chunk.choices[0].finish_reason == "tool_calls":
                    break

                
        elif is_not_stream(response):
            content = response.choices[0].message.content
            if content:
                yield Delta(content={"type": "text", "data": content})
            
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_use = ToolUse(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments)
                    )
                    yield Delta(content={"type": "tool_use", "data": tool_use})
            
            if response.usage:
                yield Delta(content={"type": "text", "data": ""}, usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                })