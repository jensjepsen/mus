import typing as t
from .types import LLM, Delta, ToolUse, ToolResult, File, Query, Assistant, LLMClientStreamArgs, ToolSimpleReturnValue, is_tool_simple_return_value, FunctionSchemaNoAnnotations

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessageToolCallParam, ChatCompletionChunk, ChatCompletion
from openai._types import NotGiven
import json
import dataclasses

def func_to_tool(func: FunctionSchemaNoAnnotations) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": func["name"],
            "description": func["description"],
            "parameters": func["schema"]
        }
    }
def functions_for_llm(functions: t.Sequence[FunctionSchemaNoAnnotations]) -> t.List[ChatCompletionToolParam]:
    return [
        func_to_tool(func)
        for func in (functions or [])
    ]

def is_valid_image_mime_type(mime_type: str) -> t.TypeGuard[t.Literal["image/png", "image/jpeg", "image/gif", "image/webp"]]:
    """
    Check if the mime type is a valid image type.
    """
    return mime_type in ["image/png", "image/jpeg", "image/gif", "image/webp"]

def file_to_image(file: File) -> str:
    if not is_valid_image_mime_type(file.b64type):
        raise ValueError(f"Only supports image/png, image/jpeg image/gif and image/webp, not: {file.b64type}")
    
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

def parse_tool_content(c: ToolSimpleReturnValue) -> t.Union[str, t.Dict[str, str]]:
    if isinstance(c, str):
        return c
    elif isinstance(c, File):
        return {"image": file_to_image(c)}
    else:
        raise ValueError(f"Invalid tool result type: {type(c)}")

def tool_result_to_content(tool_result: ToolResult) -> t.Union[str, t.List[t.Union[str, t.Dict[str, str]]]]:
    if is_tool_simple_return_value(tool_result.content):
        return [parse_tool_content(tool_result.content)]
    elif isinstance(tool_result.content, list):
        return [parse_tool_content(c) for c in tool_result.content]
    else:
        raise ValueError(f"Invalid tool result content type: {type(tool_result.content)}")

def deltas_to_messages(deltas: t.Iterable[t.Union[Query, Delta]]) -> t.List[ChatCompletionMessageParam]:
    messages = []
    for delta in deltas:
        if isinstance(delta, Delta):
            if delta.content["type"] == "text":
                if delta.content["data"]:
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

@dataclasses.dataclass
class PartialToolCall:
    id: str
    name: str
    arguments: str

class OpenAILLM(LLM[StreamArgs, MODEL_TYPE, openai.AsyncClient]):
    def __init__(self, model: MODEL_TYPE, client: t.Optional[openai.AsyncClient]=None):
        if not client:
            client = openai.AsyncClient()
        self.client = client
        self.model = model

    async def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[StreamArgs, MODEL_TYPE]]):
        messages = deltas_to_messages(kwargs.get("history", []))
        if prompt := kwargs.get("prompt", None):
            messages.insert(0, {"role": "system", "content": prompt})

        if functions := kwargs.get("functions", None):
            tools = functions_for_llm(functions)
        else:
            tools = NotGiven()
        
        stream = not kwargs.get("no_stream", False)
        extra_kwargs = kwargs.get("kwargs", None) or {}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if kwargs.get("function_choice", None) == "auto" else NotGiven(),
            max_tokens=kwargs.get("max_tokens", None) or NotGiven(),
            temperature=kwargs.get("temperature", None) or NotGiven(),
            top_p=kwargs.get("top_p", None) or NotGiven(),
            stop=kwargs.get("stop_sequences", None) or NotGiven(),
            stream=stream,
            **extra_kwargs,
        )

        def is_stream(response) -> t.TypeGuard[openai.AsyncStream[ChatCompletionChunk]]:
            return stream and hasattr(response, "__aiter__")
        
        def is_not_stream(response) -> t.TypeGuard[ChatCompletion]:
            return not stream
        
        if is_stream(response):
            partial_calls: list[PartialToolCall] = []
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield Delta(content={"type": "text", "data": delta.content})
                elif delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.type and not tool_call.type == "function":
                            raise ValueError(f"Only function tool calls are supported, not: {tool_call.type}")
                        
                        if tool_call.function:
                            if not partial_calls or (tool_call.id and partial_calls[-1].id and tool_call.id != partial_calls[-1].id):
                                partial_calls.append(PartialToolCall(
                                    id=tool_call.id if tool_call.id else "",
                                    name="",
                                    arguments=""
                                ))
                            last_call = partial_calls.pop()
                            
                            if not last_call:
                                raise ValueError("Received tool call chunk without a starting id")
                            
                            if tool_call.function.arguments:
                                last_call.arguments += tool_call.function.arguments
                            
                            if tool_call.function.name:
                                last_call.name += tool_call.function.name
                            partial_calls.append(last_call)
                            
                if chunk.usage:
                    yield Delta(content={"type": "text", "data": ""}, usage={
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                        "cache_read_input_tokens": 0,
                        "cache_written_input_tokens": 0
                    })
                if chunk.choices[0].finish_reason == "tool_calls":
                    for call in partial_calls:
                        tool_use = ToolUse(
                            id=call.id,
                            name=call.name,
                            input=json.loads(call.arguments)
                        )
                        yield Delta(content={"type": "tool_use", "data": tool_use})
                    partial_calls = []

                
        elif is_not_stream(response):
            content = response.choices[0].message.content
            if content:
                yield Delta(content={"type": "text", "data": content})
            
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.type and not tool_call.type == "function":
                        raise ValueError(f"Only function tool calls are supported, not: {tool_call.type}")
                    tool_use = ToolUse(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments)
                    )
                    yield Delta(content={"type": "tool_use", "data": tool_use})
            
            if response.usage:
                yield Delta(content={"type": "text", "data": ""}, usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "cache_read_input_tokens": 0,
                    "cache_written_input_tokens": 0
                })