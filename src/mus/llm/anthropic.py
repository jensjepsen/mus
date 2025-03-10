import typing as t
from anthropic import AsyncAnthropicBedrock, AsyncAnthropic, NotGiven
from anthropic import types as at
from dataclasses import is_dataclass
from .types import LLMClient, Delta, ToolUse, ToolResult, File, ToolCallableType, Query, Usage, Assistant, LLMClientStreamArgs
from ..functions import get_schema

def func_to_tool(func: ToolCallableType) -> at.ToolParam:
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"): # type: ignore
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

def file_to_image(file: File) -> at.ImageBlockParam:
    return at.ImageBlockParam({
        "type": "image",
        "source": {
            "data": file.content,
            "media_type": file.b64type,
            "type": "base64"
        }
    })

def str_to_text_block(s: str) -> at.TextBlockParam:
    return at.TextBlockParam({
        "type": "text",
        "text": s
    })

def parse_content(query: t.Union[str, File]):
    if isinstance(query, str):
        return str_to_text_block(query)
    elif isinstance(query, File):
        return file_to_image(query)
    else:
        raise ValueError(f"Invalid query type: {type(query)}")

def query_to_content(query: Query):
    for q in query.val:
        if isinstance(q, Assistant):
            yield at.MessageParam(
                role="assistant",
                content=[
                    at.TextBlock(
                        type="text",
                        text=q.val
                    )
                ]
            )
        else:
            yield at.MessageParam(
                role="user",
                content=[parse_content(q)]
            )

def tool_result_to_content(tool_result: ToolResult):
    if isinstance(tool_result.content, str):
        return [str_to_text_block(tool_result.content)]
    elif isinstance(tool_result.content, File):
        return [file_to_image(tool_result.content)]
    elif isinstance(tool_result.content, list):
        return [
            parse_content(c)
            for c in tool_result.content
        ]
    else:
        raise ValueError(f"Invalid tool result type: {type(tool_result.content)}")

BlockType = t.Union[at.TextBlockParam, at.ImageBlockParam, at.ToolUseBlockParam, at.ToolResultBlockParam, at.ContentBlock]

def join_content(a: t.Union[t.List[t.Union[at.TextBlockParam, at.ImageBlockParam]], str], b: t.Union[t.List[t.Union[at.TextBlockParam, at.ImageBlockParam]], str]):
    if isinstance(a, str):
        a = [str_to_text_block(a)]
    if isinstance(b, str):
        b = [str_to_text_block(b)]
    if a[0]["type"] == "text" and b[0]["type"] == "text":
        return [at.TextBlockParam({
            "type": "text",
            "text": a[0]["text"] + b[0]["text"]
        })]
    return a + b

def merge_messages(messages: t.List[at.MessageParam]):
    merged: t.List[at.MessageParam] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            last_content = merged[-1]["content"]
            merged[-1]["content"] = join_content(last_content, message["content"]) # type: ignore
        else:
            merged.append(message)
    return merged

def deltas_to_messages(deltas: t.Iterable[t.Union[Query, Delta]]):
    messages = []
    for delta in deltas:
        if isinstance(delta, Delta):
            if delta.content["type"] == "text":
                messages.append(at.MessageParam(
                    role="assistant",
                    content=[str_to_text_block(delta.content["data"])]
                ))
            elif delta.content["type"] == "tool_use":
                messages.append(at.MessageParam(
                    role="assistant",
                    content=[at.ToolUseBlockParam(
                        type="tool_use",
                        name=delta.content["data"].name,
                        input=delta.content["data"].input,
                        id=delta.content["data"].id
                    )]
                ))
            elif delta.content["type"] == "tool_result":
                messages.append(at.MessageParam(
                    role="user",
                    content=[at.ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=delta.content["data"].id,
                        content=tool_result_to_content(delta.content["data"]),
                    )]
                ))
            else:
                raise ValueError(f"Invalid delta type: {delta.content['type']}")
        else:
            messages.extend(query_to_content(delta))

    return merge_messages(messages)
                
class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]

STREAM_ARGS = StreamArgs
MODEL_TYPE = at.ModelParam


class AnthropicLLM(LLMClient[STREAM_ARGS, at.ModelParam, t.Union[AsyncAnthropicBedrock, AsyncAnthropic]]):
    def __init__(self, client: t.Optional[t.Union[AsyncAnthropicBedrock, AsyncAnthropic]]=None):
        if not client:
            client = AsyncAnthropic()
        self.client = client

    async def stream(self,
        **kwargs: t.Unpack[LLMClientStreamArgs[STREAM_ARGS, MODEL_TYPE]]
        ):
        extra_kwargs: dict[str, t.Any] = {
            **(kwargs or {})
        }
        if functions := kwargs.get("functions", None):
            extra_kwargs["tools"] = functions_for_llm(functions)
            if function_choice := kwargs.get("function_choice", None):
                extra_kwargs["tool_choice"] = {
                    "type": function_choice
                }
        if prompt := kwargs.get("prompt", None):
            extra_kwargs["system"] = prompt

        
        messages = deltas_to_messages(kwargs.get("history"))
        async with self.client.messages.stream(
            max_tokens=kwargs.get("max_tokens", None) or 4096,
            model=kwargs.get("model"),
            messages=messages,
            top_k=kwargs.get("top_k", None) or NotGiven(),
            top_p=kwargs.get("top_p", None) or NotGiven(),
            stop_sequences=kwargs.get("stop_sequences", None) or NotGiven(),
            temperature=kwargs.get("temperature", None) or NotGiven(),
            **extra_kwargs
        ) as response:
            function_blocks: t.List[at.ToolUseBlock] = []
            async for event in response:
                if event.type == "text":
                    yield Delta(content={"type": "text", "data": event.text})
                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        function_blocks.append(event.content_block)
                
                elif event.type == "message_stop":
                    usage: Usage = {
                        "input_tokens": event.message.usage.input_tokens, 
                        "output_tokens": event.message.usage.output_tokens
                    }
                    yield Delta(content={
                            "type": "text",
                            "data": "",
                        },
                        usage=usage
                    )
                    if event.message.stop_reason == "tool_use":
                        for block in function_blocks:
                            tool_use = ToolUse(id=block.id, name=block.name, input=block.input) # type: ignore
                            yield Delta(content={
                                "data": tool_use,
                                "type": "tool_use"
                            })
