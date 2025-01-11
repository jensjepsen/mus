import typing as t
from dataclasses import is_dataclass
from .types import LLMClient, Delta, ToolUse, ToolResult, File, ToolCallableType, Query, Usage
from ..functions import get_schema
import base64

from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime import type_defs as bt
import json

def func_to_tool(func: ToolCallableType):
    if hasattr(func, '__metadata__'):
        if definition := func.__metadata__.get("definition"):
            return definition
    if not func.__doc__:
        raise ValueError(f"Function {func.__name__} is missing a docstring")
    p = bt.ToolTypeDef(
        toolSpec=bt.ToolSpecificationTypeDef(
            name=func.__name__,
            description=func.__doc__,
            inputSchema=bt.ToolInputSchemaTypeDef(
                json=get_schema(func.__name__, list(func.__annotations__.items()))
            )
        )
    )
    return p

def dataclass_to_tool(dataclass):
    p = bt.ToolTypeDef(
        toolSpec=bt.ToolSpecificationTypeDef(
                name=dataclass.__name__,
                description=dataclass.__doc__,
                inputSchema=bt.ToolInputSchemaTypeDef(
                    json=get_schema(dataclass.__name__, list(dataclass.__annotations__.items()))
                )
            )
    )
    return p

def functions_for_llm(functions: t.List[ToolCallableType]):
    return [
        dataclass_to_tool(func) if is_dataclass(func) else func_to_tool(func)
        for func
        in (functions or [])
    ]

def file_to_image(file: File):
    return bt.ContentBlockTypeDef(
        image=bt.ImageBlockTypeDef(
            format=file.b64type.split("/", 1)[-1],
            source=bt.ImageSourceTypeDef(
                bytes=base64.b64decode(file.content)
            )
        )
    )

def str_to_text_block(s: str):
    return bt.ContentBlockTypeDef(
        text=s
    )

def parse_content(query: t.Union[str, File]):
    if isinstance(query, str):
        return str_to_text_block(query)
    elif isinstance(query, File):
        return file_to_image(query)
    else:
        raise ValueError(f"Invalid query type: {type(query)}")

def query_to_content(query: Query):
    return [
        parse_content(q)
        for q in query.val
    ]

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

def join_content(a: t.Union[t.List[bt.ContentBlockTypeDef], str], b: t.Union[t.List[bt.ContentBlockTypeDef], str]):
    if isinstance(a, str):
        a = [str_to_text_block(a)]
    if isinstance(b, str):
        b = [str_to_text_block(b)]
    if "text" in a[0] and "text" in b[0]:
        return [bt.ContentBlockTypeDef({
            "text": a[0]["text"] + b[0]["text"]
        })]
    return a + b

def merge_messages(messages: t.List[bt.MessageTypeDef]):
    merged: t.List[bt.MessageTypeDef] = []
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
                messages.append(bt.MessageTypeDef(
                    role="assistant",
                    content=[str_to_text_block(delta.content["data"])]
                ))
            elif delta.content["type"] == "tool_use":
                
                messages.append(bt.MessageTypeDef(
                    role="assistant",
                    content=[
                        bt.ContentBlockTypeDef(
                            toolUse=bt.ToolUseBlockTypeDef({
                                "name": delta.content["data"].name,
                                "toolUseId": delta.content["data"].id,
                                "input": delta.content["data"].input,
                            }),
                        )
                    ]
                ))
            elif delta.content["type"] == "tool_result":
                messages.append(bt.MessageTypeDef(
                    role="user",
                    content=[
                        bt.ContentBlockTypeDef(
                            toolResult=bt.ToolResultBlockTypeDef(
                                toolUseId=delta.content["data"].id,
                                content=tool_result_to_content(delta.content["data"]),
                            )
                        )
                    ]
                ))
            else:
                raise ValueError(f"Invalid delta type: {delta.content['type']}")
        else:
            messages.append(bt.MessageTypeDef(
                role="user",
                content=query_to_content(delta)
            ))

    return merge_messages(messages)
                
class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]


class BedrockLLM(LLMClient[StreamArgs, str]):
    def __init__(self, client: BedrockRuntimeClient):
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
            kwargs: t.Optional[StreamArgs]=None,
            no_stream: t.Optional[bool]=None
        ):
        _kwargs: dict[str, t.Any] = {
            **(kwargs or {})
        }
        if functions:
            _kwargs["toolConfig"] = {
                "tools": functions_for_llm(functions),
                #**({"toolChoice": {function_choice: {}}} if function_choice else {})
            }
            
        if prompt:
            _kwargs["system"] = [{
                "text": prompt
            }]

        
        messages = deltas_to_messages(history)
        """
        top_k=top_k or NotGiven(),
            top_p=top_p or NotGiven(),
            temperature=temperature or NotGiven(),
        """
        args = dict(
            modelId=model,
            messages=messages,
            inferenceConfig={ 
                "maxTokens": max_tokens or 4096,
                **({"temperature": temperature} if temperature else {}), # type: ignore
                **({"topP": top_p} if top_p else {})
            },
            **_kwargs
        )
        if not no_stream:
            response = self.client.converse_stream(
                **args
            )
            function_blocks: t.List[bt.ToolUseBlockTypeDef] = []
            current_function = None
            for event in response.get("stream"):
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]["start"]
                    if "toolUse" in start:
                        current_function = {**start["toolUse"]}
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        yield Delta(content={
                            "type": "text",
                            "data": delta["text"]
                        })
                    if "toolUse" in delta:
                        tu = delta["toolUse"]
                        if current_function:
                            current_function["input"] = current_function.get("input", "") + tu["input"]
                if "contentBlockStop" in event:
                    if current_function:
                        function_blocks.append(
                            bt.ToolUseBlockTypeDef(
                                **{
                                    **current_function,
                                    **dict(input=json.loads(current_function["input"]))
                                }
                            )
                        )
                        current_function = None

                if "messageStop" in event:
                    stop_reason = event["messageStop"]["stopReason"]
                    if stop_reason == "tool_use":
                        for block in function_blocks:
                            tool_use = ToolUse(id=block["toolUseId"], name=block["name"], input=block["input"])
                            yield Delta(content={
                                "data": tool_use,
                                "type": "tool_use"
                            })

                if "metadata" in event:
                    metadata = event["metadata"]
                    usage: Usage = {
                        "input_tokens": metadata["usage"]["inputTokens"], 
                        "output_tokens": metadata["usage"]["outputTokens"]
                    }
                    yield Delta(content={
                            "type": "text",
                            "data": "",
                        },
                        usage=usage
                    )
        else:
            response = self.client.converse(
                **args
            )

            output = response.get("output")
            tools: t.List[Delta] = []
            if "message" in output:
                message = output["message"]
                for content in message["content"]:
                    if text := content.get("text", None):
                        yield Delta(content={
                            "type": "text",
                            "data": text
                        })
                    if toolUse := content.get("toolUse", None):
                        tool_use = ToolUse(id=toolUse["toolUseId"], name=toolUse["name"], input=toolUse["input"])
                        tools.append(Delta(content={
                            "data": tool_use,
                            "type": "tool_use"
                        }))
                
            if response["stopReason"] == "tool_use":
                for tool in tools:
                    yield tool
            
            usage: Usage = {
                "input_tokens": response["usage"]["inputTokens"], 
                "output_tokens": response["usage"]["outputTokens"]
            }
            yield Delta(content={
                    "type": "text",
                    "data": "",
                },
                usage=usage
            )
            
