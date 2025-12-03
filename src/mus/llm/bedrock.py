import typing as t
from .types import LLM, Delta, DeltaText, ToolUse, ToolResult, File, Query, Usage, Assistant, LLMClientStreamArgs, is_tool_simple_return_value, FunctionSchemaNoAnnotations, DeltaToolUse, DeltaToolResult, DeltaToolInputUpdate, DeltaHistory
import base64

from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
from types_aiobotocore_bedrock_runtime import type_defs as bt
import json
import aioboto3
import aiobotocore.session


def func_schema_to_tool(func_schema: FunctionSchemaNoAnnotations):
    return bt.ToolTypeDef(
        toolSpec=bt.ToolSpecificationTypeDef(
            name=func_schema["name"],
            description=func_schema["description"],
            inputSchema=bt.ToolInputSchemaTypeDef(
                json=func_schema["schema"]
            )
        )
    )

def functions_for_llm(functions: t.Sequence[FunctionSchemaNoAnnotations]):
    return [
        func_schema_to_tool(func)
        for func
        in (functions or [])
    ]

def file_to_image(file: File):
    try:
        type_, subtype = file.b64type.split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid b64type: {file.b64type}, must be in format type/subtype")
    if type_ != "image":
        raise ValueError(f"Only supports image/[type], not: {file.b64type}")
    elif subtype not in ["png", "jpeg"]:
        raise ValueError(f"Only supports image/png and image/jpeg, not: {file.b64type}")
    # now we know subtype is either png or jpeg, so we can use Literal
    subtype = t.cast(t.Literal["png", "jpeg"], subtype)

    return bt.ContentBlockTypeDef(
        image=bt.ImageBlockTypeDef(
            format=subtype,
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

def query_to_messages(query: Query):
    for q in query.val:
        if isinstance(q, Assistant):
            yield bt.MessageTypeDef(
                role="assistant",
                content=[{
                    "text": q.val
                }]
            )
        else:
            yield bt.MessageTypeDef(
                role="user",
                content=[parse_content(q)]
            )

def parse_tool_content(c: t.Union[str, File]):
    if isinstance(c, str):
        return bt.ToolResultContentBlockOutputTypeDef({
            "text": c
        })
    elif isinstance(c, File):
        img = file_to_image(c)
        if "image" in img:
            return bt.ToolResultContentBlockTypeDef(
                image=img["image"]
            )
        else:
            raise ValueError("No image found")
    else:
        raise ValueError(f"Invalid tool result type: {type(c)}")

def tool_result_to_content(tool_result: ToolResult):
    if is_tool_simple_return_value(tool_result.content.val):
        return [
            parse_tool_content(tool_result.content.val)
        ]
    elif isinstance(tool_result.content.val, list):
        return [
            parse_tool_content(c)
            for c in tool_result.content.val
        ]
    else:
        raise ValueError(f"Invalid tool result type: {type(tool_result.content.val)}")
    
def has_reasoning_text(content: t.Union[bt.ContentBlockTypeDef, str]):
    if isinstance(content, str):
        return False
    elif "reasoningContent" in content and "reasoningText" in content["reasoningContent"]:
        return True
    return False

def join_content(a: t.Union[t.List[bt.ContentBlockTypeDef], str], b: t.Union[t.List[bt.ContentBlockTypeDef], str]):
    if isinstance(a, str):
        a = [str_to_text_block(a)]
    if isinstance(b, str):
        b = [str_to_text_block(b)]
    if "text" in a[-1] and "text" in b[0]:
        return a[:-1] + [bt.ContentBlockTypeDef({
            "text": a[-1]["text"] + b[0]["text"]
        })]
    if has_reasoning_text(a[-1]) and has_reasoning_text(b[0]):
        return a[:-1] + [bt.ContentBlockTypeDef({
            "reasoningContent": bt.ReasoningContentBlockTypeDef({
                "reasoningText": bt.ReasoningTextBlockTypeDef({
                    "text": a[-1]["reasoningContent"]["reasoningText"]["text"] + b[0]["reasoningContent"]["reasoningText"]["text"], # type: ignore
                    "signature": a[-1]["reasoningContent"]["reasoningText"]["signature"] or b[0]["reasoningContent"]["reasoningText"]["signature"], # type: ignore
                })
            })
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
            if isinstance(delta.content, DeltaText) and delta.content.subtype == "reasoning":
                metadata = delta.content.metadata or {}
                reasoning_content = bt.ReasoningContentBlockTypeDef()
                if not metadata.get("redactedContent", False):
                    reasoning_content["reasoningText"] = bt.ReasoningTextBlockTypeDef({
                        "text": delta.content.data,
                        "signature": str(metadata.get("signature", None))
                    })
                if redacted := metadata.get("redactedContent"):
                    reasoning_content["redactedContent"] = redacted  # type: ignore[typeddict-item]
                messages.append(bt.MessageTypeDef(
                    role="assistant",
                    content=[
                        bt.ContentBlockTypeDef(
                            reasoningContent=reasoning_content
                        )
                    ]
                ))
            elif isinstance(delta.content, DeltaText):
                if delta.content.data:
                    messages.append(bt.MessageTypeDef(
                        role="assistant",
                        content=[str_to_text_block(delta.content.data)]
                    ))
            elif isinstance(delta.content, DeltaToolUse):
                messages.append(bt.MessageTypeDef(
                    role="assistant",
                    content=[
                        bt.ContentBlockTypeDef(
                            toolUse=bt.ToolUseBlockTypeDef({
                                "name": delta.content.data.name,
                                "toolUseId": delta.content.data.id,
                                "input": delta.content.data.input,
                            }),
                        )
                    ]
                ))
            elif isinstance(delta.content, DeltaToolResult):
                messages.append(bt.MessageTypeDef(
                    role="user",
                    content=[
                        bt.ContentBlockTypeDef(
                            toolResult=bt.ToolResultBlockTypeDef(
                                toolUseId=delta.content.data.id,
                                content=tool_result_to_content(delta.content.data),
                            )
                        ),
                        # example of how to add a cache point, if needed
                        #bt.ContentBlockTypeDef(
                        #    cachePoint= bt.CachePointBlockTypeDef(
                        #        type="default"
                        #    )
                        #)
                    ]
                ))
            elif isinstance(delta.content, (DeltaToolInputUpdate, DeltaHistory)):
                pass
            else:
                raise t.assert_never(delta.content)
        else:
            messages.extend(query_to_messages(delta))

    return merge_messages(messages)

class reasoning_config(t.TypedDict):
    type: t.Literal["enabled"]
    budget_tokens: int

class additionalModelRequestFields(t.TypedDict):
    reasoning_config: t.Optional[reasoning_config]

class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]
    additionalModelRequestFields: t.Optional[additionalModelRequestFields]

STREAM_ARGS = StreamArgs
MODEL_TYPE = str
ALL_STREAM_ARGS = t.Union[StreamArgs]

class BedrockLLM(LLM[StreamArgs, MODEL_TYPE, BedrockRuntimeClient]):
    def __init__(self, model: MODEL_TYPE, client: t.Optional[BedrockRuntimeClient]=None, session: t.Optional[aiobotocore.session.AioSession]=None):
        self.client = client
        self.session = session if session is not None else aiobotocore.session.get_session()
        self.model = model

    async def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[StreamArgs, MODEL_TYPE]]):
        extra_kwargs: dict[str, t.Any] = {
            **(kwargs.get("kwargs", None) or {})
        }
        cache_config = kwargs.get("cache", None)
        if functions := kwargs.get("functions", None):
            toolConfig: bt.ToolConfigurationTypeDef = {
                "tools": functions_for_llm(functions)
                #**({"toolChoice": {function_choice: {}}} if function_choice else {})
            }
            if cache_config and cache_config.get("cache_tools", True):
                toolConfig["tools"] = [
                    *toolConfig["tools"],
                    {
                        "cachePoint": {
                            "type": "default"
                        }
                    }
                ]
            extra_kwargs["toolConfig"] = toolConfig

        if prompt := kwargs.get("prompt", None):
            system: list[bt.SystemContentBlockTypeDef] = [
                {
                    "text": prompt,
                }
            ]
            if cache_config and cache_config.get("cache_system_prompt", True):
                system.append({
                    "cachePoint": {
                        "type": "default"
                    }
                })
            extra_kwargs["system"] = system


        messages = deltas_to_messages(kwargs.get("history"))

        args = bt.ConverseStreamRequestTypeDef(
            modelId=str(self.model),
            messages=messages,
            inferenceConfig={
                "maxTokens": kwargs.get("max_tokens") or 4096,
                **({"temperature": kwargs.get("temperature")} if kwargs.get("temperature", None) else {}), # type: ignore
                **({"topP": kwargs.get("top_p")} if kwargs.get("top_p", None) else {}),
                **({"stopSequences": kwargs.get("stop_sequences")} if kwargs.get("stop_sequences", None) else {}),
            },
            **extra_kwargs
        )

        # Use provided client or create one from session
        if self.client is not None:
            client = self.client
            if not kwargs.get("no_stream", False):
                response = await client.converse_stream(**args)
                function_blocks: t.List[bt.ToolUseBlockTypeDef] = []
                current_function = None
                async for event in response.get("stream"):
                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"]["start"]
                        if "toolUse" in start:
                            current_function = {**start["toolUse"]}
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            yield Delta(content=DeltaText(data=delta["text"]))
                        if "reasoningContent" in delta:
                            reasoning = delta["reasoningContent"]

                            yield Delta(content=DeltaText(
                                data=reasoning.get("text", "")
                                , subtype="reasoning",
                                metadata={
                                    "signature": reasoning.get("signature", None),
                                    "redactedContent": reasoning.get("redactedContent", None)
                                }
                            ))
                        if "toolUse" in delta:
                            tu = delta["toolUse"]
                            if current_function:
                                yield Delta(content=DeltaToolInputUpdate(
                                    name=current_function["name"],
                                    id=current_function["toolUseId"],
                                    data=tu["input"]
                                ))
                                current_function["input"] = current_function.get("input", "") + tu["input"]
                    if "contentBlockStop" in event:
                        if current_function and current_function.get("input", None):
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
                                yield Delta(content=DeltaToolUse(data=tool_use))

                    if "metadata" in event:
                        metadata = event["metadata"]
                        usage = Usage(
                            input_tokens=metadata["usage"]["inputTokens"],
                            output_tokens=metadata["usage"]["outputTokens"],
                            cache_read_input_tokens=metadata["usage"].get("cacheReadInputTokens", 0),
                            cache_written_input_tokens=metadata["usage"].get("cacheWriteInputTokens", 0)
                        )
                        yield Delta(
                            content=DeltaText(
                                data="",
                            ),
                            usage=usage
                        )
            else:
                response = await client.converse(**args)

                output = response.get("output")
                tools: t.List[Delta] = []
                if "message" in output:
                    message = output["message"]
                    for content in message["content"]:
                        if text := content.get("text", None):
                            yield Delta(content=DeltaText(
                                data=text,
                            ))
                        if toolUse := content.get("toolUse", None):
                            tool_use = ToolUse(id=toolUse["toolUseId"], name=toolUse["name"], input=toolUse["input"])
                            tools.append(Delta(content=DeltaToolUse(
                                data=tool_use
                            )))

                if response["stopReason"] == "tool_use":
                    for tool in tools:
                        yield tool

                usage = Usage(
                    input_tokens=response["usage"]["inputTokens"],
                    output_tokens=response["usage"]["outputTokens"],
                    cache_read_input_tokens=response["usage"].get("cacheReadInputTokens", 0),
                    cache_written_input_tokens=response["usage"].get("cacheWrittenInputTokens", 0)
                )
                yield Delta(content=DeltaText(
                    data="",
                ), usage=usage)
        else:
            # Create client from session
            async with self.session.create_client("bedrock-runtime") as client:
                if not kwargs.get("no_stream", False):
                    response = await client.converse_stream(**args)
                    function_blocks: t.List[bt.ToolUseBlockTypeDef] = []
                    current_function = None
                    async for event in response.get("stream"):
                        if "contentBlockStart" in event:
                            start = event["contentBlockStart"]["start"]
                            if "toolUse" in start:
                                current_function = {**start["toolUse"]}
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"]["delta"]
                            if "text" in delta:
                                yield Delta(content=DeltaText(data=delta["text"]))
                            if "reasoningContent" in delta:
                                reasoning = delta["reasoningContent"]

                                yield Delta(content=DeltaText(
                                    data=reasoning.get("text", "")
                                    , subtype="reasoning",
                                    metadata={
                                        "signature": reasoning.get("signature", None),
                                        "redactedContent": reasoning.get("redactedContent", None)
                                    }
                                ))
                            if "toolUse" in delta:
                                tu = delta["toolUse"]
                                if current_function:
                                    yield Delta(content=DeltaToolInputUpdate(
                                        name=current_function["name"],
                                        id=current_function["toolUseId"],
                                        data=tu["input"]
                                    ))
                                    current_function["input"] = current_function.get("input", "") + tu["input"]
                        if "contentBlockStop" in event:
                            if current_function and current_function.get("input", None):
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
                                    yield Delta(content=DeltaToolUse(data=tool_use))

                        if "metadata" in event:
                            metadata = event["metadata"]
                            usage = Usage(
                                input_tokens=metadata["usage"]["inputTokens"],
                                output_tokens=metadata["usage"]["outputTokens"],
                                cache_read_input_tokens=metadata["usage"].get("cacheReadInputTokens", 0),
                                cache_written_input_tokens=metadata["usage"].get("cacheWriteInputTokens", 0)
                            )
                            yield Delta(
                                content=DeltaText(
                                    data="",
                                ),
                                usage=usage
                            )
                else:
                    response = await client.converse(**args)

                    output = response.get("output")
                    tools: t.List[Delta] = []
                    if "message" in output:
                        message = output["message"]
                        for content in message["content"]:
                            if text := content.get("text", None):
                                yield Delta(content=DeltaText(
                                    data=text,
                                ))
                            if toolUse := content.get("toolUse", None):
                                tool_use = ToolUse(id=toolUse["toolUseId"], name=toolUse["name"], input=toolUse["input"])
                                tools.append(Delta(content=DeltaToolUse(
                                    data=tool_use
                                )))

                    if response["stopReason"] == "tool_use":
                        for tool in tools:
                            yield tool

                    usage = Usage(
                        input_tokens=response["usage"]["inputTokens"],
                        output_tokens=response["usage"]["outputTokens"],
                        cache_read_input_tokens=response["usage"].get("cacheReadInputTokens", 0),
                        cache_written_input_tokens=response["usage"].get("cacheWrittenInputTokens", 0)
                    )
                    yield Delta(content=DeltaText(
                        data="",
                    ), usage=usage)