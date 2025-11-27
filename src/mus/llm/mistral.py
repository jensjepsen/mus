import typing as t
from .types import (
    LLM, Delta, DeltaHistory, DeltaText, DeltaToolInputUpdate, ToolUse, ToolResult, File, Query, Usage, Assistant, LLMClientStreamArgs, is_tool_simple_return_value, FunctionSchemaNoAnnotations,
    DeltaToolUse, DeltaToolResult
)
import json

from mistralai import Mistral
from mistralai.models import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ImageURLChunk,
    TextChunk,
    Tool,
    ToolCall,
    FunctionCall,
    ImageURL,
    Messages,
    Arguments,
    Function
)
import mistralai.models as mt

P = t.ParamSpec("P")
T = t.TypeVar("T")


def func_schema_to_tool(func_schema: FunctionSchemaNoAnnotations) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=func_schema["name"],
            description=func_schema["description"],
            parameters=func_schema["schema"]
        )
    )


def functions_for_llm(functions: t.Sequence[FunctionSchemaNoAnnotations]) -> t.List[Tool]:
    return [
        func_schema_to_tool(func)
        for func in (functions or [])
    ]


def file_to_image_chunk(file: File) -> ImageURLChunk:
    try:
        type_, subtype = file.b64type.split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid b64type: {file.b64type}, must be in format type/subtype")
    if type_ != "image":
        raise ValueError(f"Only supports image/[type], not: {file.b64type}")
    elif subtype not in ["png", "jpeg", "jpg", "gif", "webp"]:
        raise ValueError(f"Unsupported image format: {file.b64type}")
    
    # Create data URL from base64 content
    data_url = f"data:{file.b64type};base64,{file.content}"
    
    return ImageURLChunk(
        type="image_url",
        image_url=ImageURL(url=data_url)
    )


def str_to_text_chunk(s: str) -> TextChunk:
    return TextChunk(
        type="text",
        text=s
    )


def parse_content(query: t.Union[str, File]) -> t.Union[TextChunk, ImageURLChunk]:
    if isinstance(query, str):
        return str_to_text_chunk(query)
    elif isinstance(query, File):
        return file_to_image_chunk(query)
    else:
        raise ValueError(f"Invalid query type: {type(query)}")


def query_to_messages(query: Query) -> t.List[Messages]:
    messages = []
    for q in query.val:
        if isinstance(q, Assistant):
            messages.append(AssistantMessage(
                role="assistant",
                content=q.val,
            ))
        else:
            content = parse_content(q)
            if isinstance(content, TextChunk):
                messages.append(UserMessage(
                    role="user",
                    content=content.text
                ))
            else:
                # For images, we need to send as content array
                messages.append(UserMessage(
                    role="user",
                    content=[content]
                ))
    return messages


def parse_tool_content(c: t.Union[str, File]) -> str:
    if isinstance(c, str):
        return c
    elif isinstance(c, File):
        # For tool results, Mistral typically expects text descriptions of images
        # You might want to implement image-to-text conversion here
        return f"[Image: {c.b64type}]"
    else:
        raise ValueError(f"Invalid tool result type: {type(c)}")


def tool_result_to_content(tool_result: ToolResult) -> str:
    if is_tool_simple_return_value(tool_result.content.val):
        return parse_tool_content(tool_result.content.val)
    elif isinstance(tool_result.content.val, list):
        return "\n".join([
            parse_tool_content(c)
            for c in tool_result.content.val
        ])
    else:
        raise ValueError(f"Invalid tool result type: {type(tool_result.content.val)}")


def merge_messages(messages: t.List[Messages]) -> t.List[Messages]:
    merged: t.List[Messages] = []
    for message in messages:
        if (merged and 
            merged[-1].role == message.role and 
            isinstance(merged[-1], (UserMessage, AssistantMessage)) and
            isinstance(message, (UserMessage, AssistantMessage)) and
            isinstance(merged[-1].content, str) and
            isinstance(message.content, str)):
            # Merge consecutive messages from same role
            merged[-1].content = merged[-1].content + message.content
        else:
            merged.append(message)
    return merged


def deltas_to_messages(deltas: t.Iterable[t.Union[Query, Delta]]) -> t.List[Messages]:
    messages = []
    tool_id_to_name: t.Dict[str, str] = {}
    for delta in deltas:
        if isinstance(delta, Delta):
            if isinstance(delta.content, DeltaText):
                if delta.content.data:
                    messages.append(AssistantMessage(
                        role="assistant",
                        content=delta.content.data
                    ))
            elif isinstance(delta.content, DeltaToolUse):
                tool_use = delta.content.data
                tool_id_to_name[tool_use.id] = tool_use.name
                messages.append(AssistantMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=tool_use.id,
                            type="function",
                            function=FunctionCall(
                                name=tool_use.name,
                                arguments=json.dumps(tool_use.input)
                            )
                        )
                    ]
                ))
            elif isinstance(delta.content, DeltaToolResult):
                tool_result = delta.content.data
                tool_name = tool_id_to_name.get(tool_result.id, "unknown_tool")
                messages.append(ToolMessage(
                    role="tool",
                    tool_call_id=tool_result.id, 
                    name=tool_name,
                    content=tool_result_to_content(tool_result)
                ))
            elif isinstance(delta.content, (DeltaToolInputUpdate, DeltaHistory)):
                pass
            else:
                t.assert_never(delta.content)
        else:
            messages.extend(query_to_messages(delta))

    return merge_messages(messages)


class StreamArgs(t.TypedDict, total=False):
    extra_headers: t.Dict[str, str]
    safe_prompt: t.Optional[bool]


STREAM_ARGS = StreamArgs
MODEL_TYPE = str
ALL_STREAM_ARGS = t.Union[StreamArgs]


def convert_tool_arguments(args: Arguments):
    if isinstance(args, dict):
        return args
    elif isinstance(args, str):
        return json.loads(args)
    else:
        raise ValueError(f"Invalid arguments type: {type(args)}")

async def choice_content_to_chunks(content: mt.Content):
    if isinstance(content, str):
        yield Delta(content=DeltaText(data=content))
    else:
        for content_chunk in content:
            if isinstance(content_chunk, TextChunk):
                yield Delta(content=DeltaText(data=content_chunk.text))
            elif isinstance(content_chunk, str):
                yield Delta(content=DeltaText(data=content_chunk))
            else:
                raise ValueError(f"Unsupported content type: {type(content_chunk)}")

class MistralLLM(LLM[StreamArgs, MODEL_TYPE, Mistral]):
    def __init__(self, model: MODEL_TYPE, client: t.Optional[Mistral] = None, api_key: t.Optional[str] = None):
        if not client:
            client = Mistral(api_key=api_key)
        self.client = client
        self.model = model

    async def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[StreamArgs, MODEL_TYPE]]):
        extra_kwargs: dict[str, t.Any] = {
            **(kwargs.get("kwargs", None) or {})
        }
        
        messages = deltas_to_messages(kwargs.get("history", []))
        
        # Add system message if prompt is provided
        if prompt := kwargs.get("prompt", None):
            messages.insert(0, SystemMessage(
                role="system",
                content=prompt
            ))
        
        # Prepare tools
        tools = None
        #tool_choice = None
        if functions := kwargs.get("functions", None):
            tools = functions_for_llm(functions)
        
        request = ChatCompletionRequest(
            model=str(self.model),
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stop=kwargs.get("stop_sequences"),
            tools=tools,
            stream=not kwargs.get("no_stream", False),
            safe_prompt=extra_kwargs.get("safe_prompt"),
            **{k: v for k, v in extra_kwargs.items() if k != "safe_prompt"}
        )
        
        if not kwargs.get("no_stream", False):
            response = await self.client.chat.stream_async(**request.model_dump(exclude_none=True))
            
            async for chunk in response:
                if chunk.data and chunk.data.choices:
                    choice = chunk.data.choices[0]
                    
                    if choice.delta.content:
                        async for content_chunk in choice_content_to_chunks(choice.delta.content):
                            yield content_chunk
                        
                    if choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.function:
                                tool_use = ToolUse(
                                    id=tool_call.id or tool_call.function.name,
                                    name=tool_call.function.name,
                                    input=convert_tool_arguments(tool_call.function.arguments) if tool_call.function.arguments else {}
                                )
                                yield Delta(content=DeltaToolUse(data=tool_use))
                                    
                    
                    if choice.finish_reason:
                        # Handle usage information if available
                        if chunk.data and chunk.data.usage:
                            usage = Usage(
                                input_tokens=chunk.data.usage.prompt_tokens or 0,
                                output_tokens=chunk.data.usage.completion_tokens or 0,
                                cache_read_input_tokens=0,  # Mistral doesn't provide cache info
                                cache_written_input_tokens=0
                            )
                            yield Delta(
                                content=DeltaText(data=""),
                                usage=usage
                            )
        else:
            response = await self.client.chat.complete_async(**request.model_dump(exclude_none=True))
            
            if response.choices:
                choice = response.choices[0]
                
                if choice.message.content:
                    async for delta in choice_content_to_chunks(choice.message.content):
                        yield delta
                
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function:
                            tool_use = ToolUse(
                                id=tool_call.id or tool_call.function.name,
                                name=tool_call.function.name,
                                input=convert_tool_arguments(tool_call.function.arguments) if tool_call.function.arguments else {}
                            )
                            yield Delta(content=DeltaToolUse(data=tool_use))
                
                # Handle usage information
                if response.usage:
                    usage = Usage(
                        input_tokens=response.usage.prompt_tokens or 0,
                        output_tokens=response.usage.completion_tokens or 0,
                        cache_read_input_tokens=0,  # Mistral doesn't provide cache info
                        cache_written_input_tokens=0
                    )
                    yield Delta(
                        content=DeltaText(data=""),
                        usage=usage
                    )