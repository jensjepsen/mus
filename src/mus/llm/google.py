import typing as t
from .types import LLM, Delta, DeltaHistory, DeltaToolInputUpdate, ToolUse, ToolResult, File, Query, Usage, Assistant, LLMClientStreamArgs, is_tool_simple_return_value, FunctionSchemaNoAnnotations, DeltaText, DeltaToolUse, DeltaToolResult
import base64
import json

from google import genai
from google.genai import types as genai_types

def func_schema_to_tool(func_schema: FunctionSchemaNoAnnotations):
    return genai_types.FunctionDeclaration(
        name=func_schema["name"],
        description=func_schema["description"],
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                k: genai_types.Schema(**v) if isinstance(v, dict) else v
                for k, v in func_schema["schema"].get("properties", {}).items()
            },
            required=func_schema["schema"].get("required", [])
        )
    )

def functions_for_llm(functions: t.Sequence[FunctionSchemaNoAnnotations]):
    return [
        genai_types.Tool(function_declarations=[func_schema_to_tool(func)])
        for func in (functions or [])
    ]

def file_to_part(file: File):
    try:
        type_, subtype = file.b64type.split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid b64type: {file.b64type}, must be in format type/subtype")
    
    if type_ == "image":
        if subtype not in ["png", "jpeg", "jpg", "gif", "webp"]:
            raise ValueError(f"Unsupported image type: {file.b64type}")
        return genai_types.Part.from_bytes(
            data=base64.b64decode(file.content),
            mime_type=file.b64type
        )
    else:
        # For other file types, try to handle as generic data
        return genai_types.Part.from_bytes(
            data=base64.b64decode(file.content),
            mime_type=file.b64type
        )

def parse_content(query: t.Union[str, File]):
    if isinstance(query, str):
        return genai_types.Part.from_text(text=query)
    elif isinstance(query, File):
        return file_to_part(query)
    else:
        raise ValueError(f"Invalid query type: {type(query)}")

def query_to_contents(query: Query):
    contents = []
    current_parts = []
    current_role = None
    
    for q in query.val:
        if isinstance(q, Assistant):
            # Flush any pending user parts
            if current_parts and current_role == "user":
                contents.append(genai_types.Content(role="user", parts=current_parts))
                current_parts = []
            
            # Add assistant content
            contents.append(genai_types.Content(
                role="model", 
                parts=[genai_types.Part.from_text(text=q.val)]
            ))
            current_role = "model"
        else:
            # If we were processing model content, flush it
            if current_role == "model":
                current_parts = []
            
            current_parts.append(parse_content(q))
            current_role = "user"
    
    # Flush any remaining parts
    if current_parts:
        contents.append(genai_types.Content(role=current_role or "user", parts=current_parts))
    
    return contents

def tool_result_to_parts(tool_result: ToolResult):
    if is_tool_simple_return_value(tool_result.content):
        if isinstance(tool_result.content, str):
            return [genai_types.Part.from_text(text=tool_result.content)]
        elif isinstance(tool_result.content, File):
            return [file_to_part(tool_result.content)]
        else:
            return [genai_types.Part.from_text(text=str(tool_result.content))]
    elif isinstance(tool_result.content, list):
        parts = []
        for c in tool_result.content:
            if isinstance(c, str):
                parts.append(genai_types.Part.from_text(text=c))
            elif isinstance(c, File):
                parts.append(file_to_part(c))
            else:
                parts.append(genai_types.Part.from_text(text=str(c)))
        return parts
    else:
        raise ValueError(f"Invalid tool result type: {type(tool_result.content)}")

def deltas_to_contents(deltas: t.Iterable[t.Union[Query, Delta]]):
    contents = []
    tool_id_to_name = {}
    for delta in deltas:
        if isinstance(delta, Delta):
            if isinstance(delta.content, DeltaText):
                if delta.content.data:
                    contents.append(genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(
                            text=delta.content.data,
                            thought_signature=delta.metadata.get("thought_signature") if delta.metadata else None
                        )]
                    ))
            elif isinstance(delta.content, DeltaToolUse):
                tool_use = delta.content.data
                tool_id_to_name[tool_use.id] = tool_use.name
                contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            id=tool_use.id,
                            name=tool_use.name,
                            args=tool_use.input # type: ignore
                        ),
                        thought_signature=delta.metadata.get("thought_signature") if delta.metadata else None
                    )]
                ))
            elif isinstance(delta.content, DeltaToolResult):
                tool_result = delta.content.data
                tool_name = tool_id_to_name.get(tool_result.id, tool_result.id)
                function_response_part = genai_types.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result.content}
                )
                contents.append(genai_types.Content(
                    role="tool",
                    parts=[function_response_part]
                ))
            elif isinstance(delta.content, (DeltaToolInputUpdate, DeltaHistory)):
                pass
            else:
                t.assert_never(delta.content)
        else:
            contents.extend(query_to_contents(delta))
    
    return contents

class StreamArgs(t.TypedDict, total=False):
    generate_content_config: genai_types.GenerateContentConfig

STREAM_ARGS = StreamArgs
MODEL_TYPE = str
ALL_STREAM_ARGS = StreamArgs

class GoogleGenAILLM(LLM[StreamArgs, MODEL_TYPE, genai.Client]):
    def __init__(self, model: MODEL_TYPE, client: t.Optional[genai.Client] = None):
        if not client:
            client = genai.Client()
        self.client = client
        self.model = model

    async def stream(self, **kwargs: t.Unpack[LLMClientStreamArgs[StreamArgs, MODEL_TYPE]]):
        config_kwargs = {}
        
        if functions := kwargs.get("functions", None):
            config_kwargs["tools"] = functions_for_llm(functions)
            
        if prompt := kwargs.get("prompt", None):
            config_kwargs["system_instruction"] = prompt

        if max_tokens := kwargs.get("max_tokens", None):
            config_kwargs["max_output_tokens"] = max_tokens
            
        if temperature := kwargs.get("temperature", None):
            config_kwargs["temperature"] = temperature
            
        if top_p := kwargs.get("top_p", None):
            config_kwargs["top_p"] = top_p
            
        if stop_sequences := kwargs.get("stop_sequences", None):
            config_kwargs["stop_sequences"] = stop_sequences
        
        if generate_content_config := (kwargs.get("kwargs", {}) or {}).get("generate_content_config"):
            config_kwargs = {
                **config_kwargs,
                **generate_content_config.model_dump(exclude_unset=True)
            }

        contents = deltas_to_contents(kwargs.get("history", []))
        
        config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        def handle_response(resp: genai_types.GenerateContentResponse):
            deltas = []
            # TODO: Update to handle streaming function args
            #       when google-genai supports it
            if not resp.candidates:
                return deltas
            for candidate in resp.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    metadata = {}
                    if part.thought_signature:
                        metadata["thought_signature"] = part.thought_signature
                    if part.text:
                        deltas.append(Delta(content=DeltaText(data=part.text), metadata=metadata))
                    if part.function_call:
                        tool_use = ToolUse(
                            id=part.function_call.id or part.function_call.name, # type: ignore
                            name=part.function_call.name, # type: ignore
                            input=part.function_call.args # type: ignore
                        )
                        deltas.append(Delta(content=DeltaToolInputUpdate(
                            name=tool_use.name,
                            id=tool_use.id,
                            data=json.dumps(tool_use.input or {})
                        ), metadata=metadata))
                        deltas.append(Delta(content=DeltaToolUse(data=tool_use), metadata=metadata))
            # Handle usage information
            if resp.usage_metadata:
                usage = Usage(
                    input_tokens=resp.usage_metadata.prompt_token_count or 0,
                    output_tokens=resp.usage_metadata.candidates_token_count or 0,
                    cache_read_input_tokens=0, # TODO
                    cache_written_input_tokens=0 # TODO
                )
                deltas.append(Delta(
                    content=DeltaText(data=""),
                    usage=usage
                ))
            
            return deltas

        if not kwargs.get("no_stream", False):
            # Streaming response using native async API
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=str(self.model),
                contents=contents,
                config=config
            ):
                for delta in handle_response(chunk):
                    yield delta
        else:
            # Non-streaming response using blocking API
            response = await self.client.aio.models.generate_content(
                model=str(self.model),
                contents=contents,
                config=config
            )
            for delta in handle_response(response):
                yield delta