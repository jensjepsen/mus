import typing as t
from anthropic import AnthropicBedrock
from anthropic import types as at

from .types import LLMClient, Delta, ToolUse, ToolResult, QueryIterableType, QuerySimpleType, File
from ..functions import functions_for_llm

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


class AnthropicLLM(LLMClient):
    def __init__(self, client: AnthropicBedrock):
        self.client = client

    def stream(self, prompt: t.Optional[str], query: t.Optional[QueryIterableType], history: t.List[t.Dict[str, t.Any]], functions: t.List[t.Callable], invoke_function: t.Callable, function_choice: t.Literal["auto", "any"]) -> t.Iterable[Delta]:
        kwargs = {}
        if functions:
            # should be abstracted to LLM class
            kwargs["tools"] = functions_for_llm(functions)
            if function_choice:
                kwargs["tool_choice"] = {
                    "type": function_choice
                }
        if prompt:
            kwargs["system"] = prompt
        
        if query:
            history = self.extend_history(history, {"role": "user", "content": query_to_content(query)})
        
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
                    yield Delta(type="text", content=event.text)
                elif event.type == "content_block_stop":
                    if event.content_block.type == "tool_use":
                        function_blocks.append(event.content_block)
                
                elif event.type == "message_stop":
                    history = self.extend_history(history, {
                        "role": event.message.role,
                        "content": event.message.content
                    })
                    
                    if event.message.stop_reason == "tool_use":
                        func_message: at.MessageParam = {
                            "role": "user",
                            "content": []
                        }
                        for block in function_blocks:
                            yield Delta(type="tool_use", content=ToolUse(name=block.name, input=block.input))
                            func_result = invoke_function(block.name, block.input)

                            yield Delta(type="tool_result", content=ToolResult(content=func_result))
                            
                            func_message["content"] += [{
                                "tool_use_id": block.id,
                                "type": "tool_result",
                                "content": query_to_content(func_result)
                            }]
                        history = self.extend_history(history, func_message)
                        history = yield from self.stream(prompt, None, history, functions, invoke_function, function_choice)
        return history

    def extend_history(self, history: t.List[at.Message], new_message: at.MessageParam):
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