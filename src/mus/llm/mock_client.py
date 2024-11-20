import typing as t
from .llm import LLMClient
from .types import Delta, ToolUse, ToolResult

class StreamArgs(t.TypedDict, total=False):
    pass

class MockLLM(LLMClient[StreamArgs, str]):
    def __init__(self, *args, **kwargs):
        self.responses = []

    def put_response(self, response: Delta):
        self.responses.append(response)
    
    def put_tool_use(self, tool_use: ToolUse):
        self.responses.append(Delta(content={"type": "tool_use", "data": tool_use}))
    
    def put_tool_result(self, tool_result: ToolResult):
        self.responses.append(Delta(content={"type": "tool_result", "data": tool_result}))

    def put_text(self, text: str):
        self.responses.append(Delta(content={"type": "text", "data": text}))

    def stream(self, **kwargs):
        for response in self.responses:
            yield response