import typing as t
from .llm import LLM
from .types import Delta, ToolUse, ToolResult, Query
from collections import defaultdict

class StreamArgs(t.TypedDict, total=False):
    pass

class StubLLM(LLM[StreamArgs, str, None]):
    def __init__(self, *args, **kwargs):
        self.responses: t.Dict[str, t.List[Delta]] = defaultdict(list)

    def put_response(self, q: str, response: Delta):
        self.responses[q].append(response)
    
    def put_tool_use(self, q: str, tool_use: ToolUse):
        self.put_response(q, Delta(content={"type": "tool_use", "data": tool_use}))
    
    def put_tool_result(self, q: str, tool_result: ToolResult):
        self.put_response(q, Delta(content={"type": "tool_result", "data": tool_result}))

    def put_text(self, q: str, text: str):
        self.put_response(q, Delta(content={"type": "text", "data": text}))

    async def stream(self, **kwargs):
        query = kwargs.get("history")[-1]
        if isinstance(query, Query):
            responses = self.responses[str(query.val[0])]
        else:
            breakpoint()
            responses = [Delta(content={"type": "text", "data": "Hello"})]
        for response in responses:
            yield response