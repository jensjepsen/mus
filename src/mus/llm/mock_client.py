import typing as t
from .llm import LLM
from .types import Delta, ToolUse, ToolResult, Query, DeltaText, DeltaToolUse, DeltaToolResult
from collections import defaultdict

class StreamArgs(t.TypedDict, total=False):
    pass

class StubLLM(LLM[StreamArgs, str, None]):
    def __init__(self, *args, **kwargs):
        self.responses: t.Dict[str, t.List[Delta]] = defaultdict(list)

    def put_response(self, q: str, response: Delta):
        self.responses[q].append(response)
    
    def put_tool_use(self, q: str, tool_use: ToolUse):
        self.put_response(q, Delta(content=DeltaToolUse(data=tool_use)))
    
    def put_tool_result(self, q: str, tool_result: ToolResult):
        self.put_response(q, Delta(content=DeltaToolResult(data=tool_result)))

    def put_text(self, q: str, text: str):
        self.put_response(q, Delta(content=DeltaText(data=text)))

    async def stream(self, **kwargs):
        query = kwargs.get("history")[-1]
        if isinstance(query, Query):
            responses = self.responses[str(query.val[0])]
        else:
            responses = [Delta(content=DeltaText(data="Hello"))]
        for response in responses:
            yield response