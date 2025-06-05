from mcp import ClientSession, Tool

async def tool_to_function(tool: Tool, session: ClientSession):
    def caller():
        pass

async def tools_from_mcp(session: ClientSession):
    tools = session.list_tools()