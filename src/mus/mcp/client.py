import typing as t
import anyio
from contextlib import asynccontextmanager
import datetime
import json
from mcp import ClientSession
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from mcp import ClientSession, Tool
from mcp.types import BlobResourceContents, TextResourceContents

from ..functions import ToolCallable, FunctionSchema
from ..llm.types import ToolReturnValue
from ..llm.types import File
from ..mcp import server

@asynccontextmanager
async def make_client(server: server.MCPServer):
    """Create and return an MCP client instance, given a server instance."""

    read_stream_writer, read_stream = anyio.create_memory_object_stream()
    write_stream, write_stream_reader = anyio.create_memory_object_stream()

    async def send(line: str) -> None:
        """Send data to the MCP server."""
        await  read_stream_writer.send(SessionMessage(message=JSONRPCMessage(**json.loads(line))))
    
    async def receive() -> str:
        """Receive data from the MCP server."""
        line = await write_stream_reader.receive()
        line = line.message.model_dump_json()
        
        return line
    
    server.set_stdio(receive, send)

    async def run_server():
        """Run the MCP server."""
        try: 
            return await server.run()
        except anyio.EndOfStream as e:
            print(f"Server stopped due to end of stream: {e}")
            return None

    
    # Start the server in a background task
    async with anyio.create_task_group() as tg:
        tg.start_soon(run_server)
        try:
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=datetime.timedelta(seconds=10)) as session:
                await session.initialize()
                try:
                    yield session
                except anyio.EndOfStream:
                    # Handle end of stream gracefully
                    print("End of stream reached, closing client session.")
                finally:
                    read_stream.close()
                    write_stream.close()
        except anyio.EndOfStream:
            # Handle end of stream gracefully
            print("End of stream reached, closing client session.")
        finally:
            read_stream.close()
            write_stream.close()

async def tool_to_function(tool: Tool, session: ClientSession):
    async def call_tool(**kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        output: ToolReturnValue = []
        for item in result.content:
            if item.type == "text":
                output.append(item.text)
            elif item.type == "image":
                output.append(
                    File(
                        b64type=item.mimeType,
                        content=item.data
                    )
                )
            elif item.type == "resource":
                if isinstance(item.resource, TextResourceContents):
                    output.append(
                        File(
                            b64type=item.resource.mimeType or "text/plain",
                            content=item.resource.text
                        )
                    )
                elif isinstance(item.resource, BlobResourceContents):
                    output.append(
                        File(
                            b64type=item.resource.mimeType or "application/octet-stream",
                            content=item.resource.blob
                        )
                    )
                else:
                    raise ValueError(f"Unknown resource type: {item.type}")
        if len(output) == 1: 
            output = output[0]
        return output
                
    return ToolCallable(
        function=call_tool,
        schema=FunctionSchema(
            name=tool.name,
            description=tool.description or "",
            schema=tool.inputSchema,
            annotations=[(k, v) for k, v in call_tool.__annotations__.items() if not k =="return"]
        )
    )    

async def get_tools(session: ClientSession):
    tools = await session.list_tools()
    return [
        await tool_to_function(tool, session)
        for tool in tools.tools
    ]