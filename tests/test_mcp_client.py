import pytest

from mus.mcp.client import make_client, tool_to_function, get_tools
from mus.mcp.server import MCPServer
from mus.llm.types import File

@pytest.mark.asyncio
async def test_mcp_client_make_client():
    server = MCPServer()

    @server.tool
    async def dummy_tool() -> str:
        """A dummy tool for testing."""
        return "dummy response"

    async with make_client(server) as client:
        assert client is not None, "Expected client to be created successfully"
        
        tools = await get_tools(client)
        assert len(tools) == 1, "Expected one tool to be returned"
        assert tools[0].schema["name"] == "dummy_tool", "Expected tool name to match"
        assert tools[0].schema["description"] == "A dummy tool for testing.", "Expected tool description to match"
        assert callable(tools[0].function), "Expected function to be callable"

        # Test the tool
        result = await tools[0].function()
        assert result == "dummy response", "Expected result of dummy_tool to be 'dummy response'"

@pytest.mark.asyncio
async def test_mcp_client_tool_to_function():
    server = MCPServer()

    @server.tool
    async def add_numbers(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)

    async with make_client(server) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 1, "Expected one tool to be registered"
        
        function = await tool_to_function(tools.tools[0], client)
        assert function.schema["name"] == "add_numbers", "Expected function name to match tool name"
        assert function.schema["description"] == "Adds two numbers.", "Expected function description to match tool description"
        assert "a" in function.schema["schema"]["properties"], "Expected 'a' parameter to be in function schema"
        assert "b" in function.schema["schema"]["properties"], "Expected 'b' parameter to be in function schema"
        assert function.schema["schema"]["properties"]["a"]["type"] == "integer", "Expected 'a' parameter type to be integer"
        assert function.schema["schema"]["properties"]["b"]["type"] == "integer", "Expected 'b' parameter type to be integer"

        # Test the function
        result = await function.function(a=3, b=5)
        assert result == "8", "Expected result of add_numbers to be '8'"
    

@pytest.mark.asyncio
async def test_mcp_client_tool_return_file():
    server = MCPServer()

    @server.tool
    async def return_text_file() -> File:
        """Returns a file."""
        return File(
            b64type="text/plain",
            content="Hello, world!"
        )
    
    async with make_client(server) as client:
        tools = await get_tools(client)
        assert len(tools) == 1, "Expected one tool to be registered"

        assert tools[0].schema["name"] == "return_text_file", "Expected tool name to match"
        assert tools[0].schema["description"] == "Returns a file.", "Expected tool description to match"
        assert tools[0].schema["schema"]["type"] == "object", "Expected tool schema type to be object"

        result = await tools[0].function()
        assert isinstance(result, File), "Expected result to be of type File"
        assert result.b64type == "text/plain", "Expected file type to be text/plain"
        assert result.content == "Hello, world!", "Expected file content to match"

@pytest.mark.asyncio
async def test_mcp_client_tool_return_list():
    server = MCPServer()

    @server.tool
    async def return_list():
        """Returns a list."""
        return ["apple", "banana", "cherry", File(b64type="text/plain", content="dummy")]
    
    async with make_client(server) as client:
        tools = await get_tools(client)
        assert len(tools) == 1, "Expected one tool to be registered"

        assert tools[0].schema["name"] == "return_list", "Expected tool name to match"
        assert tools[0].schema["description"] == "Returns a list.", "Expected tool description to match"

        result = await tools[0].function()
        assert isinstance(result, list), "Expected result to be a list"
        assert len(result) == 4, "Expected list to contain four items"
        assert result[0] == "apple", "Expected first item to be 'apple'"
        assert result[1] == "banana", "Expected second item to be 'banana'"
        assert result[2] == "cherry", "Expected third item to be 'cherry'"
        assert isinstance(result[3], File), "Expected fourth item to be of type File"
        assert result[3].b64type == "text/plain", "Expected file type to be text/plain"
        assert result[3].content == "dummy", "Expected file content to match 'dummy'"
        
        
@pytest.mark.asyncio
async def test_mcp_client_get_tools():
    server = MCPServer()

    @server.tool
    async def add_numbers(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    @server.tool
    async def multiply_numbers(a: int, b: int) -> str:
        """Multiplies two numbers."""
        return str(a * b)

    async with make_client(server) as client:
        tools = await get_tools(client)
        assert len(tools) == 2, "Expected one tool to be returned"
        assert tools[0].schema["name"] == "add_numbers", "Expected tool name to match"
        assert tools[0].schema["description"] == "Adds two numbers.", "Expected tool description to match"
        assert "a" in tools[0].schema["schema"]["properties"], "Expected 'a' parameter to be in tool schema"
        assert "b" in tools[0].schema["schema"]["properties"], "Expected 'b' parameter to be in tool schema"
        assert tools[0].schema["schema"]["properties"]["a"]["type"] == "integer", "Expected 'a' parameter type to be integer"
        assert tools[0].schema["schema"]["properties"]["b"]["type"] == "integer", "Expected 'b' parameter type to be integer"

        # Test the tool
        result = await tools[0].function(a=3, b=5)
        assert result == "8", "Expected result of add_numbers to be '8'"

        assert len(tools) == 2, "Expected two tools to be returned"
        assert tools[1].schema["name"] == "multiply_numbers", "Expected second tool name to match"
        assert tools[1].schema["description"] == "Multiplies two numbers.", "Expected second tool description to match"
        assert "a" in tools[1].schema["schema"]["properties"], "Expected 'a' parameter to be in second tool schema"
        assert "b" in tools[1].schema["schema"]["properties"], "Expected 'b' parameter to be in second tool schema"
        assert tools[1].schema["schema"]["properties"]["a"]["type"] == "integer", "Expected 'a' parameter type to be integer in second tool"
        assert tools[1].schema["schema"]["properties"]["b"]["type"] == "integer", "Expected 'b' parameter type to be integer in second tool"
        # Test the second tool
        result = await tools[1].function(a=3, b=5)
        assert result == "15", "Expected result of multiply_numbers to be '15'"
