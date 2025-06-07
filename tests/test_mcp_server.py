import pytest
from pydantic import AnyUrl
import mus

from mcp import types
import mus.mcp.server
from mus.mcp.client import make_client

@pytest.mark.asyncio
async def test_mcp_server_invalid_tool_not_async():
    server = mus.mcp.server.MCPServer()

    with pytest.raises(ValueError, match="Tool function must be a coroutine"):
        # Not async function, should raise an error
        @server.tool # type: ignore # we're passing a non-async function intentionally, to test error handling
        def invalid_tool(a: int, b: int) -> str:
            """This tool is intentionally invalid."""
            return str(a + b)

@pytest.mark.asyncio
async def test_mcp_server_invalid_resource_not_async():
    server = mus.mcp.server.MCPServer()

    with pytest.raises(ValueError, match="Resource function must be a coroutine"):
        # Not async function, should raise an error
        @server.resource("mus://invalid")
        def invalid_resource(a: str) -> str:
            """This resource is intentionally invalid."""
            return "This should not work"
    
@pytest.mark.asyncio
async def test_mcp_server_tools():
    server = mus.mcp.server.MCPServer()

    @server.tool
    async def add_numbers(a: int, b: int) -> str:
        """Adds two numbers."""
        return str(a + b)
    
    @server.tool
    async def multiply_numbers(a: int, b: int):
        """Multiplies two numbers."""
        return str(a * b)
    
    async with make_client(server) as client:
        tools = await client.list_tools()
        assert len(tools.tools), "Expected tools list to contain tools"
        tool_dict = {tool.name: tool for tool in tools.tools}

        assert any(tool.name == "add_numbers" for tool in tools.tools), "Expected 'add_numbers' tool to be in the list"
        assert any(tool.name == "multiply_numbers" for tool in tools.tools), "Expected 'multiply_numbers' tool to be in the list"
        assert tool_dict["add_numbers"].description == "Adds two numbers.", "Expected 'add_numbers' tool to have correct description"
        assert tool_dict["multiply_numbers"].description == "Multiplies two numbers.", "Expected 'multiply_numbers' tool to have correct description"

        # Test add_numbers tool        
        response = await client.call_tool("add_numbers", {"a": 3, "b": 5})
        assert len(response.content) == 1, "Expected one content item in response"
        assert isinstance(response.content[0], types.TextContent), "Expected content to be of type Text"
        assert response.content[0].text == "8"
        # Test multiply_numbers tool
        response = await client.call_tool("multiply_numbers", {"a": 3, "b": 5})
        assert len(response.content) == 1, "Expected one content item in response"
        assert isinstance(response.content[0], types.TextContent), "Expected content to be of type Text"
        assert response.content[0].text == "15"
        
@pytest.mark.asyncio
async def test_mcp_server_resources():
    server = mus.mcp.server.MCPServer()

    @server.resource("mus://time")
    async def time() -> str:
        """Returns the current time."""
        return "2023-10-01T12:00:00Z"  # Fixed time for testing purposes

    @server.resource("number://{num}/square")
    async def square(num: int) -> str:
        """Returns the square of a number."""
        return str(int(num) * int(num))
    
    @server.resource("file://somefile.png")
    async def somefile():
        """Returns a dummy file content."""
        return mus.File(
            b64type="image/png",
            content="asdsa",
        )

    async with make_client(server) as client:
        resources = await client.list_resources()
        assert len(resources.resources), "Expected resources list to contain resources"
        resource_dict = {resource.name: resource for resource in resources.resources}

        time_resource = resource_dict.get("time")
        assert time_resource, "Expected 'time' resource to be in the list"
        assert time_resource.description == "Returns the current time.", "Expected 'time' resource to have correct description"
        assert str(time_resource.uri) == "mus://time", "Expected 'time' resource to have correct URI"

        resource_templates = await client.list_resource_templates()
        assert len(resource_templates.resourceTemplates), "Expected resource templates list to contain templates"

        resource_templates_dict = {template.name: template for template in resource_templates.resourceTemplates}

        square_resource = resource_templates_dict.get("square")

        assert square_resource, "Expected 'square' resource to be in the list"
        assert square_resource.description == "Returns the square of a number.", "Expected 'number://{num}/square' resource to have correct description"
        assert str(square_resource.uriTemplate) == "number://{num}/square", "Expected 'number://{num}/square' resource to have correct URI"

        # Test time resource
        response = await client.read_resource(AnyUrl("mus://time"))
        assert len(response.contents) == 1, "Expected one content item in response"
        assert isinstance(response.contents[0], types.TextResourceContents), "Expected content to be of type Text"
        assert response.contents[0].mimeType == "text/plain", "Expected time resource to return correct mimeType"
        assert response.contents[0].text == "2023-10-01T12:00:00Z", "Expected time resource to return fixed time"
        

        # Test square resource
        response = await client.read_resource(AnyUrl("number://5/square"))
        assert len(response.contents) == 1, "Expected one content item in response"
        assert isinstance(response.contents[0], types.TextResourceContents), "Expected content to be of type Text"
        assert response.contents[0].mimeType == "text/plain", "Expected square resource to return correct mimeType"
        assert response.contents[0].text == "25", "Expected square resource to return correct square value"
        

        # Test file resource
        response = await client.read_resource(AnyUrl("file://somefile.png"))
        assert len(response.contents) == 1, "Expected one content item in response"
        assert isinstance(response.contents[0], types.BlobResourceContents), "Expected content to be of type Blob"
        assert response.contents[0].mimeType == "image/png", "Expected file resource to return correct b64type"
        assert response.contents[0].blob == "asdsa", "Expected file resource to return correct content"