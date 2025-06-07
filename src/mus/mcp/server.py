#!/usr/bin/env python3
import json
import sys
import base64
import inspect
from typing import Dict, Any, Optional, Union, TypedDict, Callable, Literal, Coroutine, Protocol, Union, ParamSpec, Generic
from ..functions import func_to_schema
from .uri_matcher import URIMatcher
from ..llm.types import ToolReturnValue, File

# MCP protocol types

class Tool(TypedDict):
    """Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    annotations: Dict[str, Any]

class Resource(TypedDict):
    """A known resource that the server is capable of reading."""
    uri: str
    """The URI of this resource."""
    name: str
    """A human-readable name for this resource."""
    description: Optional[str]
    """A description of what this resource represents."""
    mimeType: Optional[str]
    """The MIME type of this resource, if known."""
    size: Optional[int]

class ResourceTemplate(TypedDict):
    name: str
    """Name of the resource template"""
    description: str
    """Description of the resource template"""
    uriTemplate: str
    """URI template for the resource, with {params} as placeholders for parameters"""

class ToolsListResponse(TypedDict):
    """Tool response"""
    tools: list[Tool]

class ResourcesListResponse(TypedDict):
    """Response for listing resources"""
    resources: list[Resource]

class ResourcesTemplatesListResponse(TypedDict):
    """Response for listing resource templates"""
    resourceTemplates: list[ResourceTemplate]


class ResourceReadResponseContentBlob(TypedDict):
    """Content of a resource read response as a blob"""
    blob: str
    mimeType: str
    name: str
    uri: str

class ResourceReadResponseContentText(TypedDict):
    """Content of a resource read response as text"""
    text: str
    mimeType: str
    name: str
    uri: str

ResourceReadResponseContent = Union[ResourceReadResponseContentBlob, ResourceReadResponseContentText]

class ResourcesReadResponse(TypedDict):
    """Response for reading a resource"""
    contents: list[ResourceReadResponseContent]


class ToolReturnValueText(TypedDict):
    """Tool return value"""
    type: Literal["text"]
    text: str

class ToolReturnValueImage(TypedDict):
    """Tool return value for images"""
    type: Literal["image"]
    data: str
    mimeType: str

class ToolsCallResponse(TypedDict):
    """Tool return value"""
    content: list[ToolReturnValueText | ToolReturnValueImage]

# JSON RPC types

class JsonRpcError(Exception):
    """JSON-RPC error class"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"{message} (code: {code})")


# Internal types for tool and resource registration

class ToolRegistration(TypedDict):
    """Tool registration"""
    tool: Tool
    function: "ToolFunction"  # Function to execute the tool

class ResourceRegistration(TypedDict):
    """Resource registration"""
    name: str
    """Name of the resource"""
    mimeType: Optional[str]
    """MIME type of the resource, if known"""
    function: "ResourceFunction"  # Function to read the resource
    """Function to read the resource at the given URI"""

class StopServer(Exception):
    """Exception to stop the server"""
    pass

class ResourceFunction(Protocol):
    """Protocol for resource functions"""
    __name__: str
    __doc__: Optional[str]
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        """Read the resource at the given URI"""
        ...

class ToolFunction(Protocol):
    """Protocol for tool functions"""
    __name__: str
    __doc__: Optional[str]
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        """Execute the tool with the given arguments"""
        ...

class ReadLineProtocol(Protocol):
    async def __call__(self) -> str:
        """Read a line from the input"""
        ...

class WriteLineProtocol(Protocol):
    async def __call__(self, line: str):
        """Write a line to the output"""
        ...


P = ParamSpec("P")
class RPCMethodProtocol(Protocol, Generic[P]):
    __name__: str
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """Method to be called by RCP clients connected to the server"""
        ...

class MCPServer:
    """Model Context Protocol (MCP) server that communicates over stdio"""
    
    def __init__(self, read_line: Optional[ReadLineProtocol] = None, write_line: Optional[WriteLineProtocol] = None):
        self.methods = {}
        self.initialized = False
        self.set_stdio(read_line or self.default_read_line, write_line or self.default_write_line)
        
        # Register methods
        self.register_method("initialize", self.initialize)
        self.register_method("tools/list", self.list_tools)
        self.register_method("tools/call", self.call_tool)
        self.register_method("resources/list", self.list_resources)
        self.register_method("resources/templates/list", self.list_resource_templates)
        self.register_method("resources/read", self.get_resource)
        
        self.register_method("ping", self.ping)
        self.tools : dict[str, ToolRegistration] = {}
        self.resources: list[Resource] = []
        self.resource_templates: list[ResourceTemplate] = []
        self.resource_uri_matcher = URIMatcher[ResourceRegistration]()
    
    def set_stdio(self, read_line: ReadLineProtocol, write_line: WriteLineProtocol):
        """Set custom read and write functions for stdin/stdout"""
        self.read_line = read_line
        self.write_line = write_line

    async def default_read_line(self) -> str:
        """Read a line from stdin"""
        return sys.stdin.readline()

    async def default_write_line(self, line: str):
        print(line, flush=True)

    def register_method(self, name: str, func: RPCMethodProtocol):
        """Register a method with the server"""
        self.methods[name] = func

    def tool(self, f: ToolFunction) -> ToolFunction:
        """Decorator to register a function as a tool"""

        # Verify that the function is a coroutine function
        if not inspect.iscoroutinefunction(f):
            raise ValueError("Tool function must be a coroutine function (async def)")
        
        input_schema = func_to_schema(f) # type: ignore # Func_to_schema is restrictive right now, and should be relaxed to allow correct return type

        # Create tool definition
        tool = ToolRegistration(
            tool=Tool(
                name=input_schema["name"],
                description=input_schema["description"],
                inputSchema=input_schema["schema"],
                annotations={}
            ),
            function=f
        )
        
        # Register the tool
        self.tools[input_schema["name"]] = tool
        
        return f
    
    def resource(self, uri_template: str, name: Optional[str] = None, mime_type: Optional[str] = None):
        """Decorator to register a function as a resource"""
        def decorator(func: ResourceFunction):
            # Create resource registration
            if not inspect.iscoroutinefunction(func):
                raise ValueError("Resource function must be a coroutine function (async def)")
        

            inferred_name = name or getattr(func, "__name__", None)
            if not inferred_name:
                raise ValueError("Resource must have a name, either provided or inferred from the function name.")
            
            inferred_description = func.__doc__ or ""
            resource = ResourceRegistration(
                name=inferred_name,
                mimeType=mime_type,
                function=func
            )
            # Register the resource
            if self.resource_uri_matcher.add_pattern(uri_template, resource) == "static":
                # Static resource, add to resources list
                self.resources.append(Resource(
                    uri=uri_template,
                    name=inferred_name,
                    description=inferred_description,
                    mimeType=mime_type,
                    size=None  # Size can be determined later if needed
                ))
            else:
                # Dynamic resource, add to templates
                self.resource_templates.append(
                    ResourceTemplate(
                        name=inferred_name,
                        description=inferred_description,
                        uriTemplate=uri_template
                    )
                )
            
            return func
        
        return decorator
       
    async def initialize(self, protocolVersion: str, capabilities: Dict[str, Any], clientInfo: Dict[str, str]) -> Dict[str, Any]:
        """Handle initialization request"""
        self.initialized = True
        
        # Return server capabilities and information
        return {
            "protocolVersion": protocolVersion,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "MCP Python Server",
                "version": "1.0.0"
            }
        }

    async def ping(self) -> Dict[str, str]:
        """Ping the server"""
        return {
            "message": "pong"
        }

    async def list_tools(self, **kwargs) -> ToolsListResponse:
        """List available tools"""
        return {
            "tools": [
                tool_registration["tool"]
                for tool_registration in self.tools.values()
            ]
                
        }
    
    async def list_resources(self, **kwargs) -> ResourcesListResponse:
        """List known resources"""
        return {
            "resources": self.resources,
        }
    
    async def list_resource_templates(self, **kwargs) -> ResourcesTemplatesListResponse:
        """List resource templates"""
        return {
            "resourceTemplates": self.resource_templates,
        }
    
    async def call_tool(self, name: str, arguments: dict, **kwargs) -> ToolsCallResponse:
        """Call a tool by name with the provided arguments"""
        if name in self.tools:
            tool_registration = self.tools[name]
            tool_function = tool_registration["function"]
            
            result = await tool_function(**arguments)

            # Call the tool function
            if isinstance(result, str):
                result = ToolReturnValueText(type="text", text=result)
            elif isinstance(result, File):
                result = ToolReturnValueImage(
                    type="image",
                    data=result.content,
                    mimeType=result.b64type,
                )
            elif isinstance(result, list):
                result = [
                    ToolReturnValueText(type="text", text=r) if isinstance(r, str) else
                    ToolReturnValueImage(type="image", data=r.content, mimeType=r.b64type) if isinstance(r, File) else
                    r for r in result
                ]
            else:
                raise JsonRpcError(-32603, "Invalid tool return type")
            return {
                "content": [result] if not isinstance(result, list) else result
            }
            
        raise JsonRpcError(-32601, f"Tool not found: {name}")
    
    async def get_resource(self, uri: str, **kwargs) -> ResourcesReadResponse:
        """Get a resource by URI"""
        resource = self.resource_uri_matcher.match(uri)
        if resource:
                
            func = resource.uri.value["function"]
            name = resource.uri.value["name"]
            
            result = await func(**resource.values)
            
            if isinstance(result, str):
                return ResourcesReadResponse(
                    contents=[
                        ResourceReadResponseContentText(
                            text=result,
                            mimeType="text/plain",
                            name=name,
                            uri=uri
                        )
                    ]
                )
            elif isinstance(result, File):
                return ResourcesReadResponse(
                    contents=[
                        ResourceReadResponseContentBlob(
                            blob=result.content,
                            mimeType=result.b64type,
                            name=name,
                            uri=uri
                        )
                    ]
                )
            else:
                raise JsonRpcError(-32603, "Invalid resource return type")
            
        
        raise JsonRpcError(-32601, f"Resource not found: {uri}")

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a JSON-RPC request and return a response"""
        
        # Check if it's a valid request
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            return self._error_response(
                request.get("id"), -32600, "Invalid Request: Not a valid JSON-RPC 2.0 request"
            )
        
        # Get request ID (might be None for notifications)
        request_id = request.get("id")
        
        # Check for method
        if "method" not in request or not isinstance(request["method"], str):
            return self._error_response(
                request_id, -32600, "Invalid Request: Method not specified or not a string"
            )
        
        method_name = request["method"]
        params = request.get("params", {})
        
        # Handle notification (no response needed)
        if request_id is None:
            if method_name == "initialized":
                # Handle initialized notification
                return None
            elif method_name == "disconnect":
                # Handle disconnect notification
                raise StopServer()
            
            # Process other notifications if needed
            return None
        
        # Check if method exists
        if method_name not in self.methods:
            return self._error_response(
                request_id, -32601, f"Method not found: {method_name}"
            )
        
        # Execute the method
        try:
            result = await self.methods[method_name](**(params or {}))
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except TypeError as e:
            return self._error_response(
                request_id, -32602, f"Invalid params: {str(e)}"
            )
        except JsonRpcError as e:
            return self._error_response(
                request_id, e.code, e.message, e.data
            )
        except Exception as e:
            return self._error_response(
                request_id, -32603, f"Internal error: {str(e)}"
            )
    
    def _error_response(self, request_id: Optional[Union[str, int]], 
                       code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create an error response"""
        error = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data
            
        return {
            "jsonrpc": "2.0",
            "error": error,
            "id": int(request_id) if request_id is not None else None
        }
    
    async def run(self):
        """Run the server, reading from stdin and writing to stdout"""
        while True:
            line = await self.read_line()
            if not line:
                # EOF or empty line, stop the server
                break
            try:
                request = json.loads(line)
                
                # Handle batch requests
                if isinstance(request, list):
                    responses = [await self.handle_request(req) for req in request]
                    # Filter out None responses (from notifications)
                    responses = [r for r in responses if r is not None]
                    if responses:
                        await self.write_line(json.dumps(responses))
                else:
                    response = await self.handle_request(request)
                    # Only send response if it's not a notification
                    #if request.get("method") == "tools/list":
                    #    exit(json.dumps(response))

                    if response is not None:
                        await self.write_line(json.dumps(response))
                        
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: Invalid JSON"
                    },
                    "id": None
                }
                await self.write_line(json.dumps(error_response))
            except StopServer:
                # Stop the server gracefully
                break

if __name__ == "__main__":
    import asyncio
    server = MCPServer()
    
    @server.tool
    async def example_tool(input: str):
        """Example tool that echoes the input"""
        return f"Echo: {input}"

    asyncio.run(server.run())