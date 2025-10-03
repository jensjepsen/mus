import wit_world # type: ignore
import mus
import mus.llm
import mus.llm.types
import mus.functions
import typing as t
import sys
import io
import contextlib
import time
import json
import cattrs
from mus.converters.delta import delta_converter

class ToolCallableWithCall(mus.functions.ToolCallable):
    async def __call__(self, *args: t.Any, **kwargs: t.Any):
        return await self.function(*args, **kwargs)

class StdOut(io.StringIO):
    def __init__(self, callback: t.Callable[[str], None]):
        super().__init__()
        self._callback = callback
        
    def write(self, s: str) -> int:
        self._callback(s)
        return super().write(s)


class PollingFile(io.TextIOBase):
    """A file-like object that calls a polling function when read operations occur."""
    
    def __init__(self, poll_function, poll_interval=0.1):
        """
        Initialize the polling file object.
        
        Args:
            poll_function: Function to call when reading. Should return a string or None.
            poll_interval: Time to wait between polling attempts if no data is available.
        """
        self.poll_function = poll_function
        self.poll_interval = poll_interval
        self.buffer = ""
        
    def readable(self):
        return True
        
    def readline(self, size=-1):
        """Read a line from the polling function."""
        # If we have buffered data with a newline, return that
        if "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            return line + "\n"
            
        # Keep polling until we get a complete line
        while "\n" not in self.buffer:
            data = self.poll_function()
            if data:
                self.buffer += data
                if "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    return line + "\n"
            else:
                time.sleep(self.poll_interval)
                
        # Should not reach here, but just in case
        if self.buffer:
            result, self.buffer = self.buffer, ""
            return result
        
        return ""
        
    def read(self, size: t.Optional[int]=-1):
        """Read up to size characters from the polling function."""
        size = size if size is not None else -1
        if size <= 0:  # Read all available
            while True:
                data = self.poll_function()
                if not data:
                    break
                self.buffer += data
            result, self.buffer = self.buffer, ""
            return result
            
        # Read specific amount
        while len(self.buffer) < size:
            data = self.poll_function()
            if data:
                self.buffer += data
            else:
                break
                
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return result

    
@contextlib.contextmanager
def redirect_stdout(callback: t.Callable[[str], None]):
    old_stdout = sys.stdout
    sys.stdout = StdOut(callback)
    try:
        yield
    finally:
        sys.stdout = old_stdout

@contextlib.contextmanager
def redirect_stdin(callback: t.Callable[[], str]):
    old_stdin = sys.stdin
    sys.stdin = PollingFile(callback)
    try:
        yield
    finally:
        sys.stdin = old_stdin

def run_coro(coro):
    try:
        while True:
            try:
                coro.send(None)
            except StopIteration as e:
                return e.value
    except Exception as e:
        # Handle any other exceptions
        raise e

class Empty(t.TypedDict):
  pass

class ProxyClient(mus.llm.types.LLM[Empty, str, None]):
  def __init__(self, model_name: str):
    self.model_name = model_name

  async def stream(self, **kwargs: t.Unpack[mus.llm.types.LLMClientStreamArgs[Empty, str]]) -> t.AsyncGenerator[mus.llm.types.Delta, None]:
    result = wit_world.startstream(self.model_name, json.dumps(cattrs.unstructure(kwargs)))
    while delta := wit_world.pollstream(result):
      if delta == "[[STOP]]":
        break
      yield delta_converter.structure(json.loads(delta), mus.llm.types.Delta)

def invoke_function(name: str, inputs: dict) -> t.Any:
    # TODO: Future, we could add support for async functions here
    #       and also support functions that return generators
    #       by polling the generator and yielding results back to the caller
    # Question: What happens if the function returns something that is not JSON serializable?
    #           We should probably catch that in the host
    result = wit_world.runfunction(name, json.dumps(delta_converter.unstructure(inputs)))
    while delta := wit_world.pollstream(result):
      if delta == "[[STOP]]":
        break
      return json.loads(delta)

def make_tool_proxies(tools: t.Dict[str, mus.llm.types.FunctionSchemaNoAnnotations]) -> dict[str, t.Callable[..., t.Any]]:
  proxies = {}
  for name, func in tools.items():
    def make_proxy(f: mus.llm.types.FunctionSchemaNoAnnotations) -> ToolCallableWithCall:
      async def proxy(*args, **inputs):
        return invoke_function(f["name"], inputs)
      return ToolCallableWithCall(
        schema=mus.llm.types.FunctionSchema(
            **f,
            annotations=[]
        ),
        function=proxy
      )
    proxies[name] = make_proxy(func)
  return proxies

def make_function_proxies(functions: t.List[str]) -> dict[str, t.Callable[..., t.Any]]:
  proxies = {}
  for name in functions:
    def make_proxy(f_name: str) -> t.Callable[..., t.Any]:
      async def proxy(*args, **inputs):
        return invoke_function(f_name, inputs)
      return proxy
    proxies[name] = make_proxy(name)
  return proxies

class WitWorld(wit_world.WitWorld):
  def run(self, code: str, inputs: str, llms: list[str], tools: str, functions: list[str]) -> str:
    try:
      tools_list = delta_converter.structure(json.loads(tools), t.Dict[str, mus.llm.types.FunctionSchemaNoAnnotations])
      deserialized_inputs = delta_converter.structure(json.loads(inputs), t.Dict[str, t.Any])
      tool_proxies = make_tool_proxies(tools_list)
      function_proxies = make_function_proxies(functions)
      indented = code.split("\n")
      code = "    " + "\n    ".join(indented)
      code_with_run = f"""\
import mus
async def main():
{code}
result = run_coro(main())
"""
      
      llm_proxies = {name: ProxyClient(name) for name in llms}
      globals = {
        "run_coro": run_coro,
        "input": wit_world.input,
        **llm_proxies,
        **function_proxies,
        **tool_proxies,
        **deserialized_inputs
      }
      
      with redirect_stdout(wit_world.print):
        exec(code_with_run, globals)
      if "result" in globals:
        return json.dumps({"status": "success", "message": "Done", "result": delta_converter.unstructure(globals["result"])})
      
      return json.dumps({"status": "success", "message": "Done", "result": None})
    except Exception as e:
      return json.dumps({"status": "error", "message": str(e)})
