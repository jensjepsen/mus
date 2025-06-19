import wit_world # type: ignore
import mus
import mus.llm
import mus.llm.types
import typing as t
import jsonpickle
import sys
import io
import contextlib
import time

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

class ExtraArgs(t.TypedDict):
  pass

class ProxyClient(mus.llm.types.LLM[ExtraArgs, str, None]):
  def __init__(self):
    pass
  async def stream(self, **kwargs: t.Unpack[mus.llm.types.LLMClientStreamArgs[ExtraArgs, str]]) -> t.AsyncGenerator[mus.llm.types.Delta, None]:
    result = wit_world.startstream(jsonpickle.dumps(kwargs)) # type: ignore # returns str
    
    while delta := wit_world.pollstream(result):
      if delta == "[[STOP]]":
        break
      yield jsonpickle.loads(delta) # type: ignore # returns Delta

class WitWorld(wit_world.WitWorld):
  def run(self, code: str):
    try:
      indented = code.split("\n")
      code = "    " + "\n    ".join(indented)
      code_with_run = f"""\
import mus
async def main():
{code}
run_coro(main())
"""
      model = ProxyClient()
      globals = {
        "model": model,
        "run_coro": run_coro,
        "input": wit_world.input,
      }
      with redirect_stdout(wit_world.print):
        exec(code_with_run, globals)
      return str("Done")
    except Exception as e:
      return str(e)
  