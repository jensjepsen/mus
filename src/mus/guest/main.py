import extism
import mus
import mus.llm
import mus.llm.types
import typing as t
import jsonpickle

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

@extism.import_fn("host", "stream") # type: ignore # exists in extism python-pdk
def stream(kwargs: str) -> str: ...

@extism.import_fn("host", "poll_stream") # type: ignore # exists in extism python-pdk
def poll_stream(q_id: str) -> str: ...

@extism.import_fn("host", "print") # type: ignore # exists in extism python-pdk
def print(msg: str): ...



class ProxyClient(mus.llm.types.LLMClient[ExtraArgs, str, None]):
  def __init__(self):
    pass
  async def stream(self, **kwargs: t.Unpack[mus.llm.types.LLMClientStreamArgs[ExtraArgs, str]]) -> t.AsyncGenerator[mus.llm.types.Delta, None]:
    result = stream(jsonpickle.dumps(kwargs)) # type: ignore # returns str
    
    while delta := poll_stream(result):
      if delta == "[[STOP]]":
        break
      yield jsonpickle.loads(delta) # type: ignore # returns Delta

@extism.plugin_fn # type: ignore # exists in extism python-pdk
def run():
  code = extism.input_str().strip() # type: ignore # exists in extism python-pdk
  indented = code.split("\n")
  code = "    " + "\n    ".join(indented)
  code_with_run = f"""
import mus
async def main():
{code}
run_coro(main())
"""
  client = ProxyClient()
  globals = {
    "client": client,
    "run_coro": run_coro,
    "print": print,
  }
  exec(code_with_run, globals)
  