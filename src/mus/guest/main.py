import extism
import mus
import mus.llm
import mus.llm.types
import typing as t
import jsonpickle
import time
def run(coro):
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

@extism.import_fn("host", "stream")
def stream(kwargs: str) -> str: ...

@extism.import_fn("host", "poll_stream")
def poll_stream(q_id: str) -> str: ...

@extism.import_fn("host", "print")
def print(msg: str): ...



class ProxyClient(mus.llm.types.LLMClient[ExtraArgs, str, None]):
  def __init__(self):
    pass
  async def stream(self, **kwargs: t.Unpack[mus.llm.types.LLMClientStreamArgs[ExtraArgs, str]]) -> t.AsyncGenerator[mus.llm.types.Delta, None]:
    result = stream(jsonpickle.dumps(kwargs))
    
    while delta := poll_stream(result):
      if delta == "[[STOP]]":
        break
      yield jsonpickle.loads(delta)

@extism.plugin_fn
def run():
  code = extism.input_str()
  
  code_with_run = f"""
async def main():
  {code}
run(main())
"""
  client = ProxyClient()
  globals = {
    "client": client,
  }
  exec(code_with_run, globals)
  