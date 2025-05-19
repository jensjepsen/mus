import wit_world # type: ignore
import mus
import mus.llm
import mus.llm.types
import typing as t
import jsonpickle
import traceback

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



class ProxyClient(mus.llm.types.LLMClient[ExtraArgs, str, None]):
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
        "print": wit_world.print,
      }
      exec(code_with_run, globals)
      return str("Done")
    except Exception as e:
      return str(e)
  