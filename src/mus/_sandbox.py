import asyncio
import jsonpickle
import threading
import queue
import uuid
import typing as t
import asyncio
import os
import inspect
from .llm.llm import LLMClient
from .llm.types import LLMClientStreamArgs
import textwrap
from .guest.bindings import imports as guest_imports
from .guest.bindings import types as guest_types
from .guest.bindings import Root, RootImports
from wasmtime import Store, Engine, Config
class Stop:
    pass

def run_coroutine_in_thread(coroutine_func, *args, **kwargs):
    q_id = str(uuid.uuid4())
    q = queue.Queue()
    queues[q_id] = q
    
    def thread_target():
        try:
            # asyncio.run creates a new event loop, runs the coroutine, and closes the loop
            asyncio.run(coroutine_func(*args, **kwargs, q_id=q_id, queue=q))
        except Exception as e:
            q.put(e)
        finally:
            q.put(Stop())
    
    thread = threading.Thread(target=thread_target)
    thread.start()
    return q_id

queues = {}


class SandboxableCallable(t.Protocol):
    async def __call__(self, model: LLMClient) -> None:
        ...

class SandboxReturnCallable(t.Protocol):
    def __call__(self, model: LLMClient) -> str:
        ...

@t.overload
def sandbox(callable: SandboxableCallable, /) -> SandboxReturnCallable:
    ...

@t.overload
def sandbox(*, model: LLMClient, code: str, fuel: t.Optional[int]=None) -> str:
    ...

def sandbox(callable: t.Optional[SandboxableCallable]=None, *, model: t.Optional[LLMClient]=None, code: t.Optional[str]=None, fuel: t.Optional[int]=None) -> t.Union[SandboxReturnCallable, str]:
    if code and callable:
        raise ValueError("Cannot provide both code and callable")
    
    if callable:
        if code or model:
            raise ValueError("Cannot provide code and client when passing a callable")
        code = "\n".join(inspect.getsource(callable).split("\n")[2:])
    else:
        if not code or not model:
            raise ValueError("Must provide either both code and client or a callable")


    code = textwrap.dedent(code)
    
    def inner(model: LLMClient) -> str:
        class Host(guest_imports.Host):
            def print(self, s: str) -> guest_types.Result[None, str]:
                print(s, end="", flush=True)
                return guest_types.Ok(None)
            def startstream(self, kwargs: str) -> guest_types.Result[str, str]:
                unpickled_kwargs = t.cast(LLMClientStreamArgs, jsonpickle.loads(kwargs))
                async def main(q_id, queue):
                    async for delta in model.stream(**unpickled_kwargs):
                        queue.put(jsonpickle.dumps(delta))
                    queue.put(Stop())
                q_id = run_coroutine_in_thread(main)
                return guest_types.Ok(q_id)
            def pollstream(self, qid: str) -> guest_types.Result[str, str]:
                if qid in queues:
                    result = queues[qid].get()
                    if isinstance(result, Exception):
                        return guest_types.Err(str(result))
                    elif isinstance(result, Stop):
                        del queues[qid]
                        return guest_types.Ok("[[STOP]]")
                    else:
                        return guest_types.Ok(str(result))
                else:
                    return guest_types.Ok("[[STOP]]")
        config = Config()
        if fuel:
            config.consume_fuel = True
        engine = Engine(config=config)
        store = Store(engine=engine)
        if fuel:
            store.set_fuel(fuel)
        root = Root(store, RootImports(host=Host()))
                
        return root.run(store, code)
        
    
    if callable:
        # If a callable is provided, return another callable
        return inner
    else:
        # If code is provided, run it directly
        if not model:
            raise ValueError("Argument 'client' must be provided when passing argument 'code'")

        return inner(model)