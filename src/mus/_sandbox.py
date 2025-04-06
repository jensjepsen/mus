import extism
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

@extism.host_fn(name="print", namespace="host")
def print_fn(msg: str):
    print(msg, end="", flush=True)

@extism.host_fn(name="poll_stream", namespace="host")
def poll_stream(q_id: str) -> str:
    if q_id in queues:
        result = queues[q_id].get()
        if isinstance(result, Exception):
            raise result
        elif isinstance(result, Stop):
            del queues[q_id]
            return "[[STOP]]"
        else:
            return str(result)
    else:
        return "[[STOP]]"

class SandboxableCallable(t.Protocol):
    async def __call__(self, model: LLMClient) -> None:
        ...

class SandboxReturnCallable(t.Protocol):
    def __call__(self, model: LLMClient) -> None:
        ...

@t.overload
def sandbox(callable: SandboxableCallable, /) -> SandboxReturnCallable:
    ...

@t.overload
def sandbox(*, model: LLMClient, code: str) -> None:
    ...

def sandbox(callable: t.Optional[SandboxableCallable]=None, *, model: t.Optional[LLMClient]=None, code: t.Optional[str]=None) -> t.Optional[SandboxReturnCallable]:
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
    
    def inner(model: LLMClient):
        @extism.host_fn(name="stream", namespace="host")
        def stream(kwargs: str) -> str:
            unpickled_kwargs = t.cast(LLMClientStreamArgs, jsonpickle.loads(kwargs)) # by design, this should be a valid LLMClientStreamArgs

            async def main(q_id, queue):
                async for delta in model.stream(**unpickled_kwargs):
                    queue.put(jsonpickle.dumps(delta))
                queue.put(Stop())
            q_id = run_coroutine_in_thread(main)
            return q_id
    
        guest_path = os.path.join(os.path.dirname(__file__), "guest.wasm")
        if not os.path.exists(guest_path):
            raise FileNotFoundError(f"Guest WASM file not found at {guest_path}")
        with extism.Plugin(guest_path, wasi=True) as plugin:
            plugin.call("run", code)
    
    if callable:
        # If a callable is provided, return another callable
        return inner
    else:
        # If code is provided, run it directly
        if not model:
            raise ValueError("Argument 'client' must be provided when passing argument 'code'")

        return inner(model)