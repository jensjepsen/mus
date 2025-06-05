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
import functools

class Stop:
    pass

def run_coroutine_in_thread(coroutine_func, *args, **kwargs):
    q_id = str(uuid.uuid4())
    q = queue.Queue()
    
    def thread_target():
        try:
            # asyncio.run creates a new event loop, runs the coroutine, and closes the loop
            asyncio.run(coroutine_func(*args, **kwargs, q_id=q_id, queue=q))
        except Exception as e:
            q.put(e)
        finally:
            q.put(Stop())
    
    # using deamon=True to ensure the thread exits when the main program exits
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return q_id, q

class SandboxSharedKwargs(t.TypedDict, total=False):
    fuel: t.Optional[int]
    stdout: t.Optional[bool]
    stdin: t.Optional[bool]

class SandboxableCallableWithModel(t.Protocol):
    async def __call__(self, model: LLMClient) -> None:
        ...
    
class SandboxableCallableWithoutModel(t.Protocol):
    async def __call__(self) -> None:
        ...

SandboxableCallable = t.Union[SandboxableCallableWithModel, SandboxableCallableWithoutModel]

class SandboxReturnCallable(t.Protocol):
    async def __call__(self, model: t.Optional[LLMClient]=None) -> str:
        ...

class SandboxDecorator(t.Protocol):
    def __call__(self, callable: SandboxableCallable, /) -> SandboxReturnCallable:
        ...

@t.overload
def sandbox(callable: SandboxableCallable, /) -> SandboxReturnCallable:
    ...

@t.overload
def sandbox(**kwargs: t.Unpack[SandboxSharedKwargs]) -> SandboxDecorator:
    ...

@t.overload
def sandbox(*, model: t.Optional[LLMClient], code: str, **kwargs: t.Unpack[SandboxSharedKwargs]) -> t.Awaitable[str]:
    ...

@t.overload
def sandbox(*, code: str, **kwargs: t.Unpack[SandboxSharedKwargs]) -> t.Awaitable[str]:
    ...

def callable_to_code(callable: SandboxableCallable) -> str:
    """Convert a callable to its source code."""
    return "\n".join(inspect.getsource(callable).split("\n")[2:])

def sandbox(callable: t.Optional[SandboxableCallable]=None, *, model: t.Optional[LLMClient]=None, code: t.Optional[str]=None, **outer_kwargs: t.Unpack[SandboxSharedKwargs]) -> t.Union[SandboxReturnCallable, t.Awaitable[str], SandboxDecorator]:
    if code and callable:
        raise ValueError("Cannot provide both code and callable")
    
    if callable:
        if code:
            raise ValueError("Cannot provide code when passing a callable")
        code = callable_to_code(callable)

    def wrapper(callable: t.Optional[SandboxableCallable]=None) -> SandboxReturnCallable:
        nonlocal code
        if not code:
            if callable:
                code = callable_to_code(callable)
            else:
                raise ValueError("Must provide either code or a callable")
        code = textwrap.dedent(code)

        async def inner(model: t.Optional[LLMClient]=None):
            nonlocal code
            kwargs = outer_kwargs

            queues = {}

            if not code:
                raise ValueError("No code provided to run in the sandbox")
            class Host(guest_imports.Host):
                def print(self, s: str) -> guest_types.Result[None, str]:
                    if not kwargs.get("stdout", False):
                        return guest_types.Err("stdout is disabled")
                    print(s, end="", flush=True)
                    return guest_types.Ok(None)
                
                def input(self) -> guest_types.Result[str, str]:
                    if not kwargs.get("stdin", False):
                        return guest_types.Err("stdin is disabled")
                    try:
                        return guest_types.Ok(input())
                    except EOFError:
                        print("EOFError: No input provided")
                        return guest_types.Err("EOFError: No input provided")
                
                def startstream(self, kwargs: str) -> guest_types.Result[str, str]:
                    if not model:
                        return guest_types.Err("No model provided")
                    
                    unpickled_kwargs = t.cast(LLMClientStreamArgs, jsonpickle.loads(kwargs))

                    async def main(q_id, queue):
                        try:
                            async for delta in model.stream(**unpickled_kwargs): # type: ignore # we know model is an LLMClient, since we check it above
                                queue.put(jsonpickle.dumps(delta))
                        except Exception as e:
                            queue.put(e)
                        finally:
                            queue.put(Stop())
                    q_id, q = run_coroutine_in_thread(main)
                    queues[q_id] = q
                    return guest_types.Ok(q_id)
                
                def pollstream(self, qid: str) -> guest_types.Result[str, str]:
                    if qid in queues:
                        result = queues[qid].get()
                        print(result)
                        if isinstance(result, Exception):
                            return guest_types.Err("Exception: " + str(result))
                        elif isinstance(result, Stop):
                            del queues[qid]
                            return guest_types.Ok("[[STOP]]")
                        else:
                            return guest_types.Ok(str(result))
                    else:
                        return guest_types.Ok("[[STOP]]")
            config = Config()
            
            fuel = kwargs.get("fuel")
            if fuel:
                config.consume_fuel = True
            engine = Engine(config=config)
            store = Store(engine=engine)
            if fuel:
                store.set_fuel(fuel)
            root = Root(store, RootImports(host=Host()))
            
            return root.run(store, code)

        if callable:
            return functools.wraps(callable)(inner) # type: ignore # we know callable is a SandboxableCallable
        else:
            return inner
    
    if not code and not callable:
        # If no code or callable is provided, return a decorator
        return wrapper
    elif callable:
        # If a callable is provided, return a callable that runs it
        return wrapper(callable)
    else: 
        # If code is provided, run it directly
        return wrapper()(model=model)
    
