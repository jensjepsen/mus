import asyncio
import jsonpickle
import threading
import queue
import uuid
import typing as t
import inspect
from .llm.llm import LLM
from .llm.types import LLMClientStreamArgs
import mus.functions
import textwrap
from .guest.bindings import imports as guest_imports
from .guest.bindings import types as guest_types
from .guest.bindings import Root, RootImports
from wasmtime import Store, Engine, Config
import functools
import json
import anyio
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

from concurrent.futures import ThreadPoolExecutor

def run_in_new_loop(coro):
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
            return result
        finally:
            loop.close()
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(_run)
        return future.result()

    

class SandboxSharedKwargs(t.TypedDict, total=False):
    fuel: t.Optional[int]
    stdout: t.Optional[bool]
    stdin: t.Optional[bool]

LLMs = dict[str, LLM]

SandboxableParams = t.ParamSpec("SandboxableParams")

class SandboxableCallable(t.Protocol[SandboxableParams]):
    async def __call__(self, *args: SandboxableParams.args, **kwargs: SandboxableParams.kwargs) -> None:
        ...

class SandboxReturnCallable(t.Protocol):
    async def __call__(self, **kwargs: t.Any) -> str:
        ...

class SandboxDecorator(t.Protocol):
    def __call__(self, callable: SandboxableCallable, /) -> SandboxReturnCallable:
        ...

@t.overload
def sandbox(callable: SandboxableCallable, /) -> SandboxReturnCallable:
    ...

@t.overload
def sandbox(*, code: str, **kwargs: t.Unpack[SandboxSharedKwargs]) -> SandboxReturnCallable:
    ...

@t.overload
def sandbox(**kwargs: t.Unpack[SandboxSharedKwargs]) -> SandboxDecorator:
    ...


def callable_to_code(callable: SandboxableCallable) -> str:
    """Convert a callable to its source code."""
    return "\n".join(inspect.getsource(callable).split("\n")[2:])

# TODO:
# Ideally the below would be able to take a list of models and functions,
# that are then available in the sandboxed code

SandboxContext = t.Union[LLM, t.Callable[..., t.Any]]

iscallable = callable

def sandbox(callable: t.Optional[SandboxableCallable]=None, *, code: t.Optional[str]=None, **outer_kwargs: t.Unpack[SandboxSharedKwargs]) -> t.Union[SandboxReturnCallable, t.Awaitable[str], SandboxDecorator]:
    if code and callable:
        raise ValueError("Cannot provide both code and callable")
    
    def wrapper(callable: t.Optional[SandboxableCallable]=None, code: t.Optional[str]=None) -> SandboxReturnCallable:
        if code and callable:
            raise ValueError("Cannot provide both code and callable")
    
        if not code:
            if callable:
                code = callable_to_code(callable)
            else:
                raise ValueError("Must provide either code or a callable")
        
        code = textwrap.dedent(code)
        async def inner(**context: SandboxContext):
            nonlocal code
            kwargs = outer_kwargs
            queues = {}

            llms = {
                name: llm
                for name, llm
                in context.items()
                if hasattr(llm, "stream")
            }

            functions = [
                func
                for name, func
                in context.items()
                if not hasattr(func, "stream") and iscallable(func)
            ]

            tools = mus.functions.parse_tools(functions)

            tool_map = {
                func.schema["name"]: func.function
                for func in tools
            }

            function_schemas = {
                func.schema["name"]: mus.functions.remove_annotations(func.schema)
                for func in tools
            }

            if not code:
                raise ValueError("No code provided to run in the sandbox")
            class Host(guest_imports.Host):
                def print(self, s: str) -> guest_types.Result[None, str]:
                    if not kwargs.get("stdout", False):
                        return guest_types.Ok(None)
                    print(s, end="", flush=True)
                    return guest_types.Ok(None)
                
                def input(self) -> guest_types.Result[str, str]:
                    if not kwargs.get("stdin", False):
                        return guest_types.Ok("")
                    try:
                        return guest_types.Ok(input())
                    except EOFError:
                        print("EOFError: No input provided")
                        return guest_types.Err("EOFError: No input provided")
                
                def startstream(self, llm: str, kwargs: str) -> guest_types.Result[str, str]:
                    if llm not in llms:
                        raise KeyError(f"LLM '{llm}' not found in context")

                    unpickled_kwargs = t.cast(LLMClientStreamArgs, jsonpickle.loads(kwargs))
                    async def main(q_id, queue):
                        try:
                            async for delta in llms[llm].stream(**unpickled_kwargs): # type: ignore # we know model is an LLMClient, since we check it above
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
                        if isinstance(result, Exception):
                            return guest_types.Err("Exception: " + str(result))
                        elif isinstance(result, Stop):
                            del queues[qid]
                            return guest_types.Ok("[[STOP]]")
                        else:
                            payload = str(result)
                            print("Sending payload", payload)
                            return guest_types.Ok(payload)
                    else:
                        return guest_types.Ok("[[STOP]]")
                def runfunction(self, name: str, inputs: str) -> guest_types.Result[str, str]:
                    if name not in function_schemas:
                        raise KeyError(f"Function '{name}' not found in context")
                    try:
                        inputs_dict = json.loads(inputs)
                        # TODO: Validate inputs against function schema
                        result = run_in_new_loop(tool_map[name](**inputs_dict))
                        return guest_types.Ok(json.dumps(result))
                    except Exception as e:
                        raise e 
            config = Config()
            
            fuel = kwargs.get("fuel")
            if fuel:
                config.consume_fuel = True
            engine = Engine(config=config)
            store = Store(engine=engine)
            if fuel:
                store.set_fuel(fuel)
            root = Root(store, RootImports(host=Host()))
            
            result = json.loads(root.run(store, code, list(llms.keys()), json.dumps(function_schemas)))
            if result.get("status") == "error":
                raise RuntimeError(result.get("message", "Unknown error in sandbox"))
            return result.get("message", "No message returned from sandbox")

        if callable:
            return functools.wraps(callable)(inner) # type: ignore # we know callable is a SandboxableCallable
        else:
            return inner
    
    if not code and not callable:
        # If no code or callable is provided, return a decorator
        return wrapper
    elif callable:
        # If a callable is provided, return a callable that runs it
        return wrapper(callable=callable)
    else: 
        # If code is provided, run it directly
        return wrapper(code=code)
    
