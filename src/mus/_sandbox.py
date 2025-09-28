import asyncio
import threading
import queue
import uuid
import typing as t
import inspect
from .llm.llm import LLM
from .llm.types import LLMClientStreamArgs, ToolCallableType
from .functions import ToolCallable
from concurrent.futures import ThreadPoolExecutor
import mus.functions
import textwrap
from .guest.bindings import imports as guest_imports
from .guest.bindings import types as guest_types
from .guest.bindings import Root, RootImports
from wasmtime import Store, Engine, Config
import functools
import json
import dataclasses as dc
from .converters.delta import delta_converter

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

LLMs = dict[str, LLM]

class SandboxContext(t.TypedDict, total=False):
    llms: t.NotRequired[LLMs]
    functions: t.NotRequired[list[t.Callable[..., t.Any]]]
    tools: t.NotRequired[list[t.Union[ToolCallableType, ToolCallable]]]

class SandboxSharedKwargs(SandboxContext, total=False):
    fuel: t.Optional[int]
    stdout: t.Optional[bool]
    stdin: t.Optional[bool]
    
class SandboxableCallable(t.Protocol):
    async def __call__(self) -> None:
        ...

class SandboxReturnCallable(t.Protocol):
    async def __call__(self, **inputs: t.Optional[t.Any]) -> str:
        ...

def callable_to_code(callable: SandboxableCallable) -> str:
    """Convert a callable to its source code."""
    return "\n".join(inspect.getsource(callable).split("\n")[2:])

def func_params_to_dataclass(func: t.Callable):
    """Convert a function's annotations to a dataclass."""
    annotations = t.get_type_hints(func)
    fields = [(name, typ, dc.field(default=None)) for name, typ in annotations.items() if name != 'return']
    return dc.make_dataclass(f"{func.__name__.capitalize()}Params", fields)

class Empty(t.TypedDict):
    pass

def sandbox(**outer_kwargs: t.Unpack[SandboxSharedKwargs]):
    config = Config()
    kwargs = outer_kwargs
    fuel = kwargs.get("fuel")
    if fuel:
        config.consume_fuel = True
    engine = Engine(config=config)
    store = Store(engine=engine)
    if fuel:
        store.set_fuel(fuel)
    
    queues = {}

    llms = kwargs.get("llms", {})
            
    functions = {
        func.__name__: func
        for func in
        kwargs.get("functions", {})
    }

    tools = mus.functions.parse_tools(kwargs.get("tools", []))

    tool_map = {
        **{
            func.schema["name"]: func.function
            for func in tools
        },
        **{
            name: func
            for name, func in functions.items()
        }
    }
    
    function_schemas = {
        func.schema["name"]: mus.functions.remove_annotations(func.schema)
        for func in tools
    }

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
            except EOFError as e:
                print("EOFError: No input provided")
                raise e
        
        def startstream(self, llm: str, kwargs: str) -> guest_types.Result[str, str]:
            if llm not in llms:
                raise KeyError(f"LLM '{llm}' not found in context")

            unpickled_kwargs = delta_converter.structure(json.loads(kwargs), LLMClientStreamArgs[Empty, str])
            async def main(q_id, queue):
                try:
                    async for delta in llms[llm].stream(**unpickled_kwargs): # type: ignore # we know model is an LLMClient, since we check it above
                        queue.put(json.dumps(delta_converter.unstructure(delta)))
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
                    raise result
                elif isinstance(result, Stop):
                    del queues[qid]
                    return guest_types.Ok("[[STOP]]")
                else:
                    payload = str(result)
                    return guest_types.Ok(payload)
            else:
                return guest_types.Ok("[[STOP]]")
        def runfunction(self, name: str, inputs: str) -> guest_types.Result[str, str]:
            if name not in tool_map:
                raise KeyError(f"Function '{name}' not found in context")
            params = func_params_to_dataclass(tool_map[name])
            
            try:
                inputs_dc = delta_converter.structure(json.loads(inputs), params)
                inputs_dict = inputs_dc.__dict__
                # TODO: Validate inputs against function schema
                
                result = run_in_new_loop(tool_map[name](**inputs_dict))
                return guest_types.Ok(json.dumps(result))
            except Exception as e:
                raise e 
    root = Root(store, RootImports(host=Host()))

    @t.overload
    def wrapper(callable_or_code: SandboxableCallable) -> SandboxReturnCallable: ...
    @t.overload
    def wrapper(callable_or_code: str, **inputs: t.Optional[t.Any]) -> t.Awaitable[str]: ...
    def wrapper(callable_or_code: t.Union[SandboxableCallable, str], **inputs: t.Optional[t.Any]) -> t.Union[SandboxReturnCallable, t.Awaitable[str]]:
        async def inner(code: str, **inputs: t.Optional[t.Any]) -> str:
            if not code:
                raise ValueError("No code provided to run in the sandbox")
            code = textwrap.dedent(code)
            serialized_inputs = json.dumps(delta_converter.unstructure(inputs))
            result = json.loads(root.run(store, code, serialized_inputs, list(llms.keys()), json.dumps(function_schemas), functions=list(functions.keys())))
            if result.get("status") == "error":
                raise RuntimeError(result.get("message", "Unknown error in sandbox"))
            return result.get("message", "No message returned from sandbox")

        if callable(callable_or_code):
            inner_with_code = functools.partial(inner, callable_to_code(callable_or_code))
            return functools.wraps(callable_or_code)(inner_with_code)
        elif isinstance(callable_or_code, str):
            return inner(callable_or_code, **inputs)
        else:
            raise ValueError("Must provide either code or a callable or code string")
            
    return wrapper
