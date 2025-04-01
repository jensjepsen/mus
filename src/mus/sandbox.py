import extism
from mus import BedrockLLM
import boto3
import asyncio
import jsonpickle
import threading
import queue
import uuid
import typing as t
import asyncio
import os
from .llm.llm import LLMClient

class Stop:
    pass

def run_coroutine_in_thread(coroutine_func, *args, **kwargs):
    """
    Run a coroutine in a thread using asyncio.run (Python 3.7+)
    """
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

def sandbox(client: LLMClient, code: str):

    @extism.host_fn(name="stream", namespace="host")
    def stream(kwargs: str) -> str:
        kwargs = jsonpickle.loads(kwargs)

        async def main(q_id, queue):
            async for delta in client.stream(**kwargs):
                queue.put(jsonpickle.dumps(delta))
            queue.put(Stop())
        q_id = run_coroutine_in_thread(main)
        return q_id

    
    guest_path = os.path.join(os.path.dirname(__file__), "guest.wasm")
    if not os.path.exists(guest_path):
        raise FileNotFoundError(f"Guest WASM file not found at {guest_path}")
    with extism.Plugin(guest_path, wasi=True) as plugin:
        plugin.call("run", code)