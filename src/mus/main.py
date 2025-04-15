
import functools
import pathlib
import typing as t
import inspect
import ast

from .llm import LLM
from .llm.types import File
from .state import StateReference, StateType, State
from .functions import tool

class Musa:
    def __init__(self):
        self.state_manager = State()
        
        self.tool = tool
        self.print = functools.partial(print, end="", flush=True)

    def state(self, name: str, default_val: StateType=None) -> StateReference[StateType]:
        return self.state_manager.init(name, default_val)
    
    def dumps(self, **dumps_kwargs: t.Any) -> str:
        return self.state_manager.dumps(**dumps_kwargs)
    
    def loads(self, data: str, **loads_kwargs) -> None:
        self.state_manager.loads(data, **loads_kwargs)
    
    def dump(self, file: t.Union[pathlib.Path, str], **dumps_kwargs: t.Any):
        with open(file, "w") as f:
            f.write(self.dumps(**dumps_kwargs))

    def load(self, file: t.Union[pathlib.Path, str]):
        with open(file, "r") as f:
            self.loads(f.read())