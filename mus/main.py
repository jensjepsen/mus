
from anthropic import AnthropicBedrock
import functools
import pathlib
import typing as t
import inspect
import ast


from .interpreter import DSLInterpreter
from .llm import LLM, AnthropicLLM
from .llm.types import File
from .state import StateManager, StateType, State
from .functions import tool
from .types import InterpretableCallable, InterpretableCallableWrappedParams

def get_anthropic_client():
    return AnthropicLLM(client=AnthropicBedrock(aws_region="us-west-2"))


class Mus:
    def __init__(self, state: t.Optional[str]=None):
        self.state_manager = StateManager()
        self.client = get_anthropic_client()

        if state:
            self.state_manager.loads(state)
        
        self.llm = functools.partial(LLM, client=get_anthropic_client())
        self.tool = tool
        self.print = functools.partial(print, end="", flush=True)

        self.functions = {
            "llm": self.llm,
            "print": self.print,
            "range": range,
            "input": input,
            "File": File,
            "state": self.state,
            "sm": self.state_manager,
            "tool": self.tool
        }

        self.interpreter = DSLInterpreter(functions=self.functions, variables={"bob": self})

    def state(self, name: str, default_val: StateType=None) -> State[StateType]:
        return self.state_manager.init(name, default_val)

        

    def interpret(self, **kwargs):
        """Run an agent script or function"""
        
        functions = {
            **self.functions,
            **kwargs
        }

        self.interpreter.functions = functions

        def execute(target: InterpretableCallable[InterpretableCallableWrappedParams]) -> t.Callable[InterpretableCallableWrappedParams, t.Any]:
            func_source = inspect.getsource(target).split("\n", 1)[1]
            first_stmt = ast.parse(func_source).body[0]
            if isinstance(first_stmt, ast.FunctionDef):
                func_contents = "\n".join(ast.unparse(stmt) for stmt in first_stmt.body)
            else:
                raise ValueError("Can only interpret functions")
            def wrapper(*args: InterpretableCallableWrappedParams.args, **kwargs: InterpretableCallableWrappedParams.kwargs):
                return self.interpreter.run(func_contents)
            return wrapper
        
        return execute
    
    def run(self, code: str):
        self.interpreter.run(code)
    
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
    
def run_file(file: pathlib.Path, state_path: t.Optional[pathlib.Path]=None):
    """Run an agent script from a file"""
    if state_path:
        if state_path.exists():
            with open(state_path, "r") as f:
                state = f.read()
        else:
            print(f"State file not found: {state_path}")
            state = None
    else:
        state = None
    bob = Mus(state=state)
    with open(file, "r") as f:
        bob.run(f.read())
    if state_path:
        new_state = bob.state_manager.dumps()
        with open(state_path, "w") as f:
            f.write(new_state)    