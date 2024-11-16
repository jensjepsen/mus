import typing as t
import jsonpickle

class Empty:
    ...

StateType = t.TypeVar("StateType")
class State(t.Generic[StateType]):
    def __init__(self, val: StateType) -> None:
        self.val = val

    def __call__(self, new_val: t.Union[StateType, Empty]=Empty()) -> StateType:
        if not isinstance(new_val, Empty):
            self.val = new_val

        return self.val

    def to_dict(self) -> t.Dict[str, StateType]:
        return {"val": self.val}
    
    @staticmethod
    def from_dict(data: t.Dict[str, t.Any]) -> "State":
        return State(data["val"])

def encode_obj(obj: t.Any) -> t.Any:
    try:
        return obj.to_dict()
    except AttributeError:
        return obj

class StateManager:
    def __init__(self) -> None:
        self.states = {}
        self.is_set = set()
    
    def init(self, name: str, default_val: StateType=None) -> State[StateType]:
        if name in self.states and not name in self.is_set:
            s = self.states[name]
        else:
            val = default_val
            s = State(val)
        
        self.states[name] = s
        self.is_set.add(name)
        return s
    
    def dumps(self, **dumps_kwargs: t.Any) -> str:
        return jsonpickle.encode({name: state.to_dict() for name, state in self.states.items()}, **dumps_kwargs)
    
    def loads(self, data: str, **loads_kwargs) -> None:
        decoded: t.Dict[str, t.Any] = jsonpickle.decode(data, **loads_kwargs)
        self.states = {name: State.from_dict(val) for name, val in decoded.items()}