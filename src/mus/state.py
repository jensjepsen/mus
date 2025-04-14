import typing as t
import jsonpickle

class Empty:
    ...

StateType = t.TypeVar("StateType")
class StateReference(t.Generic[StateType]):
    def __init__(self, val: StateType) -> None:
        self.val = val

    def __call__(self, new_val: t.Union[StateType, Empty]=Empty()) -> StateType:
        if not isinstance(new_val, Empty):
            self.val = new_val

        return self.val

    def to_dict(self) -> t.Dict[str, StateType]:
        return {"val": self.val}
    
    @staticmethod
    def from_dict(data: t.Dict[str, t.Any]) -> "StateReference":
        return StateReference(data["val"])

def encode_obj(obj: t.Any) -> t.Any:
    try:
        return obj.to_dict()
    except AttributeError:
        return obj

class State:
    def __init__(self) -> None:
        self.states = {}
        self.is_set = set()
    
    def init(self, name: str, default_val: StateType=None) -> StateReference[StateType]:
        """
        We check if a name is in states but not set by init, because it must then come from load, and should not be overwritten
        """
        if name in self.states and not name in self.is_set:
            
            state = self.states[name]
        else:
            val = default_val
            state = StateReference(val)

        self.states[name] = state
        self.is_set.add(name)
        return state
    
    def dumps(self, **dumps_kwargs: t.Any) -> str:
        result = jsonpickle.encode({name: state.to_dict() for name, state in self.states.items()}, **dumps_kwargs)
        if not result:
            raise ValueError("jsonpickle produced an empty result!")
        return t.cast(str, result)
    
    def loads(self, data: str, **loads_kwargs) -> None:
        decoded: t.Dict[str, t.Any] = t.cast(t.Dict[str, t.Any], jsonpickle.decode(data, **loads_kwargs))
        self.states = {name: StateReference.from_dict(val) for name, val in decoded.items()}