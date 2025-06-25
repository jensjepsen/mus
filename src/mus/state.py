import typing as t
import pathlib
import attrs
import cattrs
import json

class Empty:
    ...

converter = cattrs.Converter()

StateType = t.TypeVar("StateType")
@attrs.define
class StateReference(t.Generic[StateType]):
    val: StateType
    
    def __call__(self, new_val: t.Union[StateType, Empty]=Empty()) -> StateType:
        if not isinstance(new_val, Empty):
            self.val = new_val

        return self.val

    def to_dict(self) -> t.Dict[str, StateType]:
        return {"val": self.val}
    
    @staticmethod
    def from_dict(data: t.Dict[str, t.Any]) -> "StateReference":
        return StateReference(data["val"])

def structure_state_reference(
    state_ref: StateReference[StateType], 
    typ: t.Type[StateReference[StateType]]
) -> t.Dict[str, StateType]:
    return state_ref.to_dict()

converter.register_unstructure_hook(StateReference, lambda sf: sf.to_dict())
converter.register_structure_hook(StateReference, structure_state_reference)


StatesType = dict[str, StateReference]
StatesTypeAny = dict[str, StateReference[t.Any]]

class State:
    def __init__(self) -> None:
        self.states: StatesType  = {}
        self.is_set = set()
    
    def init(self, name: str, default_val: StateType=None) -> StateReference[StateType]:
        """
        We check if a name is in states but not set by init, because it must then come from load, and should not be overwritten
        """
        if name in self.states and name not in self.is_set:
            
            state = self.states[name]
        else:
            val = default_val
            state = StateReference(val)

        self.states[name] = state
        self.is_set.add(name)
        return state

    def __call__(self, name: str, default_val: StateType=None) -> StateReference[StateType]:
        return self.init(name, default_val)
    
    def dumps(self, **dumps_kwargs: t.Any) -> str:
        result = json.dumps(cattrs.unstructure(self.states), **dumps_kwargs)
        return result
    
    def dump(self, file: t.Union[pathlib.Path, str], **dumps_kwargs: t.Any) -> None:
        with open(file, "w") as f:
            f.write(self.dumps(**dumps_kwargs))
    
    def loads(self, data: str, **loads_kwargs) -> None:
        unstructured = json.loads(data, **loads_kwargs)
        self.states = cattrs.structure(unstructured, StatesTypeAny)
    
    def load(self, file: t.Union[pathlib.Path, str]) -> None:
        with open(file, "r") as f:
            self.loads(f.read())