import typing as t

if t.TYPE_CHECKING:
    from .main import Mus

InterpretableCallableWrappedParams = t.ParamSpec("InterpretableCallableWrappedParams")

class InterpretableCallable(t.Protocol, t.Generic[InterpretableCallableWrappedParams]):
    def __call__(self, mus: "Mus", *args: InterpretableCallableWrappedParams.args, **kwargs: InterpretableCallableWrappedParams.kwargs) -> t.Optional[t.Any]:
        ...

class DataClass(t.Protocol):
    def __init__(self, *args, **kwargs):
        ...
    
    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        ...
    
    def __dataclass_fields__(self) -> t.Dict[str, t.Any]:
        ...

