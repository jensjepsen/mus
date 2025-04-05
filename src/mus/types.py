import typing as t

class DataClass(t.Protocol):
    def __init__(self, *args, **kwargs):
        ...
    
    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        ...
    
    def __dataclass_fields__(self) -> t.Dict[str, t.Any]:
        ...

