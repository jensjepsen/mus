import typing as t

class FillableType(t.Protocol):
    __annotations__: t.Dict[str, t.Any]
    __doc__: str
    
    def __init__(self, *args: t.Any, **kwds: t.Any) -> None:
        ...