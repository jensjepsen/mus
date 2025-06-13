import typing as t

class FillableType(t.Protocol):
    __annotations__: t.Dict[str, t.Any]
    __doc__: str
    __name__: str

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        ...