import typing as t

class FillableTypeBase(t.Protocol):
    __annotations__: t.Dict[str, t.Any]
    __doc__: str
    
    def __init__(self, *args: t.Any, **kwds: t.Any) -> None:
        ...

class FillableTypeDict(FillableTypeBase, t.Protocol):
    __doc__: str # have to redefine to ensure it is not None

    def __getitem__(self, key: str, /) -> object:
        ...

class FillableTypeDataclass(FillableTypeBase, t.Protocol):
    __doc__: str # have to redefine to ensure it is not None
    def __getattribute__(self, name: str, /) -> t.Any:
        ...


FillableType = t.Union[FillableTypeDict, FillableTypeDataclass]
