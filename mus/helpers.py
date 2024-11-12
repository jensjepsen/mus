import json
from dataclasses import is_dataclass
import typing as t

def json_hook(obj: t.Any):
    if is_dataclass(obj):
        return obj.__dict__
    return obj

def json_dumps(obj: t.Dict[str, t.Any]):
    return json.dumps(obj, default=json_hook)