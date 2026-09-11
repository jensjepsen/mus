"""Exceptions must survive being carried across a process boundary.

A durable run checkpoints a failed step by serialising the exception, and
re-raises it from that record when the workflow is recovered in a fresh
process. Anything that cannot be reconstructed makes the recovery fail inside
the deserialiser, before any mus code runs -- so a run that survived a
transient provider error by retrying becomes unrecoverable if it later crashes.

The same applies anywhere an exception crosses processes: multiprocessing,
a task queue, a worker pool.
"""

import inspect
import pickle
import typing as t

import pytest

import mus.llm.exceptions as exceptions
from mus.llm.exceptions import LLMException, LLMRateLimitException
from mus.llm.types import StopReason


def _exception_classes() -> t.List[type]:
    return [
        cls
        for cls in vars(exceptions).values()
        if inspect.isclass(cls)
        and issubclass(cls, BaseException)
        and cls.__module__ == exceptions.__name__
    ]


def _make(cls: type) -> BaseException:
    """Build an instance, filling whatever the subclass requires.

    Built from the signature rather than a fixed argument list, so a subclass
    that adds a required field is covered the day it is written instead of
    silently falling back to a shape that does not exercise it.
    """
    supplied = {
        "provider": "stub",
        "stop_reason": StopReason(kind="max_tokens", raw="max_tokens"),
        "history": [],
    }
    params = inspect.signature(cls.__init__).parameters
    kwargs = {
        name: supplied[name]
        for name, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
        and name in supplied
    }
    missing = [
        name
        for name, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
        and name not in supplied
    ]
    assert not missing, f"{cls.__name__} needs {missing}; add them to _make"
    return cls("boom", **kwargs)


@pytest.mark.parametrize(
    "cls", _exception_classes(), ids=lambda c: c.__name__
)
def test_exception_survives_a_pickle_round_trip(cls):
    """``provider`` is keyword-only, so it is absent from ``args``.

    Pickle reconstructs an exception by calling ``cls(*args)``, which cannot
    supply a keyword-only parameter:

        TypeError: LLMException.__init__() missing 1 required keyword-only
        argument: 'provider'
    """
    original = _make(cls)

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is cls
    assert str(restored) == "boom"
    if isinstance(original, LLMException):
        assert restored.provider == original.provider


def test_pickling_keeps_subclass_specific_fields():
    """Fields a subclass adds must survive too, not just the base ones."""
    original = LLMRateLimitException(
        "slow down", provider="stub", status_code=429, retry_after=12.5
    )

    restored = pickle.loads(pickle.dumps(original))

    assert restored.retry_after == 12.5
    assert restored.status_code == 429
    assert restored.provider == "stub"
