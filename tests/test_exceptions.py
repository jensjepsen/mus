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
import threading
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


def _unpicklable_payloads():
    """Objects a provider SDK realistically attaches to an error.

    The stdlib lock stands in for the general case and is always available. The
    multidict proxy is the specific thing that bites: Google's ``raw_response``
    is an ``aiohttp.ClientResponse``, whose headers are a
    ``CIMultiDictProxy``. Verified against a live Google 404, which fails with
    "TypeError: can't pickle multidict._multidict.CIMultiDictProxy objects".
    """
    payloads = [pytest.param(threading.Lock(), id="thread-lock")]
    try:
        import multidict
    except ImportError:  # pragma: no cover - only with the google extra
        return payloads
    payloads.append(
        pytest.param(
            multidict.CIMultiDictProxy(multidict.CIMultiDict(server="gvs")),
            id="aiohttp-headers",
        )
    )
    return payloads


@pytest.mark.parametrize("payload", _unpicklable_payloads())
def test_exception_pickles_despite_an_unpicklable_raw_response(payload):
    """``raw_response`` carries whatever the provider SDK raised with.

    It is typed ``Optional[object]`` precisely because mus does not control it,
    and a live response object is routinely unpicklable -- it holds headers,
    sockets, a connection. Pickling the exception must not depend on it: a
    durable run serialises a failed step's exception and rebuilds it when
    recovering in another process, so an unpicklable payload makes the whole
    error unrecoverable, and the diagnostics that *are* portable -- provider,
    status code, message -- are lost with it.
    """
    exc = LLMException(
        "upstream exploded", provider="google", status_code=503,
        request_id="req-42", raw_response=payload,
    )

    restored = pickle.loads(pickle.dumps(exc))

    assert type(restored) is LLMException
    assert str(restored) == "upstream exploded"
    assert restored.provider == "google"
    assert restored.status_code == 503
    assert restored.request_id == "req-42"


def test_a_picklable_raw_response_is_kept():
    """Degrading is a fallback, not the rule -- most SDKs pickle fine.

    openai, anthropic and mistral all store an ``httpx.Response``, which
    survives a round trip even mid-stream. Dropping it unconditionally would
    throw away a useful diagnostic on every provider to accommodate one.
    """
    exc = LLMException(
        "boom", provider="openai", raw_response={"status": 500, "body": "nope"}
    )

    restored = pickle.loads(pickle.dumps(exc))

    assert restored.raw_response == {"status": 500, "body": "nope"}


def test_an_unpicklable_raw_response_degrades_without_leaking_it():
    """What is dropped should be named, but not quoted.

    A response ``repr`` can include headers, and headers carry credentials --
    writing those into a durable checkpoint would be worse than losing the
    object. The type name says what went missing and nothing else.
    """
    secret = {"authorization": "Bearer sk-do-not-persist"}

    class Live:
        def __init__(self):
            self.lock = threading.Lock()  # makes it unpicklable
            self.headers = secret

        def __repr__(self):
            return f"<Live headers={self.headers}>"

    exc = LLMException("boom", provider="google", raw_response=Live())

    restored = pickle.loads(pickle.dumps(exc))

    assert "unpicklable" in restored.raw_response
    assert "Live" in restored.raw_response
    assert "sk-do-not-persist" not in restored.raw_response
