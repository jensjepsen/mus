"""Tests for the general error-recovery hook (retry with a modified history).

When an LLM call fails *pre-stream* (before any delta is yielded), mus offers the
failure to an optional ``error_recovery_hook``. The consumer returns a modified
history to re-issue with, or ``None`` to give up. The motivating case is a
context-length overflow (recover by trimming), but the hook is general: mus hands
it any ``LLMException`` that escapes the transport-retry loop and the consumer
decides, per error, whether/how to recover.
"""

import typing as t

import pytest

from mus import Bot, Delta, DeltaText, DeltaStreamReset, RetryPolicy
from mus.llm.types import LLM, History
from mus.llm.exceptions import (
    LLMContextLengthExceededException,
    LLMBadRequestException,
    LLMServerException,
    is_context_length_error,
)


# --- Overflow detection helper ---


@pytest.mark.parametrize(
    "message",
    [
        "Input is too long for requested model",
        "prompt is too long: 215334 tokens > 200000 maximum",
        "This model's maximum context length is 128000 tokens",
        "The request exceeds the model's context window",
    ],
)
def test_detects_overflow_messages(message):
    assert is_context_length_error(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # These are NOT overflows and must not misfire — they were the source of
        # brittleness (broad "exceeds the maximum" / token-count phrasing).
        "temperature exceeds the maximum allowed value",
        "number of images exceeds the maximum of 20",
        "stop_sequences: exceeds the maximum number of tokens per sequence",
        "invalid value for parameter 'top_p'",
        "",
    ],
)
def test_ignores_non_overflow_messages(message):
    assert is_context_length_error(message) is False


def test_structured_code_beats_message():
    # The machine code is authoritative even when the message says nothing.
    assert is_context_length_error(
        "generic failure", code="context_length_exceeded"
    ) is True
    # An unrelated code does not force a match.
    assert is_context_length_error("all good", code="rate_limit_exceeded") is False


class StreamArgs(t.TypedDict, total=False):
    pass


class ScriptedLLM(LLM[StreamArgs, str, None]):
    """Mock LLM driven by a per-call script.

    Each script entry is either ``("fail", exception)`` (raise pre-stream) or
    ``("stream", [deltas])`` (yield those deltas). The history each call was
    issued with is recorded in ``self.histories`` so tests can assert which
    history a re-issue used.
    """

    def __init__(self, script: list[tuple[str, object]]):
        self.script = script
        self.call_count = 0
        self.histories: list[object] = []

    async def stream(self, **kwargs):
        self.histories.append(kwargs.get("history"))
        idx = self.call_count
        self.call_count += 1
        kind, payload = self.script[idx] if idx < len(self.script) else self.script[-1]
        if kind == "fail":
            raise t.cast(Exception, payload)
        for d in t.cast(list, payload):
            yield d


class PartialThenFailLLM(LLM[StreamArgs, str, None]):
    """Yields a delta, then raises mid-stream — every call (never succeeds)."""

    def __init__(self, *, partial: list[Delta], exception: Exception):
        self.partial = partial
        self.exception = exception
        self.call_count = 0

    async def stream(self, **kwargs):
        self.call_count += 1
        for d in self.partial:
            yield d
        raise self.exception


def _overflow(msg: str = "Input is too long") -> LLMContextLengthExceededException:
    return LLMContextLengthExceededException(msg, provider="test", status_code=400)


@pytest.mark.asyncio
async def test_overflow_recovers_via_hook():
    """Pre-stream overflow → hook called once → re-issue with returned history."""
    recovered_history: History = [Delta(content=DeltaText(data="[trimmed]"))]
    model = ScriptedLLM(
        [
            ("fail", _overflow()),
            ("stream", [Delta(content=DeltaText(data="ok"))]),
        ]
    )

    calls: list[tuple] = []

    async def hook(history, error, attempt):
        calls.append((list(history), error, attempt))
        return recovered_history

    bot = Bot(prompt="test", model=model, error_recovery_hook=hook)

    deltas = []
    async for msg in bot.query("hi"):
        deltas.append(msg)

    # Hook invoked exactly once, with the failing error and attempt index 0.
    assert len(calls) == 1
    failed_history, error, attempt = calls[0]
    assert isinstance(error, LLMContextLengthExceededException)
    assert attempt == 0

    # The re-issue used the history the hook returned.
    assert model.call_count == 2
    assert model.histories[1] == recovered_history

    # The turn completed with the post-recovery text, exactly once (no dupes).
    text = [d.content.data for d in deltas if isinstance(d.content, DeltaText)]
    assert text == ["ok"]

    # A recovery reset was emitted so the UI can show "compacting context…".
    resets = [d for d in deltas if isinstance(d.content, DeltaStreamReset)]
    assert len(resets) == 1
    assert resets[0].content.reason == "error_recovery"
    assert resets[0].content.attempt == 1


@pytest.mark.asyncio
async def test_hook_returns_none_reraises():
    """Hook gives up (returns None) → mus propagates the original error."""
    model = ScriptedLLM([("fail", _overflow("Input is too long"))])

    async def hook(history, error, attempt):
        return None

    bot = Bot(prompt="test", model=model, error_recovery_hook=hook)

    with pytest.raises(LLMContextLengthExceededException, match="Input is too long"):
        async for _ in bot.query("hi"):
            pass

    # Hook was consulted once; no re-issue happened.
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_recovery_capped_by_max_recovery_attempts():
    """Model keeps failing → mus stops after max_recovery_attempts and re-raises."""
    # Always fails, regardless of the returned history.
    model = ScriptedLLM([("fail", _overflow())])

    call_count = {"n": 0}

    async def hook(history, error, attempt):
        call_count["n"] += 1
        # Always return *some* history so only the cap stops the loop.
        return [Delta(content=DeltaText(data=f"trim-{attempt}"))]

    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(max_recovery_attempts=3),
        error_recovery_hook=hook,
    )

    with pytest.raises(LLMContextLengthExceededException):
        async for _ in bot.query("hi"):
            pass

    # Hook called exactly max_recovery_attempts times (attempts 0,1,2), then the
    # cap forces a re-raise before a 4th recovery.
    assert call_count["n"] == 3
    # Initial issue + 3 recovery re-issues = 4 stream calls.
    assert model.call_count == 4


@pytest.mark.asyncio
async def test_no_hook_raises_immediately():
    """Without a hook, an overflow propagates immediately (today's behavior)."""
    model = ScriptedLLM([("fail", _overflow())])
    hook_calls = {"transform": 0}

    async def transform_history_hook(history):
        hook_calls["transform"] += 1
        return history

    bot = Bot(
        prompt="test",
        model=model,
        transform_history_hook=transform_history_hook,
    )

    with pytest.raises(LLMContextLengthExceededException):
        async for _ in bot.query("hi"):
            pass

    assert model.call_count == 1
    # transform_history_hook runs once for the normal first pass, and is NOT
    # re-invoked for recovery (there is no recovery without an error_recovery_hook).
    assert hook_calls["transform"] == 1


@pytest.mark.asyncio
async def test_mid_stream_error_is_not_recovered():
    """A delta was yielded before the error → the pre-stream guard blocks recovery."""
    model = PartialThenFailLLM(
        partial=[Delta(content=DeltaText(data="partial"))],
        exception=_overflow(),
    )

    calls = {"n": 0}

    async def hook(history, error, attempt):
        calls["n"] += 1
        return [Delta(content=DeltaText(data="trimmed"))]

    bot = Bot(
        prompt="test",
        model=model,
        # A non-transient error is not retried by the transport loop either.
        retry_policy=RetryPolicy(max_transport_retries=3, initial_backoff=0.0, jitter=0.0),
        error_recovery_hook=hook,
    )

    deltas = []
    with pytest.raises(LLMContextLengthExceededException):
        async for msg in bot.query("hi"):
            deltas.append(msg)

    # The hook was never consulted (mid-stream failure), and the partial delta was
    # yielded exactly once (no duplication from a phantom re-issue).
    assert calls["n"] == 0
    assert model.call_count == 1
    text = [d.content.data for d in deltas if isinstance(d.content, DeltaText)]
    assert text == ["partial"]


@pytest.mark.asyncio
async def test_hook_is_general_non_overflow_error_propagates():
    """The hook is offered any pre-stream LLMException; returning None propagates it."""
    model = ScriptedLLM([("fail", LLMBadRequestException("bad params", provider="test"))])

    seen: list[Exception] = []

    async def hook(history, error, attempt):
        # A realistic consumer dispatches on the error type and only recovers
        # from overflows; everything else falls through to None.
        seen.append(error)
        if isinstance(error, LLMContextLengthExceededException):
            return [Delta(content=DeltaText(data="trimmed"))]
        return None

    bot = Bot(prompt="test", model=model, error_recovery_hook=hook)

    with pytest.raises(LLMBadRequestException, match="bad params"):
        async for _ in bot.query("hi"):
            pass

    # The hook WAS offered the non-overflow error (confirming generality) but
    # chose not to recover it.
    assert len(seen) == 1
    assert isinstance(seen[0], LLMBadRequestException)
    assert not isinstance(seen[0], LLMContextLengthExceededException)
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_transient_exhaustion_offered_to_hook_then_propagates():
    """A transient error that exhausts its retries is still offered to the hook.

    The hook can't fix a rate-limit by trimming, so it returns None and the
    exhausted transient error propagates unchanged.
    """
    exc = LLMServerException("server melt", provider="test", status_code=500)
    model = ScriptedLLM([("fail", exc)])

    seen: list[Exception] = []

    async def hook(history, error, attempt):
        seen.append(error)
        return None  # not an overflow; nothing to trim

    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(max_transport_retries=2, initial_backoff=0.0, jitter=0.0),
        error_recovery_hook=hook,
    )

    with pytest.raises(LLMServerException, match="server melt"):
        async for _ in bot.query("hi"):
            pass

    # Transport loop tried initial + 2 retries = 3 stream calls before the
    # exhausted error reached the recovery layer.
    assert model.call_count == 3
    # The hook was offered the exhausted transient error exactly once.
    assert len(seen) == 1
    assert isinstance(seen[0], LLMServerException)
