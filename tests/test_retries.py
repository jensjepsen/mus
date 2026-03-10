import pytest
from unittest.mock import AsyncMock, patch

from mus import Bot, Delta, DeltaText, DeltaStreamReset, RetryPolicy
from mus.llm.llm import IterableResult, _compute_backoff
from mus.llm.types import LLM, Usage, DeltaHistory
from mus.llm.exceptions import (
    LLMRateLimitException,
    LLMServerException,
    LLMTimeoutException,
    LLMConnectionException,
    LLMModelException,
    LLMAuthenticationException,
    LLMBadRequestException,
    LLMNotFoundException,
)

import typing as t


class StreamArgs(t.TypedDict, total=False):
    pass


class FailThenSucceedLLM(LLM[StreamArgs, str, None]):
    """Mock LLM that fails N times then succeeds."""

    def __init__(
        self,
        *,
        fail_count: int,
        exception: Exception,
        responses: list[Delta],
    ):
        self.fail_count = fail_count
        self.exception = exception
        self.responses = responses
        self.call_count = 0

    async def stream(self, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.exception
        for r in self.responses:
            yield r


class FailMidStreamLLM(LLM[StreamArgs, str, None]):
    """Mock LLM that yields some deltas then fails, succeeds on retry."""

    def __init__(
        self,
        *,
        fail_count: int,
        exception: Exception,
        partial_responses: list[Delta],
        full_responses: list[Delta],
    ):
        self.fail_count = fail_count
        self.exception = exception
        self.partial_responses = partial_responses
        self.full_responses = full_responses
        self.call_count = 0

    async def stream(self, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            for r in self.partial_responses:
                yield r
            raise self.exception
        for r in self.full_responses:
            yield r


class AlwaysFailLLM(LLM[StreamArgs, str, None]):
    """Mock LLM that always raises."""

    def __init__(self, exception: Exception):
        self.exception = exception
        self.call_count = 0

    async def stream(self, **kwargs):
        self.call_count += 1
        raise self.exception
        yield  # make it an async generator


# --- _compute_backoff tests ---


def test_compute_backoff_uses_retry_after():
    policy = RetryPolicy()
    assert _compute_backoff(0, policy, retry_after=5.0) == 5.0


def test_compute_backoff_exponential():
    policy = RetryPolicy(initial_backoff=1.0, backoff_multiplier=2.0, jitter=0.0)
    assert _compute_backoff(0, policy) == 1.0
    assert _compute_backoff(1, policy) == 2.0
    assert _compute_backoff(2, policy) == 4.0


def test_compute_backoff_respects_max():
    policy = RetryPolicy(
        initial_backoff=1.0, backoff_multiplier=10.0, max_backoff=5.0, jitter=0.0
    )
    assert _compute_backoff(2, policy) == 5.0


def test_compute_backoff_jitter_range():
    policy = RetryPolicy(initial_backoff=10.0, backoff_multiplier=1.0, jitter=0.5)
    values = {_compute_backoff(0, policy) for _ in range(100)}
    assert min(values) >= 5.0
    assert max(values) <= 15.0


# --- Transport retry tests ---


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    model = FailThenSucceedLLM(
        fail_count=1,
        exception=LLMServerException("server error", provider="test", status_code=500),
        responses=[Delta(content=DeltaText(data="Hello"))],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=0.0, jitter=0.0),
    )

    deltas = []
    async for msg in bot.query("hi"):
        deltas.append(msg)

    assert model.call_count == 2
    # Should have: DeltaStreamReset, DeltaText("Hello"), DeltaHistory
    reset_deltas = [d for d in deltas if isinstance(d.content, DeltaStreamReset)]
    text_deltas = [d for d in deltas if isinstance(d.content, DeltaText)]
    assert len(reset_deltas) == 1
    assert reset_deltas[0].content.attempt == 1
    assert len(text_deltas) == 1
    assert text_deltas[0].content.data == "Hello"


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    exc = LLMConnectionException("connection failed", provider="test")
    model = AlwaysFailLLM(exc)
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(
            max_transport_retries=2, initial_backoff=0.0, jitter=0.0
        ),
    )

    with pytest.raises(LLMConnectionException, match="connection failed"):
        async for _ in bot.query("hi"):
            pass

    assert model.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_class",
    [
        LLMRateLimitException,
        LLMServerException,
        LLMTimeoutException,
        LLMConnectionException,
        LLMModelException,
    ],
)
async def test_all_transient_exceptions_are_retried(exception_class):
    kwargs = {"message": "fail", "provider": "test"}
    if exception_class == LLMRateLimitException:
        kwargs["retry_after"] = None
    exc = exception_class(**kwargs)
    model = FailThenSucceedLLM(
        fail_count=1,
        exception=exc,
        responses=[Delta(content=DeltaText(data="ok"))],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=0.0, jitter=0.0),
    )

    text = []
    async for msg in bot.query("hi"):
        if isinstance(msg.content, DeltaText):
            text.append(msg.content.data)

    assert model.call_count == 2
    assert text == ["ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_class",
    [LLMAuthenticationException, LLMBadRequestException, LLMNotFoundException],
)
async def test_non_retryable_exceptions_propagate_immediately(exception_class):
    exc = exception_class("nope", provider="test")
    model = AlwaysFailLLM(exc)
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(max_transport_retries=3),
    )

    with pytest.raises(exception_class):
        async for _ in bot.query("hi"):
            pass

    assert model.call_count == 1  # no retries


@pytest.mark.asyncio
async def test_rate_limit_respects_retry_after():
    exc = LLMRateLimitException(
        "rate limited", provider="test", status_code=429, retry_after=0.1
    )
    model = FailThenSucceedLLM(
        fail_count=1,
        exception=exc,
        responses=[Delta(content=DeltaText(data="ok"))],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=999.0, jitter=0.0),
    )

    with patch("mus.llm.llm.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async for _ in bot.query("hi"):
            pass
        mock_sleep.assert_called_once_with(0.1)


# --- stream_id tests ---


@pytest.mark.asyncio
async def test_deltas_carry_stream_id():
    model = FailThenSucceedLLM(
        fail_count=0,
        exception=None,
        responses=[
            Delta(content=DeltaText(data="Hello")),
            Delta(content=DeltaText(data=" world")),
        ],
    )
    bot = Bot(prompt="test", model=model)

    stream_ids = set()
    async for msg in bot.query("hi"):
        if isinstance(msg.content, DeltaText):
            assert msg.stream_id is not None
            stream_ids.add(msg.stream_id)

    # All deltas from same stream call share a stream_id
    assert len(stream_ids) == 1


@pytest.mark.asyncio
async def test_retry_yields_new_stream_id():
    model = FailThenSucceedLLM(
        fail_count=1,
        exception=LLMServerException("fail", provider="test", status_code=500),
        responses=[Delta(content=DeltaText(data="ok"))],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=0.0, jitter=0.0),
    )

    reset_stream_id = None
    new_stream_id = None
    async for msg in bot.query("hi"):
        if isinstance(msg.content, DeltaStreamReset):
            reset_stream_id = msg.content.stream_id
        elif isinstance(msg.content, DeltaText):
            new_stream_id = msg.stream_id

    # Reset points to the old stream_id, new deltas have a different one
    assert reset_stream_id is not None
    assert new_stream_id is not None
    assert reset_stream_id != new_stream_id


# --- Mid-stream failure tests ---


@pytest.mark.asyncio
async def test_mid_stream_failure_resets_and_retries():
    model = FailMidStreamLLM(
        fail_count=1,
        exception=LLMTimeoutException("timeout", provider="test"),
        partial_responses=[Delta(content=DeltaText(data="partial"))],
        full_responses=[Delta(content=DeltaText(data="complete"))],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=0.0, jitter=0.0),
    )

    deltas = []
    async for msg in bot.query("hi"):
        deltas.append(msg)

    assert model.call_count == 2

    # We should see: partial text, reset, complete text, history
    types = [type(d.content).__name__ for d in deltas]
    assert "DeltaStreamReset" in types

    # The reset should reference the stream_id of the partial deltas
    partial_id = next(
        d.stream_id for d in deltas if isinstance(d.content, DeltaText) and d.content.data == "partial"
    )
    reset = next(d for d in deltas if isinstance(d.content, DeltaStreamReset))
    assert reset.content.stream_id == partial_id


# --- IterableResult rollback tests ---


@pytest.mark.asyncio
async def test_iterable_result_rollback_on_reset():
    """IterableResult should discard text from the reset stream_id."""
    model = FailMidStreamLLM(
        fail_count=1,
        exception=LLMServerException("fail", provider="test", status_code=500),
        partial_responses=[
            Delta(content=DeltaText(data="bad"), usage=Usage(input_tokens=10, output_tokens=5)),
        ],
        full_responses=[
            Delta(content=DeltaText(data="good"), usage=Usage(input_tokens=20, output_tokens=10)),
        ],
    )
    bot = Bot(
        prompt="test",
        model=model,
        retry_policy=RetryPolicy(initial_backoff=0.0, jitter=0.0),
    )

    result = bot("hi")
    async for _ in result:
        pass

    # Text should only contain the successful stream
    assert result.total == "good"
    # Usage should include ALL calls (both failed and successful)
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 15


@pytest.mark.asyncio
async def test_iterable_result_no_rollback_without_reset():
    """Without retries, IterableResult behaves normally."""
    model = FailThenSucceedLLM(
        fail_count=0,
        exception=None,
        responses=[
            Delta(content=DeltaText(data="hello ")),
            Delta(content=DeltaText(data="world")),
        ],
    )
    bot = Bot(prompt="test", model=model)

    result = bot("hi")
    async for _ in result:
        pass

    assert result.total == "hello world"


# --- Default RetryPolicy ---


def test_default_retry_policy():
    policy = RetryPolicy()
    assert policy.max_transport_retries == 3
    assert policy.initial_backoff == 1.0
    assert policy.max_backoff == 60.0
    assert policy.backoff_multiplier == 2.0
    assert policy.jitter == 0.5


def test_bot_default_retry_policy():
    from mus.llm.mock_client import StubLLM

    model = StubLLM()
    bot = Bot(prompt="test", model=model)
    assert isinstance(bot.retry_policy, RetryPolicy)
    assert bot.retry_policy.max_transport_retries == 3
