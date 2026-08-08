import typing as t


class LLMException(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: t.Optional[int] = None,
        request_id: t.Optional[str] = None,
        raw_response: t.Optional[object] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.raw_response = raw_response


class LLMAuthenticationException(LLMException):
    """Raised when authentication fails (bad credentials, missing API key, etc.)."""

    pass


class LLMRateLimitException(LLMException):
    """Raised when the provider rate-limits or quota-limits the request."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: t.Optional[int] = None,
        request_id: t.Optional[str] = None,
        raw_response: t.Optional[object] = None,
        retry_after: t.Optional[float] = None,
    ):
        super().__init__(
            message,
            provider=provider,
            status_code=status_code,
            request_id=request_id,
            raw_response=raw_response,
        )
        self.retry_after = retry_after


class LLMConnectionException(LLMException):
    """Raised when the provider cannot be reached."""

    pass


class LLMTimeoutException(LLMException):
    """Raised when the request times out."""

    pass


class LLMBadRequestException(LLMException):
    """Raised when the request is invalid (bad params, validation errors)."""

    pass


class LLMContextLengthExceededException(LLMBadRequestException):
    """Raised when the request exceeds the model's context window.

    A subclass of ``LLMBadRequestException`` so existing handlers keep catching
    it. Distinguished so an error-recovery hook can recognise it and retry with
    a smaller history.
    """

    pass


class LLMServerException(LLMException):
    """Raised on provider-side 5xx / internal errors."""

    pass


class LLMNotFoundException(LLMException):
    """Raised when the model or resource is not found."""

    pass


class LLMModelException(LLMException):
    """Raised on model-specific failures (overloaded, not ready, etc.)."""

    pass


class LLMToolParseException(LLMException):
    """Raised when the model returns malformed JSON for a tool call."""

    pass


class LLMCachingException(LLMException):
    """Raised when a request uses prompt-caching features the model doesn't support.

    For example, sending a ``CachePoint`` to a Bedrock model that predates prompt
    caching. The underlying provider error may surface under a different code
    (Bedrock reports it as ``AccessDeniedException``); this exception normalises
    it so it isn't mistaken for an authentication failure.
    """

    pass


# Machine-readable error codes that unambiguously mean context-window overflow.
# Prefer these over message matching: they're stable across message wording and
# localisation. Only some providers expose one (e.g. OpenAI's
# ``context_length_exceeded``); Bedrock/Anthropic/Google report overflow as a
# generic bad-request with no distinct code, so we fall back to the message.
_CONTEXT_LENGTH_CODES: t.FrozenSet[str] = frozenset({"context_length_exceeded"})

# Substrings that identify a context-window / input-too-long error when no
# structured code is available. Best-effort by necessity — kept deliberately
# *specific* to overflow so we don't misfire on unrelated validation errors
# (e.g. a bad ``max_tokens`` or ``temperature``). A false positive is the costly
# case: the recovery hook would trim-and-retry a request that can never succeed.
_CONTEXT_LENGTH_MARKERS: t.Tuple[str, ...] = (
    "input is too long",
    "prompt is too long",
    "too many tokens",
    "maximum context length",
    "context window",
    "context_length_exceeded",
)


def is_context_length_error(
    message: str, *, code: t.Optional[str] = None
) -> bool:
    """True if this looks like a context-window overflow (any provider).

    ``code`` is a provider's machine-readable error code when available and is
    the authoritative signal; ``message`` matching is the best-effort fallback
    for providers that don't expose one.
    """
    if code is not None and code in _CONTEXT_LENGTH_CODES:
        return True
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS)
