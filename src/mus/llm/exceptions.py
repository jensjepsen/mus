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
