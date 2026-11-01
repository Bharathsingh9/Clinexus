"""Typed LLM provider exceptions.

Each exception carries a machine-readable error code that maps directly
to the AgentExecutionRecord.error_code field.  No exception message,
repr, or str ever contains an API key, request body, or response body.
"""


class LLMProviderError(Exception):
    """Base class for all LLM provider errors."""

    error_code: str = "LLM_ERROR"

    def __init__(self, message: str) -> None:
        # Sanitise: never allow a key or body to leak through the message.
        super().__init__(message)


class LLMAuthError(LLMProviderError):
    """API key missing, invalid, or lacking permissions (HTTP 401/403)."""

    error_code = "LLM_AUTH_ERROR"


class LLMRateLimited(LLMProviderError):
    """Rate limit exceeded after all retries (HTTP 429)."""

    error_code = "LLM_RATE_LIMITED"


class LLMTimeout(LLMProviderError):
    """Request timed out after all retries."""

    error_code = "LLM_TIMEOUT"


class LLMInvalidResponse(LLMProviderError):
    """Response is unusable: empty candidates, safety block, truncation,
    non-JSON when JSON was requested, or a 400 from the API."""

    error_code = "LLM_INVALID_RESPONSE"


class LLMCallCapReached(LLMProviderError):
    """The per-run hard cap on LLM calls (EVAL_MAX_LLM_CALLS) was hit."""

    error_code = "LLM_CALL_CAP_REACHED"


class LLMUnavailable(LLMProviderError):
    """Connection failures after all retries (5xx, network errors)."""

    error_code = "LLM_UNAVAILABLE"
