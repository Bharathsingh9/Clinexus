"""API Error Definitions and Sanitization.

Specification Section 2, Section 7:
Error body structure: {error: {code, message, details}}
Codes:
- VALIDATION_ERROR (422)
- UNAUTHENTICATED (401)
- FORBIDDEN (403)
- NOT_FOUND (404)
- INVALID_STATE (409)
- CONFLICT (409)
- RATE_LIMITED (429)
- INTERNAL_ERROR (500)
- NO_ABSTENTION_DECISION (404)

Safety Invariant: Validation details must NOT echo submitted values (patient free text).
"""

from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: ErrorBody


class AppAPIException(HTTPException):
    """Base exception for structured API errors."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.details = details


class ValidationAPIError(AppAPIException):
    def __init__(self, message: str = "Validation failed.", details: Optional[Any] = None):
        super().__init__(status.HTTP_422_UNPROCESSABLE_ENTITY, "VALIDATION_ERROR", message, details)


class UnauthenticatedAPIError(AppAPIException):
    def __init__(self, message: str = "Authentication required.", details: Optional[Any] = None):
        super().__init__(
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHENTICATED",
            message,
            details,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenAPIError(AppAPIException):
    def __init__(self, message: str = "Permission denied.", details: Optional[Any] = None):
        super().__init__(status.HTTP_403_FORBIDDEN, "FORBIDDEN", message, details)


class NotFoundAPIError(AppAPIException):
    def __init__(self, message: str = "Resource not found.", details: Optional[Any] = None):
        super().__init__(status.HTTP_404_NOT_FOUND, "NOT_FOUND", message, details)


class InvalidStateAPIError(AppAPIException):
    def __init__(self, message: str = "Invalid state or transition.", details: Optional[Any] = None):
        super().__init__(status.HTTP_409_CONFLICT, "INVALID_STATE", message, details)


class ConflictAPIError(AppAPIException):
    def __init__(self, message: str = "Conflict detected.", details: Optional[Any] = None):
        super().__init__(status.HTTP_409_CONFLICT, "CONFLICT", message, details)


class RateLimitedAPIError(AppAPIException):
    def __init__(self, message: str = "Rate limit exceeded.", details: Optional[Any] = None):
        super().__init__(status.HTTP_429_TOO_MANY_REQUESTS, "RATE_LIMITED", message, details)


class InternalAPIError(AppAPIException):
    def __init__(self, message: str = "An internal server error occurred.", details: Optional[Any] = None):
        super().__init__(status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", message, details)


def sanitize_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize FastAPI / Pydantic validation errors by stripping all 'input' values.

    Guarantees that patient narrative, clinical notes, and sensitive strings are never
    echoed back in HTTP 422 responses.
    """
    sanitized: List[Dict[str, Any]] = []
    for err in errors:
        d = dict(err)
        d.pop("input", None)
        if "ctx" in d and isinstance(d["ctx"], dict):
            clean_ctx = dict(d["ctx"])
            clean_ctx.pop("input", None)
            d["ctx"] = clean_ctx
        sanitized.append(d)
    return sanitized
