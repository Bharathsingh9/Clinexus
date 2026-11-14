"""FastAPI Route Dependencies and Security Invariants.

Specification Section 2, Section 5.4, Section 7:
- Database session lifecycle management.
- Production LLM provider injection (Gemini or None; tests override with FakeLLMProvider).
- Bearer token authentication and server-side RBAC enforcement.
"""

from typing import Callable, Generator, List, Optional
from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.api.errors import ForbiddenAPIError, UnauthenticatedAPIError
from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import User as UserModel
from backend.app.security.jwt import (
    InvalidTokenError,
    TokenExpiredError,
    decode_access_token,
)
from backend.app.security.roles import UserRole

http_bearer = HTTPBearer(auto_error=False)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional SQLAlchemy database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_llm_provider() -> Optional[LLMProvider]:
    """Provide production LLM provider strictly from environment configuration.

    Production wiring builds GeminiProvider for 'gemini', and None for 'none' or 'fake'.
    When None, agents raise LLM_UNAVAILABLE failure semantics per SPEC Section 3.
    Tests override this dependency with FakeLLMProvider via FastAPI dependency overrides.
    Production code contains no test stubs.
    """
    from backend.app.core.llm_provider import create_llm_provider
    return create_llm_provider(settings)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    db: Session = Depends(get_db),
) -> UserModel:
    """Extract, decode, and validate the bearer JWT token, returning the authenticated user.

    Raises:
        UnauthenticatedAPIError: If token is missing, expired, invalid, or user is inactive.
    """
    if not credentials or not credentials.credentials:
        raise UnauthenticatedAPIError("Missing bearer authentication token.")

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except TokenExpiredError as exc:
        raise UnauthenticatedAPIError("Bearer token has expired.") from exc
    except InvalidTokenError as exc:
        raise UnauthenticatedAPIError("Bearer token is invalid or tampered.") from exc
    except Exception as exc:
        raise UnauthenticatedAPIError("Could not validate authentication credentials.") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthenticatedAPIError("Token payload missing subject identifier.")

    user = db.get(UserModel, user_id)
    if not user or not user.is_active:
        raise UnauthenticatedAPIError("Authenticated user no longer exists or is deactivated.")

    return user


def require_role(allowed_roles: List[str]) -> Callable[[UserModel], UserModel]:
    """Dependency factory enforcing server-side Role-Based Access Control."""
    def role_checker(current_user: UserModel = Depends(get_current_user)) -> UserModel:
        if current_user.role not in allowed_roles:
            raise ForbiddenAPIError(
                f"Role '{current_user.role}' is not authorized. Required: {allowed_roles}"
            )
        return current_user

    return role_checker


# Canonical RBAC callable dependencies
require_admin = require_role([UserRole.ADMIN.value])
require_clinician_or_admin = require_role([UserRole.CLINICIAN_REVIEWER.value, UserRole.ADMIN.value])
