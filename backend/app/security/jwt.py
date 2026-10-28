"""JWT Bearer Token Management Service.

Specification Section 2, Section 5.4, Rule R11:
- Uses PyJWT for encoding and decoding signed bearer tokens.
- Pinned version is logged at module import.
- Never log tokens, passwords, or patient text.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, Optional
import jwt

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)

# Pin and log the PyJWT version explicitly
PYJWT_VERSION = jwt.__version__
logger.info("Using PyJWT version %s for token management", PYJWT_VERSION)


class TokenExpiredError(Exception):
    """Raised when an access token has expired."""
    pass


class InvalidTokenError(Exception):
    """Raised when an access token signature or payload is invalid or tampered with."""
    pass


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Generate a signed JWT access token.

    Safety Invariant: Never log the generated token or user email.
    """
    settings = Settings()
    key = secret_key or settings.SECRET_KEY
    algo = algorithm or settings.ALGORITHM

    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    token = jwt.encode(payload, key, algorithm=algo)
    return token


def decode_access_token(
    token: str,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode and validate a signed JWT access token.

    Raises:
        TokenExpiredError: If token expiration (exp) is in the past.
        InvalidTokenError: If token signature or format is invalid/tampered.
    """
    settings = Settings()
    key = secret_key or settings.SECRET_KEY
    algo = algorithm or settings.ALGORITHM

    try:
        payload = jwt.decode(token, key, algorithms=[algo])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token signature has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError("Invalid or tampered token.") from exc
