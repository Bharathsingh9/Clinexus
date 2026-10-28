"""Security and Authentication Package.

Exports password hashing, JWT bearer token management, role-based access control,
and first-admin bootstrap logic.
"""

from backend.app.security.auth_service import authenticate_user, ensure_first_admin
from backend.app.security.jwt import (
    InvalidTokenError,
    TokenExpiredError,
    create_access_token,
    decode_access_token,
)
from backend.app.security.password import hash_password, verify_password
from backend.app.security.roles import (
    ADMIN_ACTIONS,
    READ_ACTIONS,
    REVIEW_ACTIONS,
    UnauthorizedActionError,
    UserRole,
    authorize_action,
    can_perform_action,
)

__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "TokenExpiredError",
    "InvalidTokenError",
    "UserRole",
    "REVIEW_ACTIONS",
    "ADMIN_ACTIONS",
    "READ_ACTIONS",
    "UnauthorizedActionError",
    "can_perform_action",
    "authorize_action",
    "authenticate_user",
    "ensure_first_admin",
]
