"""Argon2 Password Hashing Service.

Specification Section 2, Section 5.4, Rule R11:
- Uses Argon2 for one-way password hashing.
- Never log plain passwords, tokens, or patient text.
"""

import logging
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

logger = logging.getLogger(__name__)

# Standard Argon2id hasher instance
_hasher = PasswordHasher()


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using Argon2id.

    Safety Invariant: Never log the plain password.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against an Argon2 hash.

    Returns True if match, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError):
        return False
