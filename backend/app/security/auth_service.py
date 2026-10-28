"""Authentication Service and First-Admin Provisioning.

Specification Section 2, Section 7 (SEC-1, SEC-2, Req ID: SEC-1):
- First-admin creation from ADMIN_EMAIL and ADMIN_PASSWORD only when no user exists.
- Password storage strictly as Argon2id hashes.
- Never log passwords, tokens, or patient text.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import Settings
from backend.app.persistence.models import User as UserModel
from backend.app.security.password import hash_password, verify_password
from backend.app.security.roles import UserRole

logger = logging.getLogger(__name__)


def authenticate_user(
    email: str,
    password: str,
    session: Session,
) -> Optional[UserModel]:
    """Authenticate user credentials using email and Argon2 password verification.

    Returns UserModel if valid and active, else None.
    Safety Invariant: Never log password or token.
    """
    if not email or not password:
        return None

    stmt = select(UserModel).where(UserModel.email == email.strip().lower())
    user = session.execute(stmt).scalar_one_or_none()
    if not user:
        return None

    if not user.is_active:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def ensure_first_admin(
    session: Session,
    admin_email: Optional[str] = None,
    admin_password: Optional[str] = None,
) -> Optional[UserModel]:
    """Create the initial system administrator ONLY when no user exists in the database.

    Specification Section 7:
    "If no user exists at startup, create one admin from ADMIN_EMAIL and ADMIN_PASSWORD
     environment variables ([ENGINEERING-DECISION])."
    """
    # Check if any user already exists
    count_stmt = select(func.count(UserModel.id))
    user_count = session.execute(count_stmt).scalar() or 0

    if user_count > 0:
        logger.debug("Existing users present in database (%d); skipping first admin creation.", user_count)
        return None

    settings = Settings()
    email = admin_email or settings.ADMIN_EMAIL
    pwd = admin_password or settings.ADMIN_PASSWORD

    hashed = hash_password(pwd)
    admin_user = UserModel(
        id=str(uuid.uuid4()),
        email=email.strip().lower(),
        hashed_password=hashed,
        role=UserRole.ADMIN.value,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    session.add(admin_user)
    session.commit()
    session.refresh(admin_user)

    logger.info("First administrator account provisioned successfully.")
    return admin_user
