"""Unit Tests for Authentication, Password Hashing, JWT, and Roles (SEC-1, SEC-2).

Specification Section 2, Section 7:
- JWT bearer tokens with expiration and signature validation.
- Argon2 password hashing.
- Roles: clinician_reviewer, admin, viewer.
- Viewer is read-only and cannot perform review actions; clinician_reviewer and admin can.
- Expired or tampered token is rejected.
- First admin creation only when no user exists.
- Passwords stored as argon2 hashes only.
"""

from datetime import timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.persistence.database import Base
from backend.app.persistence.models import User as UserModel
from backend.app.security import (
    InvalidTokenError,
    TokenExpiredError,
    UnauthorizedActionError,
    UserRole,
    authenticate_user,
    authorize_action,
    can_perform_action,
    create_access_token,
    decode_access_token,
    ensure_first_admin,
    hash_password,
    verify_password,
)


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for security tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_argon2_password_hashing():
    """Verify password hashing creates valid Argon2 hashes and verifies accurately."""
    pwd = "CorrectHorseBatteryStaple123!"
    hashed = hash_password(pwd)

    # Argon2 hashes start with $argon2
    assert hashed.startswith("$argon2")
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False


def test_jwt_create_and_decode_valid():
    """Verify standard JWT token creation and decoding."""
    token = create_access_token(
        user_id="u-123",
        email="doctor@clinexus.local",
        role=UserRole.CLINICIAN_REVIEWER.value,
        secret_key="a"*32,
    )
    assert isinstance(token, str)

    payload = decode_access_token(token, secret_key="a"*32)
    assert payload["sub"] == "u-123"
    assert payload["email"] == "doctor@clinexus.local"
    assert payload["role"] == "clinician_reviewer"


def test_jwt_expired_token_rejected():
    """Verify expired token is rejected with TokenExpiredError."""
    token = create_access_token(
        user_id="u-exp",
        email="user@test.local",
        role="viewer",
        expires_delta=timedelta(seconds=-10),  # expired 10 seconds ago
        secret_key="b"*32,
    )

    with pytest.raises(TokenExpiredError):
        decode_access_token(token, secret_key="b"*32)


def test_jwt_tampered_token_rejected():
    """Verify tampered token is rejected with InvalidTokenError."""
    token = create_access_token(
        user_id="u-valid",
        email="user@test.local",
        role="viewer",
        secret_key="c"*32,
    )

    # Tamper with the token string
    tampered = token[:-4] + "abcd"
    with pytest.raises(InvalidTokenError):
        decode_access_token(tampered, secret_key="c"*32)


def test_role_authorization_review_actions():
    """Verify only clinician_reviewer and admin can perform review actions; viewer is rejected."""
    review_actions = [
        "REQUEST_CLARIFICATION",
        "REQUEST_EVIDENCE_REFRESH",
        "REQUEST_REVISION",
        "MARK_REVIEWED",
        "ADD_NOTE",
    ]

    for action in review_actions:
        # Clinician reviewer can act
        assert can_perform_action(UserRole.CLINICIAN_REVIEWER.value, action) is True
        assert authorize_action(UserRole.CLINICIAN_REVIEWER.value, action) is True

        # Admin can act
        assert can_perform_action(UserRole.ADMIN.value, action) is True
        assert authorize_action(UserRole.ADMIN.value, action) is True

        # Viewer CANNOT act
        assert can_perform_action(UserRole.VIEWER.value, action) is False
        with pytest.raises(UnauthorizedActionError):
            authorize_action(UserRole.VIEWER.value, action)


def test_authenticate_user_success_and_failure(sqlite_session):
    """Verify user authentication logic with argon2 password verification."""
    pwd = "SecureClinicianPassword123!"
    user = UserModel(
        id="usr-1",
        email="clinician@hospital.org",
        hashed_password=hash_password(pwd),
        role=UserRole.CLINICIAN_REVIEWER.value,
        is_active=True,
    )
    sqlite_session.add(user)
    sqlite_session.commit()

    # Success
    auth_user = authenticate_user("clinician@hospital.org", pwd, sqlite_session)
    assert auth_user is not None
    assert auth_user.id == "usr-1"

    # Case insensitive email
    auth_ci = authenticate_user("CLINICIAN@HOSPITAL.ORG", pwd, sqlite_session)
    assert auth_ci is not None

    # Wrong password
    assert authenticate_user("clinician@hospital.org", "WrongPassword", sqlite_session) is None

    # Non-existent user
    assert authenticate_user("nobody@hospital.org", pwd, sqlite_session) is None


def test_first_admin_creation_only_when_no_user_exists(sqlite_session):
    """Verify first admin account is created ONLY when no user exists in database."""
    # 1. DB is initially empty -> creates first admin
    admin = ensure_first_admin(
        sqlite_session,
        admin_email="superadmin@clinexus.local",
        admin_password="AdminPassword123!",
    )
    assert admin is not None
    assert admin.email == "superadmin@clinexus.local"
    assert admin.role == UserRole.ADMIN.value
    assert admin.hashed_password.startswith("$argon2")

    # 2. Subsequent call when user exists returns None and does not duplicate
    admin_repeat = ensure_first_admin(
        sqlite_session,
        admin_email="another@clinexus.local",
        admin_password="Password123!",
    )
    assert admin_repeat is None
