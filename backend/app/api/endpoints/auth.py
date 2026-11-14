"""Authentication and User Management Endpoints.

Specification Section 2, Section 7 (API-2, SEC-1, SEC-2):
- POST /auth/login: Authenticates user, issues JWT token with role and expiry.
  Rate-limited per IP. Equal error for unknown email and bad password.
- GET /users: Admin-only. Lists users; never returns password hashes.
- POST /users: Admin-only. Creates user with password >= 12 chars; rejects duplicates.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, require_admin
from backend.app.api.errors import (
    ConflictAPIError,
    UnauthenticatedAPIError,
    ValidationAPIError,
)
from backend.app.api.rate_limiter import rate_limit_login
from backend.app.core.config import settings
from backend.app.persistence.models import User as UserModel
from backend.app.security.auth_service import authenticate_user
from backend.app.security.jwt import create_access_token
from backend.app.security.password import hash_password
from backend.app.security.roles import UserRole

router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(..., min_length=3, max_length=255)
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if "@" not in clean or "." not in clean.split("@")[-1]:
            raise ValueError(f"Invalid email address format: '{v}'")
        return clean


class LoginResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int


class UserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=12, description="Password must be at least 12 characters")
    role: str = Field(default=UserRole.VIEWER.value)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if "@" not in clean or "." not in clean.split("@")[-1]:
            raise ValueError(f"Invalid email address format: '{v}'")
        return clean


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    summary="Authenticate user and issue JWT bearer token",
    dependencies=[Depends(rate_limit_login)],
)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """Authenticate email and password, returning signed JWT bearer token.

    Safety Invariant: Returns identical 401 error for non-existent email and wrong password.
    Never logs passwords or tokens.
    """
    user = authenticate_user(str(payload.email), payload.password, session=db)
    if not user:
        raise UnauthenticatedAPIError("Invalid email or password.")

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
    )
    expires_in_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        expires_in=expires_in_seconds,
    )


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all users (Admin only)",
    dependencies=[Depends(require_admin)],
)
def list_users(db: Session = Depends(get_db)) -> List[UserResponse]:
    """Retrieve list of platform users. Never exposes password hashes."""
    stmt = select(UserModel).order_by(UserModel.created_at.asc())
    users = list(db.scalars(stmt).all())
    return [
        UserResponse(
            id=u.id,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in users
    ]


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=201,
    summary="Create a new user (Admin only)",
    dependencies=[Depends(require_admin)],
)
def create_user(payload: CreateUserRequest, db: Session = Depends(get_db)) -> UserResponse:
    """Create a new user with minimum 12-character password. Admin-only."""
    # Validate role enum
    valid_roles = {UserRole.ADMIN.value, UserRole.CLINICIAN_REVIEWER.value, UserRole.VIEWER.value}
    if payload.role not in valid_roles:
        raise ValidationAPIError(
            f"Invalid role '{payload.role}'. Must be one of: {sorted(valid_roles)}"
        )

    clean_email = str(payload.email).strip().lower()

    # Check duplicate
    stmt = select(UserModel).where(UserModel.email == clean_email)
    existing = db.scalars(stmt).first()
    if existing:
        raise ConflictAPIError(f"User with email '{clean_email}' already exists.")

    hashed = hash_password(payload.password)
    new_user = UserModel(
        id=str(uuid.uuid4()),
        email=clean_email,
        hashed_password=hashed,
        role=payload.role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        role=new_user.role,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
    )
