"""Database Session and Engine Management.

Specification Section 2, Section 6.
Supports PostgreSQL (production and integration tests) and SQLite (fast unit tests).
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.core.config import settings

Base = declarative_base()

# Synchronous engine for Alembic and sync tools
sync_engine = create_engine(
    settings.DATABASE_SYNC_URL,
    echo=False,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
SessionLocal = SyncSessionLocal

# Async engine for FastAPI application runtime
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for yielding an async session."""
    async with AsyncSessionLocal() as session:
        yield session
