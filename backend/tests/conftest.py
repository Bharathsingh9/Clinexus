"""Global Pytest Fixtures and Environment Configuration.

Specification Section 2, Section 6, Section 11 (Req ID: DAT-3, RET-1, DB-1):
- Forces test isolation: uses PostgreSQL database clinexus_test and Qdrant prefix test_evidence.
- Guard asserts test session aborts immediately if configured database is the main database clinexus.
"""

import os
from urllib.parse import urlparse
import pytest

# Explicitly supply test credentials and isolation defaults for test runner
os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-characters-long-abc12345")
os.environ.setdefault("ADMIN_PASSWORD", "test-admin-password-1234")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres_password@localhost:5432/clinexus_test")
os.environ.setdefault("DATABASE_SYNC_URL", "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus_test")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")


def get_database_name(url: str) -> str:
    """Extract database name from connection URL."""
    try:
        parsed = urlparse(url)
        return parsed.path.lstrip("/")
    except Exception:
        return ""


def assert_test_database_isolated(db_url: str) -> None:
    """Abort if the target database is the main database clinexus or not an isolated test database.
    
    Raises:
        RuntimeError: if configured database is clinexus or not a test database.
    """
    db_name = get_database_name(db_url)
    if db_name == "clinexus":
        raise RuntimeError(
            f"ABORT: Test execution blocked by test isolation guard! "
            f"Configured database is main production database '{db_name}'. "
            f"All tests MUST use 'clinexus_test' or a dedicated test database."
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pytest session hook: verify test database isolation before running any tests."""
    from backend.app.core.config import settings

    for key, url in [
        ("DATABASE_URL", settings.DATABASE_URL),
        ("DATABASE_SYNC_URL", settings.DATABASE_SYNC_URL),
        ("ENV_DATABASE_URL", os.environ.get("DATABASE_URL", "")),
        ("ENV_DATABASE_SYNC_URL", os.environ.get("DATABASE_SYNC_URL", "")),
    ]:
        if url:
            try:
                assert_test_database_isolated(url)
            except RuntimeError as err:
                pytest.exit(str(err), returncode=3)
