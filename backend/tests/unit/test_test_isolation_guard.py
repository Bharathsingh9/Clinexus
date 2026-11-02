"""Unit tests for Test Database Isolation Guard.

Specification Section 2, Section 6, Section 11 (Req ID: DAT-3, DB-1).
Asserts that test runner blocks execution if configured database is the main database.
"""

import pytest
from backend.tests.conftest import assert_test_database_isolated, get_database_name


def test_get_database_name():
    """Verify database name extraction from URL."""
    assert get_database_name("postgresql+asyncpg://user:pass@localhost:5432/clinexus") == "clinexus"
    assert get_database_name("postgresql+psycopg2://user:pass@localhost:5432/clinexus_test") == "clinexus_test"
    assert get_database_name("sqlite:///test.db") == "test.db"


def test_isolation_guard_aborts_on_main_database():
    """Assert guard raises RuntimeError when URL targets main production database clinexus."""
    with pytest.raises(RuntimeError, match="ABORT: Test execution blocked by test isolation guard"):
        assert_test_database_isolated("postgresql+asyncpg://user:pass@localhost:5432/clinexus")

    with pytest.raises(RuntimeError, match="ABORT: Test execution blocked by test isolation guard"):
        assert_test_database_isolated("postgresql+psycopg2://user:pass@localhost:5432/clinexus")


def test_isolation_guard_allows_test_database():
    """Assert guard passes cleanly when URL targets clinexus_test or custom test db."""
    # Should not raise
    assert_test_database_isolated("postgresql+asyncpg://user:pass@localhost:5432/clinexus_test")
    assert_test_database_isolated("postgresql+psycopg2://user:pass@localhost:5432/clinexus_test")
    assert_test_database_isolated("sqlite:///test.db")
