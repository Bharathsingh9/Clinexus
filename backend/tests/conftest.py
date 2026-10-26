"""Global Pytest Fixtures and Environment Configuration.

Explicitly supplies test settings for testing (Task 1).
"""

import os
import pytest

# Explicitly supply test credentials for test runner
os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-characters-long-abc12345")
os.environ.setdefault("ADMIN_PASSWORD", "test-admin-password-1234")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres_password@localhost:5432/clinexus")
os.environ.setdefault("DATABASE_SYNC_URL", "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
