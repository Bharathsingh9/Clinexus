"""Unit Tests for Immutable Append-Only Audit Repository.

Specification Section 5.3, Section 6, Rule R11.
Asserts that:
- Events can be appended and retrieved.
- AuditRepository exposes NO update or delete methods.
"""

import inspect
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import Base
from backend.app.schemas.executions import AuditEvent as AuditEventSchema


@pytest.fixture
def test_db_session():
    """In-memory SQLite database session for audit repository test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_audit_repository_has_no_update_or_delete_methods():
    """Security Invariant: The audit repository MUST NOT expose any update or delete methods."""
    methods = [
        name for name, _ in inspect.getmembers(AuditRepository, predicate=inspect.isfunction)
    ]
    forbidden_prefixes = ["update", "delete", "remove", "modify", "patch", "drop", "truncate", "destroy"]
    for method_name in methods:
        for prefix in forbidden_prefixes:
            assert not method_name.lower().startswith(prefix), (
                f"AuditRepository violates append-only invariant by exposing '{method_name}'"
            )


def test_append_and_retrieve_audit_event(test_db_session):
    """Verify appending and reading events."""
    repo = AuditRepository(test_db_session)

    event = AuditEventSchema(
        event_type="CASE_SUBMITTED",
        actor_id="user-123",
        actor_role="clinician_reviewer",
        resource_type="clinical_case",
        resource_id="case-456",
        payload={"raw_text_length": 250},
    )

    created = repo.append_event(event)
    assert created.id == event.event_id
    assert created.event_type == "CASE_SUBMITTED"

    fetched = repo.get_by_id(event.event_id)
    assert fetched is not None
    assert fetched.actor_id == "user-123"

    events = repo.list_by_resource("clinical_case", "case-456")
    assert len(events) == 1
    assert events[0].id == event.event_id
