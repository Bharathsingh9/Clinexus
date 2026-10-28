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


def test_audit_events_linked_across_report_revisions(test_db_session):
    """Verify audit events are accurately linked to specific report versions across revisions."""
    repo = AuditRepository(test_db_session)

    # Event for Report Version 1
    ev_v1 = AuditEventSchema(
        event_type="REPORT_SYNTHESIZED",
        actor_id="fusion-agent",
        actor_role="agent",
        resource_type="report_version",
        resource_id="rv-version-1",
        payload={"version_number": 1, "confidence": 0.85},
    )
    repo.append_event(ev_v1)

    # Event for Report Version 2 (revision)
    ev_v2 = AuditEventSchema(
        event_type="REPORT_REVISED",
        actor_id="fusion-agent",
        actor_role="agent",
        resource_type="report_version",
        resource_id="rv-version-2",
        payload={"version_number": 2, "confidence": 0.91},
    )
    repo.append_event(ev_v2)

    # Retrieve and verify separate version linkage
    v1_events = repo.list_by_resource("report_version", "rv-version-1")
    assert len(v1_events) == 1
    assert v1_events[0].id == ev_v1.event_id
    assert "version_number\": 1" in v1_events[0].payload

    v2_events = repo.list_by_resource("report_version", "rv-version-2")
    assert len(v2_events) == 1
    assert v2_events[0].id == ev_v2.event_id
    assert "version_number\": 2" in v2_events[0].payload
