"""Unit Tests for Clinician Review State Machine (EXT-D).

Specification Section 5.4:
States:
- DRAFT
- ANALYZING
- EVIDENCE_REVIEW_REQUIRED
- ABSTAINED
- READY_FOR_REVIEW
- REVIEWED
- REVISION_REQUESTED
- FAILED

Allowed transitions:
- DRAFT to ANALYZING
- ANALYZING to EVIDENCE_REVIEW_REQUIRED, ABSTAINED, READY_FOR_REVIEW, or FAILED
- EVIDENCE_REVIEW_REQUIRED to ANALYZING
- ABSTAINED to ANALYZING or REVIEWED
- READY_FOR_REVIEW to REVIEWED or REVISION_REQUESTED
- REVISION_REQUESTED to ANALYZING
- REVIEWED to REVISION_REQUESTED
- FAILED to ANALYZING

Test requirements:
- Generate full 8x8 Cartesian product matrix.
- Every allowed transition succeeds.
- Every disallowed pair strictly raises InvalidTransition.
- Persistence of state change appends an immutable AuditEvent and updates case status.
"""

import itertools
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.orchestration.states import (
    ALLOWED_TRANSITIONS,
    CaseReviewState,
    InvalidTransition,
    can_transition,
    transition_case_state,
    validate_transition,
)
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import Base
from backend.app.persistence.models import ClinicalCase as ClinicalCaseModel


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for testing state machine persistence."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_full_state_transition_matrix_truth_table():
    """Test all 64 state pairs (8 states x 8 states) against SPEC 5.4 allowed transitions."""
    all_states = list(CaseReviewState)
    total_pairs = len(all_states) * len(all_states)
    assert total_pairs == 64

    allowed_count = 0
    disallowed_count = 0

    for src_state, dst_state in itertools.product(all_states, all_states):
        is_allowed = dst_state in ALLOWED_TRANSITIONS.get(src_state, frozenset())

        if is_allowed:
            # Must succeed with can_transition and validate_transition
            assert can_transition(src_state, dst_state) is True
            validate_transition(src_state, dst_state)
            allowed_count += 1
        else:
            # Must return False and raise InvalidTransition
            assert can_transition(src_state, dst_state) is False
            with pytest.raises(InvalidTransition) as exc_info:
                validate_transition(src_state, dst_state)
            assert exc_info.value.current_state == src_state.value
            assert exc_info.value.attempted_state == dst_state.value
            disallowed_count += 1

    # Exact expected counts based on SPEC 5.4 allowed transitions:
    # DRAFT: 1 (ANALYZING)
    # ANALYZING: 4 (EVIDENCE_REVIEW_REQUIRED, ABSTAINED, READY_FOR_REVIEW, FAILED)
    # EVIDENCE_REVIEW_REQUIRED: 1 (ANALYZING)
    # ABSTAINED: 2 (ANALYZING, REVIEWED)
    # READY_FOR_REVIEW: 2 (REVIEWED, REVISION_REQUESTED)
    # REVISION_REQUESTED: 1 (ANALYZING)
    # REVIEWED: 1 (REVISION_REQUESTED)
    # FAILED: 1 (ANALYZING)
    # Total allowed = 1 + 4 + 1 + 2 + 2 + 1 + 1 + 1 = 13
    assert allowed_count == 13
    assert disallowed_count == 64 - 13


def test_transition_case_state_updates_db_and_emits_audit(sqlite_session):
    """Verify transition_case_state updates ClinicalCase and writes state-change AuditEvent."""
    audit_repo = AuditRepository(sqlite_session)

    # Setup initial DRAFT case
    case = ClinicalCaseModel(
        id="case-sm-1",
        raw_text_hash="a" * 64,
        chief_complaint="Chest tightness",
        status=CaseReviewState.DRAFT.value,
    )
    sqlite_session.add(case)
    sqlite_session.commit()

    # 1. Allowed transition: DRAFT -> ANALYZING
    updated = transition_case_state(
        session=sqlite_session,
        case_id="case-sm-1",
        target_state=CaseReviewState.ANALYZING,
        actor_id="orch-1",
        actor_role="system",
        audit_repo=audit_repo,
        reason="Workflow starting",
    )
    assert updated.status == CaseReviewState.ANALYZING.value

    # Check audit event
    events = audit_repo.list_by_resource("clinical_case", "case-sm-1")
    assert len(events) == 1
    assert events[0].event_type == "CASE_STATE_CHANGED"
    assert "DRAFT" in events[0].payload
    assert "ANALYZING" in events[0].payload

    # 2. Allowed transition: ANALYZING -> READY_FOR_REVIEW
    updated2 = transition_case_state(
        session=sqlite_session,
        case_id="case-sm-1",
        target_state=CaseReviewState.READY_FOR_REVIEW,
        actor_id="orch-1",
        actor_role="system",
        audit_repo=audit_repo,
        reason="Passed all gates",
    )
    assert updated2.status == CaseReviewState.READY_FOR_REVIEW.value

    events = audit_repo.list_by_resource("clinical_case", "case-sm-1")
    assert len(events) == 2


def test_disallowed_transition_raises_and_rolls_back(sqlite_session):
    """Verify disallowed transition raises InvalidTransition and leaves status untouched."""
    audit_repo = AuditRepository(sqlite_session)

    case = ClinicalCaseModel(
        id="case-sm-2",
        raw_text_hash="b" * 64,
        chief_complaint="Dyspnea",
        status=CaseReviewState.DRAFT.value,
    )
    sqlite_session.add(case)
    sqlite_session.commit()

    # DRAFT cannot jump directly to REVIEWED
    with pytest.raises(InvalidTransition):
        transition_case_state(
            session=sqlite_session,
            case_id="case-sm-2",
            target_state=CaseReviewState.REVIEWED,
            actor_id="user-1",
            actor_role="clinician_reviewer",
            audit_repo=audit_repo,
        )

    # Status must still be DRAFT
    refreshed = sqlite_session.get(ClinicalCaseModel, "case-sm-2")
    assert refreshed.status == CaseReviewState.DRAFT.value
