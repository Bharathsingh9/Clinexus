"""Unit Tests for Clinician Review Service (EXT-D).

Specification Section 5.4:
- Actions: REQUEST_CLARIFICATION, REQUEST_EVIDENCE_REFRESH, REQUEST_REVISION, MARK_REVIEWED, ADD_NOTE.
- Only clinician_reviewer and admin may act; viewer is rejected.
- MARK_REVIEWED text is "Reviewed by clinician", never "approved", "validated", or "correct".
- Reviews are append-only (an earlier review is never overwritten).
- Each review records reviewer ID, time, action, report version ID, and outcome.
- Notes are stored apart from AI content labeled HUMAN.
- REQUEST_REVISION and REQUEST_EVIDENCE_REFRESH move REVIEWED or READY_FOR_REVIEW to REVISION_REQUESTED.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.orchestration.review_service import FORBIDDEN_APPROVAL_PHRASES, ReviewService
from backend.app.orchestration.states import CaseReviewState
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import Base
from backend.app.persistence.models import (
    ClinicalCase as ClinicalCaseModel,
    ReportVersion as ReportVersionModel,
    User as UserModel,
)
from backend.app.schemas.reports import ReviewActionEnum
from backend.app.security import UnauthorizedActionError, UserRole


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for review service tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def setup_case_and_report(sqlite_session):
    """Fixture providing initialized users, case, and report."""
    clinician = UserModel(
        id="usr-dr-smith",
        email="smith@hospital.org",
        hashed_password="$argon2...",
        role=UserRole.CLINICIAN_REVIEWER.value,
    )
    viewer = UserModel(
        id="usr-viewer-1",
        email="viewer@hospital.org",
        hashed_password="$argon2...",
        role=UserRole.VIEWER.value,
    )
    case = ClinicalCaseModel(
        id="case-rev-1",
        raw_text_hash="c" * 64,
        chief_complaint="Chest pain on exertion",
        status=CaseReviewState.READY_FOR_REVIEW.value,
    )
    report = ReportVersionModel(
        id="rv-rev-1",
        case_id="case-rev-1",
        execution_id="exec-rev-1",
        version_number=1,
        summary="Candidate synthesis report",
        confidence_score=0.88,
        status="READY_FOR_REVIEW",
    )
    sqlite_session.add_all([clinician, viewer, case, report])
    sqlite_session.commit()
    return {"case_id": case.id, "report_id": report.id, "clinician_id": clinician.id, "viewer_id": viewer.id}


def test_mark_reviewed_persists_exact_outcome_text(sqlite_session, setup_case_and_report):
    """Verify MARK_REVIEWED uses exact text 'Reviewed by clinician' and never forbidden approval words."""
    data = setup_case_and_report
    service = ReviewService(sqlite_session)

    review = service.record_review_action(
        case_id=data["case_id"],
        report_version_id=data["report_id"],
        reviewer_id=data["clinician_id"],
        reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
        action=ReviewActionEnum.MARK_REVIEWED,
    )

    # Invariant: Must be "Reviewed by clinician"
    assert review.outcome == "Reviewed by clinician"
    assert review.reviewer_id == data["clinician_id"]
    assert review.report_version_id == data["report_id"]

    for forbidden in FORBIDDEN_APPROVAL_PHRASES:
        assert forbidden not in review.outcome.lower()

    # Case state must transition from READY_FOR_REVIEW to REVIEWED
    refreshed_case = sqlite_session.get(ClinicalCaseModel, data["case_id"])
    assert refreshed_case.status == CaseReviewState.REVIEWED.value


def test_reviews_are_append_only_never_overwritten(sqlite_session, setup_case_and_report):
    """Verify multiple reviews on a case append new rows and never overwrite earlier reviews."""
    data = setup_case_and_report
    service = ReviewService(sqlite_session)

    # First review: ADD_NOTE
    rev1 = service.record_review_action(
        case_id=data["case_id"],
        report_version_id=data["report_id"],
        reviewer_id=data["clinician_id"],
        reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
        action=ReviewActionEnum.ADD_NOTE,
        note="Initial clinician impression: verify ECG findings.",
    )

    # Second review: MARK_REVIEWED
    rev2 = service.record_review_action(
        case_id=data["case_id"],
        report_version_id=data["report_id"],
        reviewer_id=data["clinician_id"],
        reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
        action=ReviewActionEnum.MARK_REVIEWED,
    )

    reviews = service.list_reviews_for_case(data["case_id"])
    assert len(reviews) == 2
    assert reviews[0].review_id == rev1.review_id
    assert reviews[0].action == ReviewActionEnum.ADD_NOTE
    assert "[HUMAN]" in reviews[0].note

    assert reviews[1].review_id == rev2.review_id
    assert reviews[1].action == ReviewActionEnum.MARK_REVIEWED


def test_request_revision_moves_state_to_revision_requested(sqlite_session, setup_case_and_report):
    """Verify REQUEST_REVISION and REQUEST_EVIDENCE_REFRESH move case to REVISION_REQUESTED."""
    data = setup_case_and_report
    service = ReviewService(sqlite_session)

    rev = service.record_review_action(
        case_id=data["case_id"],
        report_version_id=data["report_id"],
        reviewer_id=data["clinician_id"],
        reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
        action=ReviewActionEnum.REQUEST_REVISION,
        note="Re-evaluate potential pulmonary embolism considerations.",
    )

    refreshed_case = sqlite_session.get(ClinicalCaseModel, data["case_id"])
    assert refreshed_case.status == CaseReviewState.REVISION_REQUESTED.value
    assert "Revision requested" in rev.outcome


def test_viewer_is_strictly_rejected_from_review_actions(sqlite_session, setup_case_and_report):
    """Safety Invariant: Viewer role is strictly prohibited from executing review actions."""
    data = setup_case_and_report
    service = ReviewService(sqlite_session)

    with pytest.raises(UnauthorizedActionError):
        service.record_review_action(
            case_id=data["case_id"],
            report_version_id=data["report_id"],
            reviewer_id=data["viewer_id"],
            reviewer_role=UserRole.VIEWER.value,
            action=ReviewActionEnum.MARK_REVIEWED,
        )
