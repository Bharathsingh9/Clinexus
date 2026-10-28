"""Pure Finite State Machine for Clinician Review Workflow.

Specification Section 5.4 (EXT-D, Req ID: EXT-D):
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
- DRAFT -> ANALYZING
- ANALYZING -> EVIDENCE_REVIEW_REQUIRED, ABSTAINED, READY_FOR_REVIEW, FAILED
- EVIDENCE_REVIEW_REQUIRED -> ANALYZING
- ABSTAINED -> ANALYZING, REVIEWED
- READY_FOR_REVIEW -> REVIEWED, REVISION_REQUESTED
- REVISION_REQUESTED -> ANALYZING
- REVIEWED -> REVISION_REQUESTED
- FAILED -> ANALYZING

All other transitions raise InvalidTransition (HTTP 409 in API).
"""

from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Dict, FrozenSet, Optional
import uuid

from sqlalchemy.orm import Session

from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.models import ClinicalCase as ClinicalCaseModel
from backend.app.schemas.executions import AuditEvent as AuditEventSchema

logger = logging.getLogger(__name__)


class CaseReviewState(str, Enum):
    """The 8 authoritative review states defined in docs/SPEC.md Section 5.4."""
    DRAFT = "DRAFT"
    ANALYZING = "ANALYZING"
    EVIDENCE_REVIEW_REQUIRED = "EVIDENCE_REVIEW_REQUIRED"
    ABSTAINED = "ABSTAINED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEWED = "REVIEWED"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    FAILED = "FAILED"


class InvalidTransition(Exception):
    """Raised when an illegal state transition is attempted."""
    def __init__(self, current_state: str, attempted_state: str):
        self.current_state = current_state
        self.attempted_state = attempted_state
        super().__init__(
            f"Invalid state transition from '{current_state}' to '{attempted_state}'. "
            f"Strictly prohibited by SPEC 5.4 review state machine."
        )


# Pure transition table mapping from_state -> set of allowed to_states
ALLOWED_TRANSITIONS: Dict[CaseReviewState, FrozenSet[CaseReviewState]] = {
    CaseReviewState.DRAFT: frozenset({
        CaseReviewState.ANALYZING,
    }),
    CaseReviewState.ANALYZING: frozenset({
        CaseReviewState.EVIDENCE_REVIEW_REQUIRED,
        CaseReviewState.ABSTAINED,
        CaseReviewState.READY_FOR_REVIEW,
        CaseReviewState.FAILED,
    }),
    CaseReviewState.EVIDENCE_REVIEW_REQUIRED: frozenset({
        CaseReviewState.ANALYZING,
    }),
    CaseReviewState.ABSTAINED: frozenset({
        CaseReviewState.ANALYZING,
        CaseReviewState.REVIEWED,
    }),
    CaseReviewState.READY_FOR_REVIEW: frozenset({
        CaseReviewState.REVIEWED,
        CaseReviewState.REVISION_REQUESTED,
    }),
    CaseReviewState.REVISION_REQUESTED: frozenset({
        CaseReviewState.ANALYZING,
    }),
    CaseReviewState.REVIEWED: frozenset({
        CaseReviewState.REVISION_REQUESTED,
    }),
    CaseReviewState.FAILED: frozenset({
        CaseReviewState.ANALYZING,
    }),
}


# Mapping table from orchestrator workflow outcome to review state (docs/architecture.md)
WORKFLOW_OUTCOME_TO_REVIEW_STATE: Dict[str, CaseReviewState] = {
    "start_run": CaseReviewState.ANALYZING,
    "finalize_ready": CaseReviewState.READY_FOR_REVIEW,
    "finalize_abstain": CaseReviewState.ABSTAINED,
    "blocking_clarification": CaseReviewState.EVIDENCE_REVIEW_REQUIRED,
    "failure": CaseReviewState.FAILED,
}


def can_transition(current: CaseReviewState, target: CaseReviewState) -> bool:
    """Pure predicate: returns True if transition is valid, False otherwise."""
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    return target in allowed


def validate_transition(current: CaseReviewState, target: CaseReviewState) -> None:
    """Validate transition, raising InvalidTransition if disallowed."""
    if not can_transition(current, target):
        raise InvalidTransition(current.value, target.value)


def transition_case_state(
    session: Session,
    case_id: str,
    target_state: CaseReviewState,
    actor_id: str,
    actor_role: str,
    report_version_id: Optional[str] = None,
    audit_repo: Optional[AuditRepository] = None,
    reason: Optional[str] = None,
) -> ClinicalCaseModel:
    """Perform a validated state transition on a ClinicalCase and append an AuditEvent.

    Raises:
        ValueError: If case_id does not exist.
        InvalidTransition: If the transition is prohibited by SPEC 5.4.
    """
    case_row = session.get(ClinicalCaseModel, case_id)
    if not case_row:
        raise ValueError(f"ClinicalCase '{case_id}' not found.")

    try:
        current_enum = CaseReviewState(case_row.status)
    except ValueError:
        raise InvalidTransition(case_row.status, target_state.value)

    validate_transition(current_enum, target_state)

    old_status = case_row.status
    case_row.status = target_state.value

    # Build and append immutable AuditEvent for state change
    payload = {
        "case_id": case_id,
        "from_state": old_status,
        "to_state": target_state.value,
        "report_version_id": report_version_id,
        "reason": reason,
    }

    event = AuditEventSchema(
        event_id=str(uuid.uuid4()),
        event_type="CASE_STATE_CHANGED",
        actor_id=actor_id,
        actor_role=actor_role,
        resource_type="clinical_case",
        resource_id=case_id,
        payload=payload,
        timestamp=datetime.now(timezone.utc),
    )

    if audit_repo is not None:
        audit_repo.append_event(event)
    else:
        # Direct write via AuditRepository
        ar = AuditRepository(session)
        ar.append_event(event)

    session.commit()
    session.refresh(case_row)
    logger.info(
        "Transitioned ClinicalCase '%s' from %s to %s by actor %s (%s)",
        case_id, old_status, target_state.value, actor_id, actor_role
    )
    return case_row
