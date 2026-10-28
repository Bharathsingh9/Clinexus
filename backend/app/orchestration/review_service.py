"""Clinician Review Service Layer.

Specification Section 5.4 (EXT-D, Req ID: EXT-D):
Actions:
- REQUEST_CLARIFICATION
- REQUEST_EVIDENCE_REFRESH
- REQUEST_REVISION
- MARK_REVIEWED
- ADD_NOTE

Rules:
- Only clinician_reviewer and admin may act; viewer is rejected.
- MARK_REVIEWED text is "Reviewed by clinician", never "approved", "validated", or "correct".
- Reviews are append-only (an earlier review is never overwritten).
- Each review records reviewer ID, time, action, report version ID, and outcome.
- Notes are stored apart from AI content labeled HUMAN.
- REQUEST_REVISION and REQUEST_EVIDENCE_REFRESH move REVIEWED or READY_FOR_REVIEW to REVISION_REQUESTED.
- A later workflow execution creates a NEW report version (old versions remain immutable
  with their provenance and audit links preserved).
"""

from datetime import datetime, timezone
import json
import logging
from typing import List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.orchestration.states import (
    CaseReviewState,
    InvalidTransition,
    transition_case_state,
)
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.models import (
    ClinicalCase as ClinicalCaseModel,
    HumanReview as HumanReviewModel,
    ReportVersion as ReportVersionModel,
)
from backend.app.schemas.executions import AuditEvent as AuditEventSchema
from backend.app.schemas.reports import HumanReview as HumanReviewSchema, ReviewActionEnum
from backend.app.security.roles import UserRole, authorize_action

logger = logging.getLogger(__name__)

# Strict invariant: forbidden approval phrases
FORBIDDEN_APPROVAL_PHRASES = ("approved", "validated", "correct", "accepted", "verified-approved")


class ReviewService:
    """Service handling clinician review actions, state transitions, and audit records."""

    def __init__(self, session: Session, audit_repo: Optional[AuditRepository] = None):
        self.session = session
        self.audit_repo = audit_repo or AuditRepository(session)

    def record_review_action(
        self,
        case_id: str,
        report_version_id: str,
        reviewer_id: str,
        reviewer_role: str,
        action: ReviewActionEnum,
        note: Optional[str] = None,
    ) -> HumanReviewSchema:
        """Execute and persist an append-only clinician review action.

        Raises:
            UnauthorizedActionError: If reviewer role is not authorized.
            ValueError: If case or report version is not found.
            InvalidTransition: If the action causes an illegal state transition.
        """
        # 1. Enforce Role-Based Access Control
        act_str = action.value if hasattr(action, "value") else str(action)
        authorize_action(reviewer_role, act_str)

        # 2. Validate case and report version existence
        case_row = self.session.get(ClinicalCaseModel, case_id)
        if not case_row:
            raise ValueError(f"ClinicalCase '{case_id}' not found.")

        report_row = self.session.get(ReportVersionModel, report_version_id)
        if not report_row:
            raise ValueError(f"ReportVersion '{report_version_id}' not found.")

        current_case_state = CaseReviewState(case_row.status)

        # 3. Determine Outcome and State Machine Transition
        target_state: Optional[CaseReviewState] = None
        outcome: str = ""

        if action == ReviewActionEnum.MARK_REVIEWED:
            # SPEC 5.4: MARK_REVIEWED text is "Reviewed by clinician", never "approved", "validated", or "correct"
            outcome = "Reviewed by clinician"
            # Target state is REVIEWED (allowed from READY_FOR_REVIEW or ABSTAINED)
            target_state = CaseReviewState.REVIEWED

        elif action in (ReviewActionEnum.REQUEST_REVISION, ReviewActionEnum.REQUEST_EVIDENCE_REFRESH):
            if action == ReviewActionEnum.REQUEST_REVISION:
                outcome = f"Revision requested: {note or 'Further clinical revision required'}"
            else:
                outcome = f"Evidence refresh requested: {note or 'Refresh literature evidence and re-analyze'}"
            # Target state is REVISION_REQUESTED (allowed from READY_FOR_REVIEW or REVIEWED)
            target_state = CaseReviewState.REVISION_REQUESTED

        elif action == ReviewActionEnum.REQUEST_CLARIFICATION:
            outcome = f"Clarification requested: {note or 'Missing patient or clinical information'}"
            # May remain in current state or move to EVIDENCE_REVIEW_REQUIRED if applicable

        elif action == ReviewActionEnum.ADD_NOTE:
            outcome = "Clinician note recorded"
            # State remains unchanged

        # Safety Assert: outcome must never contain forbidden approval terms
        for forbidden in FORBIDDEN_APPROVAL_PHRASES:
            if forbidden in outcome.lower() and action == ReviewActionEnum.MARK_REVIEWED:
                raise ValueError(
                    f"Safety invariant violated: MARK_REVIEWED outcome cannot contain '{forbidden}'"
                )

        # 4. Perform state transition if required
        if target_state is not None and target_state != current_case_state:
            transition_case_state(
                session=self.session,
                case_id=case_id,
                target_state=target_state,
                actor_id=reviewer_id,
                actor_role=reviewer_role,
                report_version_id=report_version_id,
                audit_repo=self.audit_repo,
                reason=outcome,
            )

        # 5. Append-only review persistence (HUMAN note stored distinctly)
        review_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        labeled_note = f"[HUMAN] {note}" if note else None

        db_review = HumanReviewModel(
            id=review_id,
            case_id=case_id,
            report_version_id=report_version_id,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            action=act_str,
            outcome=outcome,
            note=labeled_note,
            created_at=now,
        )
        self.session.add(db_review)

        # 6. Append review AuditEvent
        review_event = AuditEventSchema(
            event_id=str(uuid.uuid4()),
            event_type="HUMAN_REVIEW_RECORDED",
            actor_id=reviewer_id,
            actor_role=reviewer_role,
            resource_type="report_version",
            resource_id=report_version_id,
            payload={
                "case_id": case_id,
                "review_id": review_id,
                "action": act_str,
                "outcome": outcome,
                "has_note": bool(note),
            },
            timestamp=now,
        )
        self.audit_repo.append_event(review_event)

        self.session.commit()
        self.session.refresh(db_review)

        logger.info(
            "Recorded review action '%s' for report_version '%s' by reviewer '%s'",
            act_str, report_version_id, reviewer_id
        )

        return HumanReviewSchema(
            review_id=db_review.id,
            reviewer_id=db_review.reviewer_id,
            reviewer_role=db_review.reviewer_role,
            action=ReviewActionEnum(db_review.action),
            report_version_id=db_review.report_version_id,
            outcome=db_review.outcome,
            note=db_review.note,
            created_at=db_review.created_at,
        )

    def list_reviews_for_case(self, case_id: str) -> List[HumanReviewSchema]:
        """List all append-only reviews recorded for a clinical case."""
        stmt = (
            select(HumanReviewModel)
            .where(HumanReviewModel.case_id == case_id)
            .order_by(HumanReviewModel.created_at.asc())
        )
        rows = list(self.session.scalars(stmt).all())
        return [
            HumanReviewSchema(
                review_id=r.id,
                reviewer_id=r.reviewer_id,
                reviewer_role=r.reviewer_role,
                action=ReviewActionEnum(r.action),
                report_version_id=r.report_version_id,
                outcome=r.outcome,
                note=r.note,
                created_at=r.created_at,
            )
            for r in rows
        ]
