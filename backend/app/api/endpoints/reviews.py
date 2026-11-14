"""Clinician Review and HITL Endpoints.

Specification Section 7, Section 5.4, EXT-D (API-8):
- POST /cases/{id}/reviews: Executes clinician review action (MARK_REVIEWED, REQUEST_REVISION,
  REQUEST_EVIDENCE_REFRESH, REQUEST_CLARIFICATION, ADD_NOTE) via ReviewService.
  Reviewer identity and role from JWT token only.
  Validates that report_version_id belongs to the case.
  MARK_REVIEWED label is strictly "Reviewed by clinician".
- POST /cases/{id}/clarifications: Records REQUEST_CLARIFICATION review action and note.
- POST /cases/{id}/refresh-evidence: Records REQUEST_EVIDENCE_REFRESH and transitions to REVISION_REQUESTED.
- POST /cases/{id}/revisions: Records REQUEST_REVISION and transitions to REVISION_REQUESTED.
STRICT INVARIANT: None of these endpoints starts a new analysis. (A new run starts only through analyze).
"""

from datetime import datetime
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, require_clinician_or_admin
from backend.app.api.errors import (
    ConflictAPIError,
    NotFoundAPIError,
    ValidationAPIError,
)
from backend.app.orchestration.review_service import ReviewService
from backend.app.persistence.models import (
    ClinicalCase as ClinicalCaseModel,
    ReportVersion as ReportVersionModel,
    User as UserModel,
)
from backend.app.schemas.reports import ReviewActionEnum

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reviews"])


class ReviewSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(..., description="REQUEST_CLARIFICATION | REQUEST_EVIDENCE_REFRESH | REQUEST_REVISION | MARK_REVIEWED | ADD_NOTE")
    report_version_id: str
    note: Optional[str] = None


class ActionNoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: Optional[str] = None
    report_version_id: Optional[str] = None


class ReviewRecordResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    case_id: str
    report_version_id: str
    reviewer_id: str
    reviewer_role: str
    action: str
    outcome: str
    note: Optional[str] = None
    created_at: datetime
    kind: str = "HUMAN"


class ActionTransitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    status: str
    action: str
    outcome: str
    report_version_id: str
    kind: str = "HUMAN"


def _resolve_report_version(db: Session, case_id: str, requested_version_id: Optional[str]) -> ReportVersionModel:
    """Resolve and validate that the report version exists and belongs to the given case."""
    if requested_version_id:
        report_row = db.get(ReportVersionModel, requested_version_id)
        if not report_row:
            raise NotFoundAPIError(f"ReportVersion '{requested_version_id}' not found.")
        if report_row.case_id != case_id:
            raise ConflictAPIError(
                f"ReportVersion '{requested_version_id}' does not belong to case '{case_id}'."
            )
        return report_row

    # Fall back to latest version
    stmt = (
        select(ReportVersionModel)
        .where(ReportVersionModel.case_id == case_id)
        .order_by(desc(ReportVersionModel.version_number))
    )
    latest = db.scalars(stmt).first()
    if not latest:
        raise NotFoundAPIError(f"No report version found for case '{case_id}'.")
    return latest


@router.post(
    "/cases/{id}/reviews",
    response_model=ReviewRecordResponse,
    status_code=201,
    summary="Record a clinician review action on a report",
)
def record_review(
    id: str,
    payload: ReviewSubmissionRequest,
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
) -> ReviewRecordResponse:
    """Record an append-only clinician review action. Reviewer identity from token only."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    # Validate action enum
    try:
        action_enum = ReviewActionEnum(payload.action.strip().upper())
    except ValueError:
        valid_actions = [e.value for e in ReviewActionEnum]
        raise ValidationAPIError(
            f"Invalid review action '{payload.action}'. Allowed: {valid_actions}"
        )

    # Validate report version belongs to case
    report_row = _resolve_report_version(db, id, payload.report_version_id)

    review_svc = ReviewService(session=db)
    review_schema = review_svc.record_review_action(
        case_id=id,
        report_version_id=report_row.id,
        reviewer_id=current_user.id,
        reviewer_role=current_user.role,
        action=action_enum,
        note=payload.note,
    )

    return ReviewRecordResponse(
        review_id=review_schema.review_id,
        case_id=id,
        report_version_id=review_schema.report_version_id,
        reviewer_id=review_schema.reviewer_id,
        reviewer_role=review_schema.reviewer_role,
        action=review_schema.action.value,
        outcome=review_schema.outcome,
        note=review_schema.note,
        created_at=review_schema.created_at,
        kind="HUMAN",
    )


@router.post(
    "/cases/{id}/clarifications",
    response_model=ActionTransitionResponse,
    summary="Record clinician clarification request",
)
def request_clarification(
    id: str,
    payload: ActionNoteRequest,
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
) -> ActionTransitionResponse:
    """Record a clarification request action with note. Does NOT trigger analysis."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    report_row = _resolve_report_version(db, id, payload.report_version_id)
    review_svc = ReviewService(session=db)

    review_schema = review_svc.record_review_action(
        case_id=id,
        report_version_id=report_row.id,
        reviewer_id=current_user.id,
        reviewer_role=current_user.role,
        action=ReviewActionEnum.REQUEST_CLARIFICATION,
        note=payload.note,
    )

    db.refresh(case_row)
    return ActionTransitionResponse(
        case_id=id,
        status=case_row.status,
        action="REQUEST_CLARIFICATION",
        outcome=review_schema.outcome,
        report_version_id=report_row.id,
        kind="HUMAN",
    )


@router.post(
    "/cases/{id}/refresh-evidence",
    response_model=ActionTransitionResponse,
    summary="Request evidence refresh on case report",
)
def request_evidence_refresh(
    id: str,
    payload: ActionNoteRequest,
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
) -> ActionTransitionResponse:
    """Record evidence refresh request and transition to REVISION_REQUESTED. Does NOT trigger analysis."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    report_row = _resolve_report_version(db, id, payload.report_version_id)
    review_svc = ReviewService(session=db)

    review_schema = review_svc.record_review_action(
        case_id=id,
        report_version_id=report_row.id,
        reviewer_id=current_user.id,
        reviewer_role=current_user.role,
        action=ReviewActionEnum.REQUEST_EVIDENCE_REFRESH,
        note=payload.note,
    )

    db.refresh(case_row)
    return ActionTransitionResponse(
        case_id=id,
        status=case_row.status,
        action="REQUEST_EVIDENCE_REFRESH",
        outcome=review_schema.outcome,
        report_version_id=report_row.id,
        kind="HUMAN",
    )


@router.post(
    "/cases/{id}/revisions",
    response_model=ActionTransitionResponse,
    summary="Request report revision",
)
def request_revision(
    id: str,
    payload: ActionNoteRequest,
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
) -> ActionTransitionResponse:
    """Record revision request and transition to REVISION_REQUESTED. Does NOT trigger analysis."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    report_row = _resolve_report_version(db, id, payload.report_version_id)
    review_svc = ReviewService(session=db)

    review_schema = review_svc.record_review_action(
        case_id=id,
        report_version_id=report_row.id,
        reviewer_id=current_user.id,
        reviewer_role=current_user.role,
        action=ReviewActionEnum.REQUEST_REVISION,
        note=payload.note,
    )

    db.refresh(case_row)
    return ActionTransitionResponse(
        case_id=id,
        status=case_row.status,
        action="REQUEST_REVISION",
        outcome=review_schema.outcome,
        report_version_id=report_row.id,
        kind="HUMAN",
    )
