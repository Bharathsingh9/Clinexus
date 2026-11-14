"""Clinical Case Endpoints.

Specification Section 7, Section 8, SAF-4 (API-3):
- POST /cases: Validates free text (1 to CASE_MAX_CHARS), evaluates urgent rules before analysis,
  writes audit event, returns urgent banner. Case starts in DRAFT.
- GET /cases: Lists cases with id, status, created_at, urgent flag, latest execution status.
  STRICT INVARIANT: Contains NO free text.
- GET /cases/{id}: Returns case details including narrative text, findings, and urgent banner.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, get_db, require_clinician_or_admin
from backend.app.api.errors import NotFoundAPIError, ValidationAPIError
from backend.app.core.config import settings
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.models import (
    AuditEvent as AuditEventModel,
    ClinicalCase as ClinicalCaseModel,
    User as UserModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.safety.urgent import evaluate_urgent_rules
from backend.app.schemas.clinical_case import CaseSubmission
from backend.app.schemas.executions import AuditEvent as AuditEventSchema

router = APIRouter(tags=["cases"])


class CreateCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_text: str = Field(..., min_length=1, description="Raw narrative clinical text")


class CreateCaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str
    urgent: bool
    urgent_banner: Optional[str] = None
    created_at: datetime


class CaseSummaryResponse(BaseModel):
    """Case summary strictly omitting any patient narrative or free text."""
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str
    created_at: datetime
    urgent: bool
    latest_execution_status: Optional[str] = None


class CaseDetailResponse(BaseModel):
    """Detailed case representation including narrative text."""
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str
    raw_text: str
    created_at: datetime
    urgent: bool
    urgent_banner: Optional[str] = None
    latest_execution_status: Optional[str] = None


def _check_case_urgent(db: Session, case_id: str) -> Tuple[bool, Optional[str]]:
    """Check whether a case has an urgent presentation match in audit trail."""
    stmt = (
        select(AuditEventModel)
        .where(
            AuditEventModel.resource_id == case_id,
            AuditEventModel.event_type == "URGENT_SYMPTOM_MATCHED",
        )
        .order_by(AuditEventModel.timestamp.desc())
    )
    event = db.scalars(stmt).first()
    if event:
        banner = "Possible urgent presentation: do not delay appropriate urgent human assessment"
        if event.payload and isinstance(event.payload, dict):
            banner = event.payload.get("banner_text", banner)
        return True, banner
    return False, None


def _get_latest_execution_status(db: Session, case_id: str) -> Optional[str]:
    stmt = (
        select(WorkflowExecutionModel)
        .where(WorkflowExecutionModel.case_id == case_id)
        .order_by(desc(WorkflowExecutionModel.started_at))
    )
    latest_exec = db.scalars(stmt).first()
    return latest_exec.status if latest_exec else None


@router.post(
    "/cases",
    response_model=CreateCaseResponse,
    status_code=201,
    summary="Submit a new clinical case",
)
def submit_case(
    payload: CreateCaseRequest,
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
) -> CreateCaseResponse:
    """Submit a new clinical case narrative.

    Validates text length (1 to CASE_MAX_CHARS), evaluates urgent rules before analysis,
    appends audit trail event, and registers case in DRAFT status.
    """
    raw_text = payload.raw_text.strip()
    if len(raw_text) > settings.CASE_MAX_CHARS:
        raise ValidationAPIError(
            f"Case text length ({len(raw_text)}) exceeds limit of {settings.CASE_MAX_CHARS} characters."
        )

    # Validate CaseSubmission schema
    case_sub = CaseSubmission(raw_text=raw_text)

    case_id = str(uuid.uuid4())
    audit_repo = AuditRepository(db)

    # 1. Run urgent rules BEFORE analysis (SAF-4, SPEC Section 8)
    urgent_eval = evaluate_urgent_rules(
        raw_text=raw_text,
        findings=None,
        case_id=case_id,
        audit_repo=audit_repo,
    )

    # 2. Append submission audit event
    submission_event = AuditEventSchema(
        event_id=str(uuid.uuid4()),
        event_type="CASE_SUBMITTED",
        actor_id=current_user.id,
        actor_role=current_user.role,
        resource_type="clinical_case",
        resource_id=case_id,
        payload={
            "case_id": case_id,
            "text_length": len(raw_text),
            "is_urgent": urgent_eval.is_urgent,
        },
        timestamp=datetime.now(timezone.utc),
    )
    audit_repo.append_event(submission_event)

    # 3. Create case in PostgreSQL in DRAFT state
    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    case_row = ClinicalCaseModel(
        id=case_id,
        raw_text_hash=text_hash,
        chief_complaint=raw_text,
        status="DRAFT",
        created_at=datetime.now(timezone.utc),
    )
    db.add(case_row)
    db.commit()
    db.refresh(case_row)

    return CreateCaseResponse(
        id=case_row.id,
        status=case_row.status,
        urgent=urgent_eval.is_urgent,
        urgent_banner=urgent_eval.banner_text if urgent_eval.is_urgent else None,
        created_at=case_row.created_at,
    )


@router.get(
    "/cases",
    response_model=List[CaseSummaryResponse],
    summary="List all clinical cases (No narrative free text)",
    dependencies=[Depends(get_current_user)],
)
def list_cases(db: Session = Depends(get_db)) -> List[CaseSummaryResponse]:
    """Retrieve all cases. Strictly omits narrative free text."""
    stmt = select(ClinicalCaseModel).order_by(desc(ClinicalCaseModel.created_at))
    case_rows = list(db.scalars(stmt).all())

    summaries: List[CaseSummaryResponse] = []
    for c in case_rows:
        is_urgent, _ = _check_case_urgent(db, c.id)
        exec_status = _get_latest_execution_status(db, c.id)
        summaries.append(
            CaseSummaryResponse(
                id=c.id,
                status=c.status,
                created_at=c.created_at,
                urgent=is_urgent,
                latest_execution_status=exec_status,
            )
        )
    return summaries


@router.get(
    "/cases/{id}",
    response_model=CaseDetailResponse,
    summary="Retrieve clinical case detail",
    dependencies=[Depends(get_current_user)],
)
def get_case(id: str, db: Session = Depends(get_db)) -> CaseDetailResponse:
    """Retrieve case detail including narrative text and urgent banner."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    is_urgent, banner = _check_case_urgent(db, case_row.id)
    exec_status = _get_latest_execution_status(db, case_row.id)

    return CaseDetailResponse(
        id=case_row.id,
        status=case_row.status,
        raw_text=case_row.chief_complaint,
        created_at=case_row.created_at,
        urgent=is_urgent,
        urgent_banner=banner,
        latest_execution_status=exec_status,
    )
