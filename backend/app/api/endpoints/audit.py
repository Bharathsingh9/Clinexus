"""Audit Trail Endpoints.

Specification Section 7, Section 5.3, EXT-C (API-9):
- GET /cases/{id}/audit: Returns chronological immutable audit trail events for a case,
  with report_version_id and event details.
"""

from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, get_db
from backend.app.api.errors import NotFoundAPIError
from backend.app.persistence.models import (
    AuditEvent as AuditEventModel,
    ClinicalCase as ClinicalCaseModel,
    ReportVersion as ReportVersionModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audit"])


class AuditEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    event_type: str
    actor_id: str
    actor_role: str
    resource_type: str
    resource_id: str
    report_version_id: Optional[str] = None
    payload: Dict[str, Any]
    timestamp: datetime
    kind: str = "HUMAN"


@router.get(
    "/cases/{id}/audit",
    response_model=List[AuditEventResponse],
    summary="Retrieve ordered audit trail for a clinical case",
    dependencies=[Depends(get_current_user)],
)
def get_case_audit_trail(id: str, db: Session = Depends(get_db)) -> List[AuditEventResponse]:
    """Retrieve immutable audit events for a case ordered chronologically."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    # Find all report_versions belonging to this case to associate report_version_ids
    rep_stmt = select(ReportVersionModel.id).where(ReportVersionModel.case_id == id)
    report_version_ids = set(db.scalars(rep_stmt).all())

    # Query events linked to this case or any of its report versions
    stmt = (
        select(AuditEventModel)
        .where(
            or_(
                AuditEventModel.resource_id == id,
                AuditEventModel.resource_id.in_(report_version_ids) if report_version_ids else False,
                AuditEventModel.payload.like(f"%{id}%"),
            )
        )
        .order_by(AuditEventModel.timestamp.asc())
    )
    events = list(db.scalars(stmt).all())

    results: List[AuditEventResponse] = []
    for e in events:
        parsed_payload: Dict[str, Any] = {}
        if e.payload:
            try:
                parsed_payload = json.loads(e.payload) if isinstance(e.payload, str) else e.payload
            except Exception:
                parsed_payload = {"raw": e.payload}

        # Resolve report_version_id
        r_ver_id = None
        if e.resource_type == "report_version" and e.resource_id in report_version_ids:
            r_ver_id = e.resource_id
        elif "report_version_id" in parsed_payload:
            r_ver_id = parsed_payload.get("report_version_id")

        results.append(
            AuditEventResponse(
                event_id=e.id,
                event_type=e.event_type,
                actor_id=e.actor_id,
                actor_role=e.actor_role,
                resource_type=e.resource_type,
                resource_id=e.resource_id,
                report_version_id=r_ver_id,
                payload=parsed_payload,
                timestamp=e.timestamp,
                kind="HUMAN",
            )
        )

    return results
