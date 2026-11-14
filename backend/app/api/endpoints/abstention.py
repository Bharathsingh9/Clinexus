"""Abstention Endpoint.

Specification Section 7, Section 5.2 (API-7, EXT-B):
- GET /cases/{id}/abstention: Returns structured abstention decision and explanation.
  If case has no abstention decision, returns 404 with code NO_ABSTENTION_DECISION.
"""

from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, get_db
from backend.app.api.errors import AppAPIException, NotFoundAPIError
from backend.app.persistence.models import (
    AbstentionDecision as AbstentionDecisionModel,
    ClinicalCase as ClinicalCaseModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["abstention"])


class AbstentionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    case_id: str
    execution_id: str
    report_version_id: Optional[str] = None
    abstained: bool
    reason_codes: List[str]
    explanation: str
    affected_claim_ids: List[str]
    affected_sections: List[str]
    missing_evidence: List[str]
    missing_information: List[str]
    suggested_next_steps: List[str]
    requires: List[str]
    kind: str = "INTERPRETATION"
    created_at: datetime


@router.get(
    "/cases/{id}/abstention",
    response_model=AbstentionResponse,
    summary="Retrieve abstention decision for a clinical case",
    dependencies=[Depends(get_current_user)],
)
def get_case_abstention(id: str, db: Session = Depends(get_db)) -> AbstentionResponse:
    """Retrieve explicit clinical abstention decision for a case.

    Raises 404 with code NO_ABSTENTION_DECISION if no abstention was recorded.
    """
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    stmt = (
        select(AbstentionDecisionModel)
        .where(AbstentionDecisionModel.case_id == id)
        .order_by(desc(AbstentionDecisionModel.created_at))
    )
    abst_row = db.scalars(stmt).first()
    if not abst_row:
        raise AppAPIException(
            status_code=404,
            code="NO_ABSTENTION_DECISION",
            message=f"No abstention decision recorded for case '{id}'.",
        )

    def _parse_list(val: Optional[str]) -> List[str]:
        if not val:
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except Exception:
            return [val]

    return AbstentionResponse(
        id=abst_row.id,
        case_id=abst_row.case_id,
        execution_id=abst_row.execution_id,
        report_version_id=abst_row.report_version_id,
        abstained=abst_row.abstained,
        reason_codes=_parse_list(abst_row.reason_codes),
        explanation=abst_row.explanation,
        affected_claim_ids=_parse_list(abst_row.affected_claim_ids),
        affected_sections=_parse_list(abst_row.affected_sections),
        missing_evidence=_parse_list(abst_row.missing_evidence),
        missing_information=_parse_list(abst_row.missing_information),
        suggested_next_steps=_parse_list(abst_row.suggested_next_steps),
        requires=_parse_list(abst_row.requires),
        kind="INTERPRETATION",
        created_at=abst_row.created_at,
    )
