"""Report and Claim Endpoints.

Specification Section 7, Section 5.1, 5.2, 5.3 (API-5, EXT-A, EXT-B, EXT-C):
- GET /cases/{id}/report: Latest report version (sections, claims, urgent banner, benchmark label, abstention).
- GET /cases/{id}/reports: List of all report versions for a case.
- GET /reports/{version_id}/claims: List claims with verification status, evidence IDs, limitations,
  material and safety-critical flags.
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
from backend.app.api.errors import NotFoundAPIError
from backend.app.persistence.models import (
    AbstentionDecision as AbstentionDecisionModel,
    AuditEvent as AuditEventModel,
    CandidateClaim as CandidateClaimModel,
    ClaimVerification as ClaimVerificationModel,
    ClinicalCase as ClinicalCaseModel,
    ProvenanceEdge as ProvenanceEdgeModel,
    ReportVersion as ReportVersionModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.safety.gate import is_material, is_safety_critical

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


class ClaimItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    text: str
    category: str
    section: str
    status: str
    supporting_evidence_ids: List[str]
    contradicting_evidence_ids: List[str]
    limitations: Optional[str] = None
    is_material: bool
    is_safety_critical: bool
    kind: str = "INTERPRETATION"


class ReportVersionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    case_id: str
    execution_id: str
    version_number: int
    status: str
    confidence_score: float
    summary: str
    created_at: datetime


class LatestReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_version_id: str
    case_id: str
    version_number: int
    status: str
    confidence_score: float
    summary: str
    sections: Dict[str, Any]
    claims: List[ClaimItemResponse]
    urgent_banner: Optional[str] = None
    benchmark_label: Optional[str] = None
    abstention: Optional[Dict[str, Any]] = None
    kind: str = "INTERPRETATION"
    created_at: datetime


def _get_claims_for_report(db: Session, report_version_id: str) -> List[ClaimItemResponse]:
    """Retrieve candidate claims with their verifications and linked evidence IDs."""
    stmt = (
        select(CandidateClaimModel)
        .where(CandidateClaimModel.report_version_id == report_version_id)
        .order_by(CandidateClaimModel.id.asc())
    )
    claim_rows = list(db.scalars(stmt).all())

    # Pre-fetch provenance edges for evidence ID mapping
    edge_stmt = select(ProvenanceEdgeModel).where(
        ProvenanceEdgeModel.report_version_id == report_version_id
    )
    edge_rows = list(db.scalars(edge_stmt).all())

    supporting_map: Dict[str, List[str]] = {}
    contradicting_map: Dict[str, List[str]] = {}

    for e in edge_rows:
        if not e.claim_id or not e.evidence_chunk_id:
            continue
        rel = (e.relation or "").upper()
        if rel in ("SUPPORTS", "PARTIALLY_SUPPORTS"):
            supporting_map.setdefault(e.claim_id, []).append(e.evidence_chunk_id)
        elif rel == "CONTRADICTS":
            contradicting_map.setdefault(e.claim_id, []).append(e.evidence_chunk_id)

    results: List[ClaimItemResponse] = []
    for c in claim_rows:
        verif_stmt = (
            select(ClaimVerificationModel)
            .where(ClaimVerificationModel.claim_id == c.id)
            .order_by(desc(ClaimVerificationModel.verified_at))
        )
        verif = db.scalars(verif_stmt).first()

        cat = verif.category if verif and verif.category else "clinical_observation"
        stat = verif.status if verif and verif.status else "UNVERIFIABLE"
        limits = verif.limitations if verif else None

        sup_ids = supporting_map.get(c.id, [])
        contra_ids = contradicting_map.get(c.id, [])

        mat_flag = c.is_material if c.is_material is not None else (cat.lower() != "background")
        safe_flag = c.is_safety_critical if c.is_safety_critical is not None else False

        results.append(
            ClaimItemResponse(
                claim_id=c.id,
                text=c.claim_text,
                category=cat,
                section=c.section,
                status=stat,
                supporting_evidence_ids=sup_ids,
                contradicting_evidence_ids=contra_ids,
                limitations=limits,
                is_material=mat_flag,
                is_safety_critical=safe_flag,
                kind="INTERPRETATION",
            )
        )
    return results


@router.get(
    "/cases/{id}/report",
    response_model=LatestReportResponse,
    summary="Retrieve latest report for a clinical case",
    dependencies=[Depends(get_current_user)],
)
def get_latest_report(id: str, db: Session = Depends(get_db)) -> LatestReportResponse:
    """Retrieve the latest synthesized or abstained clinical report version for a case."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    stmt = (
        select(ReportVersionModel)
        .where(ReportVersionModel.case_id == id)
        .order_by(desc(ReportVersionModel.version_number))
    )
    report_row = db.scalars(stmt).first()
    if not report_row:
        raise NotFoundAPIError(f"No clinical report found for case '{id}'.")

    # Fetch claims
    claims = _get_claims_for_report(db, report_row.id)

    # Check urgent presentation
    urg_stmt = (
        select(AuditEventModel)
        .where(
            AuditEventModel.resource_id == id,
            AuditEventModel.event_type == "URGENT_SYMPTOM_MATCHED",
        )
        .order_by(AuditEventModel.timestamp.desc())
    )
    urg_event = db.scalars(urg_stmt).first()
    urgent_banner = None
    if urg_event and urg_event.payload and isinstance(urg_event.payload, dict):
        urgent_banner = urg_event.payload.get(
            "banner_text",
            "Possible urgent presentation: do not delay appropriate urgent human assessment",
        )
    elif urg_event:
        urgent_banner = "Possible urgent presentation: do not delay appropriate urgent human assessment"

    # Check benchmark mode from execution
    exec_row = db.get(WorkflowExecutionModel, report_row.execution_id)
    benchmark_label = "BENCHMARK, NOT CLINICAL USE" if (exec_row and exec_row.is_benchmark) else None

    # Check abstention
    abst_data = None
    if report_row.status == "ABSTAINED":
        abst_stmt = (
            select(AbstentionDecisionModel)
            .where(AbstentionDecisionModel.report_version_id == report_row.id)
            .order_by(desc(AbstentionDecisionModel.created_at))
        )
        abst_row = db.scalars(abst_stmt).first()
        if abst_row:
            abst_data = {
                "abstained": abst_row.abstained,
                "reason_codes": json.loads(abst_row.reason_codes),
                "explanation": abst_row.explanation,
                "affected_claim_ids": json.loads(abst_row.affected_claim_ids),
                "affected_sections": json.loads(abst_row.affected_sections),
                "missing_evidence": json.loads(abst_row.missing_evidence),
                "missing_information": json.loads(abst_row.missing_information),
                "suggested_next_steps": json.loads(abst_row.suggested_next_steps),
                "requires": json.loads(abst_row.requires),
                "kind": "INTERPRETATION",
            }

    # Construct sections dictionary
    sections: Dict[str, Any] = {}
    try:
        sections = json.loads(report_row.summary)
        if not isinstance(sections, dict):
            sections = {"case_summary": report_row.summary}
    except Exception:
        sections = {"case_summary": report_row.summary}

    return LatestReportResponse(
        report_version_id=report_row.id,
        case_id=report_row.case_id,
        version_number=report_row.version_number,
        status=report_row.status,
        confidence_score=report_row.confidence_score,
        summary=report_row.summary,
        sections=sections,
        claims=claims,
        urgent_banner=urgent_banner,
        benchmark_label=benchmark_label,
        abstention=abst_data,
        kind="INTERPRETATION",
        created_at=report_row.created_at,
    )


@router.get(
    "/cases/{id}/reports",
    response_model=List[ReportVersionSummaryResponse],
    summary="List report versions for a case",
    dependencies=[Depends(get_current_user)],
)
def list_reports_for_case(id: str, db: Session = Depends(get_db)) -> List[ReportVersionSummaryResponse]:
    """Retrieve all report version summaries for a case in chronological order."""
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    stmt = (
        select(ReportVersionModel)
        .where(ReportVersionModel.case_id == id)
        .order_by(ReportVersionModel.version_number.asc())
    )
    rows = list(db.scalars(stmt).all())
    return [
        ReportVersionSummaryResponse(
            id=r.id,
            case_id=r.case_id,
            execution_id=r.execution_id,
            version_number=r.version_number,
            status=r.status,
            confidence_score=r.confidence_score,
            summary=r.summary,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get(
    "/reports/{version_id}/claims",
    response_model=List[ClaimItemResponse],
    summary="List candidate claims for a report version",
    dependencies=[Depends(get_current_user)],
)
def get_report_claims(version_id: str, db: Session = Depends(get_db)) -> List[ClaimItemResponse]:
    """Retrieve all claims and verification statuses for a specific report version."""
    report_row = db.get(ReportVersionModel, version_id)
    if not report_row:
        raise NotFoundAPIError(f"ReportVersion '{version_id}' not found.")

    return _get_claims_for_report(db, version_id)
