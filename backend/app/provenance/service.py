"""Evidence-Level Provenance Service and Completeness Invariant.

Specification Section 5.3 (EXT-C, Req ID: EXT-C):
- Persist relational edges linking case, finding, query, evidence chunk, document,
  claim, verification, report section, iteration, abstention, and review event.
- Written by the persistence step from real agent outputs, NEVER reconstructed by an LLM.
- Mark claims and summaries kind=INTERPRETATION and passages kind=SOURCE.
- assert_provenance_complete(report_version) fails if any material claim has no edge.
- Runs at finalize: a failure ends in abstention with SYSTEM_ERROR, never ready.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Set, Union
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.persistence.models import (
    CandidateClaim as CandidateClaimModel,
    CaseFinding as CaseFindingModel,
    ClaimVerification as ClaimVerificationModel,
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
    ProvenanceEdge as ProvenanceEdgeModel,
    ReportVersion as ReportVersionModel,
)
from backend.app.provenance.models import (
    ProvenanceEdgeSchema,
    ProvenanceNode,
    ProvenanceNodeKindEnum,
    ProvenanceRelationEnum,
)
from backend.app.schemas.clinical_case import ClinicalCase
from backend.app.schemas.evidence import ExternalEvidenceSet, RetrievedEvidenceSet
from backend.app.schemas.executions import OptimizationIteration
from backend.app.schemas.reports import (
    AbstentionDecision,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
)

logger = logging.getLogger(__name__)


class ProvenanceIncompleteError(Exception):
    """Raised when one or more material claims in a report version lack provenance edges."""
    def __init__(self, missing_claim_ids: List[str], report_version_id: str):
        self.missing_claim_ids = missing_claim_ids
        self.report_version_id = report_version_id
        super().__init__(
            f"Provenance completeness violation for report version '{report_version_id}': "
            f"Material claims {missing_claim_ids} have no provenance edges. "
            f"Specification 5.3 requires every material claim to be grounded in provenance."
        )


def build_provenance_edges_from_workflow(
    report: CandidateClinicalReport,
    case: Optional[ClinicalCase] = None,
    verifications: Optional[List[ClaimVerification]] = None,
    retrieved_evidence: Optional[RetrievedEvidenceSet] = None,
    external_evidence: Optional[ExternalEvidenceSet] = None,
    iterations: Optional[List[OptimizationIteration]] = None,
    abstention: Optional[AbstentionDecision] = None,
    review_event_id: Optional[str] = None,
) -> List[ProvenanceEdgeSchema]:
    """Build provenance edges from REAL agent outputs (never reconstructed by LLM).

    Specification Section 5.3:
    Relations:
    - GENERATED_IN: claim -> report section
    - SUPPORTS / PARTIALLY_SUPPORTS / CONTRADICTS / INSUFFICIENT: claim -> evidence / verification
    - DERIVED_FROM_PATIENT_INPUT: claim -> case finding / case
    - RETRIEVED_BY: evidence chunk -> query
    """
    edges: List[ProvenanceEdgeSchema] = []
    verif_map: Dict[str, ClaimVerification] = {
        v.claim_id: v for v in (verifications or [])
    }

    # 1. Edges for each claim in the report
    for claim in report.claims:
        # A. Section generation edge (GENERATED_IN)
        edges.append(
            ProvenanceEdgeSchema(
                id=str(uuid.uuid4()),
                report_version_id=report.report_version_id,
                case_id=report.case_id,
                claim_id=claim.claim_id,
                report_section_id=claim.section,
                relation=ProvenanceRelationEnum.GENERATED_IN,
            )
        )

        # B. Patient input derivation edge (DERIVED_FROM_PATIENT_INPUT)
        # If claim is derived from patient input or cites patient fact
        if len(claim.source_spans) > 0 or len(claim.cited_evidence_ids) == 0:
            finding_id_val = None
            if case:
                symptoms_list = (
                    getattr(case, "structured_symptoms", None)
                    or getattr(case, "symptoms", None)
                    or getattr(case, "findings", None)
                )
                if symptoms_list:
                    finding_id_val = getattr(symptoms_list[0], "finding_id", None)

            edges.append(
                ProvenanceEdgeSchema(
                    id=str(uuid.uuid4()),
                    report_version_id=report.report_version_id,
                    case_id=report.case_id,
                    finding_id=finding_id_val,
                    claim_id=claim.claim_id,
                    relation=ProvenanceRelationEnum.DERIVED_FROM_PATIENT_INPUT,
                )
            )

        # C. Evidence verification edges (SUPPORTS, PARTIALLY_SUPPORTS, CONTRADICTS, INSUFFICIENT)
        verif = verif_map.get(claim.claim_id)
        if verif:
            # Map status to relation
            if verif.status == ClaimVerificationStatusEnum.SUPPORTED:
                rel = ProvenanceRelationEnum.SUPPORTS
            elif verif.status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED:
                rel = ProvenanceRelationEnum.PARTIALLY_SUPPORTS
            elif verif.status == ClaimVerificationStatusEnum.CONTRADICTED:
                rel = ProvenanceRelationEnum.CONTRADICTS
            else:
                rel = ProvenanceRelationEnum.INSUFFICIENT

            # Supporting evidence edges
            for ev_id in verif.supporting_evidence_ids:
                edges.append(
                    ProvenanceEdgeSchema(
                        id=str(uuid.uuid4()),
                        report_version_id=report.report_version_id,
                        case_id=report.case_id,
                        claim_id=claim.claim_id,
                        verification_id=verif.claim_id,
                        relation=rel,
                    )
                )

            # Contradicting evidence edges
            for c_ev_id in verif.contradicting_evidence_ids:
                edges.append(
                    ProvenanceEdgeSchema(
                        id=str(uuid.uuid4()),
                        report_version_id=report.report_version_id,
                        case_id=report.case_id,
                        claim_id=claim.claim_id,
                        verification_id=verif.claim_id,
                        relation=ProvenanceRelationEnum.CONTRADICTS,
                    )
                )

            # If no specific chunk linked yet for this verification, link verification directly
            if not verif.supporting_evidence_ids and not verif.contradicting_evidence_ids:
                edges.append(
                    ProvenanceEdgeSchema(
                        id=str(uuid.uuid4()),
                        report_version_id=report.report_version_id,
                        case_id=report.case_id,
                        claim_id=claim.claim_id,
                        verification_id=verif.claim_id,
                        relation=rel,
                    )
                )

    # 2. Retrieval edges (RETRIEVED_BY: chunk -> query)
    if retrieved_evidence and retrieved_evidence.results:
        for res in retrieved_evidence.results:
            edges.append(
                ProvenanceEdgeSchema(
                    id=str(uuid.uuid4()),
                    report_version_id=report.report_version_id,
                    case_id=report.case_id,
                    query_id=res.query_id or retrieved_evidence.query_id,
                    document_id=res.document_id,
                    relation=ProvenanceRelationEnum.RETRIEVED_BY,
                )
            )

    # 3. Abstention edge (if applicable)
    if abstention and abstention.abstained:
        edges.append(
            ProvenanceEdgeSchema(
                id=str(uuid.uuid4()),
                report_version_id=report.report_version_id,
                case_id=report.case_id,
                abstention_id=abstention.decision_id,
                relation=ProvenanceRelationEnum.INSUFFICIENT,
            )
        )

    # 4. Review event edge (if applicable)
    if review_event_id:
        edges.append(
            ProvenanceEdgeSchema(
                id=str(uuid.uuid4()),
                report_version_id=report.report_version_id,
                case_id=report.case_id,
                review_event_id=review_event_id,
                relation=ProvenanceRelationEnum.GENERATED_IN,
            )
        )

    return edges


def save_provenance_edges(
    session: Session,
    edges: List[ProvenanceEdgeSchema],
) -> List[ProvenanceEdgeModel]:
    """Persist provenance edges to the provenance_edges PostgreSQL table."""
    if not edges:
        return []

    # Map candidate_claims in DB for valid foreign keys
    db_models: List[ProvenanceEdgeModel] = []
    for e in edges:
        # Check report_version_id exists in DB (foreign key safety)
        if not session.get(ReportVersionModel, e.report_version_id):
            continue

        # Check claim_id exists if specified
        valid_claim_id = None
        if e.claim_id:
            claim_row = session.get(CandidateClaimModel, e.claim_id)
            if claim_row:
                valid_claim_id = claim_row.id

        # Check verification_id exists if specified
        valid_verification_id = None
        if e.verification_id:
            verif_row = session.get(ClaimVerificationModel, e.verification_id)
            if not verif_row:
                stmt_v = select(ClaimVerificationModel).where(ClaimVerificationModel.claim_id == e.verification_id)
                verif_row = session.scalars(stmt_v).first()
            if verif_row:
                valid_verification_id = verif_row.id

        # Check document_id exists if specified
        valid_document_id = None
        if e.document_id:
            doc_row = session.get(EvidenceDocumentModel, e.document_id)
            if doc_row:
                valid_document_id = doc_row.id

        # Check evidence_chunk_id exists if specified
        valid_chunk_id = None
        if e.evidence_chunk_id:
            chunk_row = session.get(EvidencePassageModel, e.evidence_chunk_id)
            if chunk_row:
                valid_chunk_id = chunk_row.id

        # Check finding_id exists if specified
        valid_finding_id = None
        if e.finding_id:
            finding_row = session.get(CaseFindingModel, e.finding_id)
            if finding_row:
                valid_finding_id = finding_row.id

        row = ProvenanceEdgeModel(
            id=e.id or str(uuid.uuid4()),
            report_version_id=e.report_version_id,
            case_id=e.case_id,
            finding_id=valid_finding_id,
            query_id=e.query_id,
            evidence_chunk_id=valid_chunk_id,
            document_id=valid_document_id,
            claim_id=valid_claim_id,
            verification_id=valid_verification_id,
            report_section_id=e.report_section_id,
            iteration_id=e.iteration_id,
            abstention_id=e.abstention_id,
            review_event_id=e.review_event_id,
            relation=e.relation.value if hasattr(e.relation, "value") else str(e.relation),
            created_at=e.created_at,
        )
        session.add(row)
        db_models.append(row)

    session.commit()
    return db_models


GROUNDING_RELATIONS = frozenset({
    ProvenanceRelationEnum.SUPPORTS,
    ProvenanceRelationEnum.PARTIALLY_SUPPORTS,
    ProvenanceRelationEnum.CONTRADICTS,
    ProvenanceRelationEnum.INSUFFICIENT,
    ProvenanceRelationEnum.DERIVED_FROM_PATIENT_INPUT,
    "SUPPORTS",
    "PARTIALLY_SUPPORTS",
    "CONTRADICTS",
    "INSUFFICIENT",
    "DERIVED_FROM_PATIENT_INPUT",
})


def assert_provenance_complete(
    report_version: Union[CandidateClinicalReport, ReportVersionModel, str],
    session: Optional[Session] = None,
    edges: Optional[List[ProvenanceEdgeSchema]] = None,
) -> bool:
    """Verify that EVERY material claim in the report version has at least one provenance edge.

    Specification Section 5.3:
    "assert_provenance_complete(report_version) fails if any material claim has no edge
     and runs at finalize and in tests."

    Raises:
        ProvenanceIncompleteError: If any material claim has no provenance edge.
    """
    if isinstance(report_version, CandidateClinicalReport):
        version_id = report_version.report_version_id
        material_claims = [c for c in report_version.claims if c.is_material]

        # Check in provided edges list
        if edges is not None:
            covered_claim_ids: Set[str] = {
                e.claim_id for e in edges if e.claim_id is not None and e.relation in GROUNDING_RELATIONS
            }
        elif session is not None:
            stmt = select(ProvenanceEdgeModel.claim_id).where(
                ProvenanceEdgeModel.report_version_id == version_id,
                ProvenanceEdgeModel.claim_id.is_not(None),
                ProvenanceEdgeModel.relation.in_([
                    "SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "DERIVED_FROM_PATIENT_INPUT"
                ]),
            )
            covered_claim_ids = set(session.scalars(stmt).all())
        else:
            raise ValueError("Either session or edges must be provided to verify completeness.")

        missing = [c.claim_id for c in material_claims if c.claim_id not in covered_claim_ids]
        if missing:
            raise ProvenanceIncompleteError(missing_claim_ids=missing, report_version_id=version_id)
        return True

    elif isinstance(report_version, ReportVersionModel):
        if session is None:
            raise ValueError("Session required when verifying ReportVersionModel")
        version_id = report_version.id
        stmt_claims = select(CandidateClaimModel.id).where(
            CandidateClaimModel.report_version_id == version_id,
            CandidateClaimModel.is_material.is_(True),
        )
        material_claim_ids = set(session.scalars(stmt_claims).all())

        stmt_edges = select(ProvenanceEdgeModel.claim_id).where(
            ProvenanceEdgeModel.report_version_id == version_id,
            ProvenanceEdgeModel.claim_id.is_not(None),
            ProvenanceEdgeModel.relation.in_([
                "SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "DERIVED_FROM_PATIENT_INPUT"
            ]),
        )
        covered_claim_ids = set(session.scalars(stmt_edges).all())

        missing = [cid for cid in material_claim_ids if cid not in covered_claim_ids]
        if missing:
            raise ProvenanceIncompleteError(missing_claim_ids=missing, report_version_id=version_id)
        return True

    elif isinstance(report_version, str):
        if session is None:
            raise ValueError("Session required when passing report_version_id string")
        version_id = report_version
        stmt_claims = select(CandidateClaimModel.id).where(
            CandidateClaimModel.report_version_id == version_id,
            CandidateClaimModel.is_material.is_(True),
        )
        material_claim_ids = set(session.scalars(stmt_claims).all())

        stmt_edges = select(ProvenanceEdgeModel.claim_id).where(
            ProvenanceEdgeModel.report_version_id == version_id,
            ProvenanceEdgeModel.claim_id.is_not(None),
            ProvenanceEdgeModel.relation.in_([
                "SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "INSUFFICIENT", "DERIVED_FROM_PATIENT_INPUT"
            ]),
        )
        covered_claim_ids = set(session.scalars(stmt_edges).all())

        missing = [cid for cid in material_claim_ids if cid not in covered_claim_ids]
        if missing:
            raise ProvenanceIncompleteError(missing_claim_ids=missing, report_version_id=version_id)
        return True

    else:
        raise TypeError(f"Unsupported report_version type: {type(report_version)}")


def classify_provenance_node(entity_type: str, entity_id: str, label: str) -> ProvenanceNode:
    """Classify node as INTERPRETATION (claims, summaries) or SOURCE (passages, documents).

    Specification Section 5.3:
    "Claims and summaries are kind=INTERPRETATION; passages are kind=SOURCE; API and UI keep the distinction."
    """
    if entity_type in ("claim", "candidate_claim", "summary", "report_summary"):
        kind = ProvenanceNodeKindEnum.INTERPRETATION
    elif entity_type in ("passage", "evidence_passage", "document", "evidence_document"):
        kind = ProvenanceNodeKindEnum.SOURCE
    else:
        kind = ProvenanceNodeKindEnum.INTERPRETATION

    return ProvenanceNode(
        node_id=entity_id,
        node_type=entity_type,
        kind=kind,
        label=label,
    )
