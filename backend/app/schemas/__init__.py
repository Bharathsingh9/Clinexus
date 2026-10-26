"""Clinexus Pydantic v2 Schema Registry.

Specification Section 6 & Appendix A.
"""

from backend.app.schemas.clinical_case import (
    CandidateHypothesis,
    ClarificationRequest,
    ClarificationStatusEnum,
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)
from backend.app.schemas.evidence import (
    EvidenceChunk,
    EvidenceDocument,
    EvidenceSourceEnum,
    ExternalEvidenceSet,
    QueryStatusEnum,
    RetrievalQuery,
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)
from backend.app.schemas.executions import (
    AgentExecutionRecord,
    AgentStatusEnum,
    AuditEvent,
    OptimizationIteration,
    WorkflowResult,
    WorkflowState,
)
from backend.app.schemas.ids import (
    build_evidence_id,
    parse_evidence_id,
    validate_evidence_id,
)
from backend.app.schemas.reports import (
    AbstentionDecision,
    AbstentionReasonCode,
    AbstentionRequirement,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    HumanReview,
    ReportStatusEnum,
    ReviewActionEnum,
    VerificationMethodEnum,
)

__all__ = [
    # IDs
    "build_evidence_id",
    "parse_evidence_id",
    "validate_evidence_id",
    # Clinical Case
    "SourceSpan",
    "PolarityEnum",
    "OriginEnum",
    "ClarificationStatusEnum",
    "StructuredSymptom",
    "PatientContext",
    "ClarificationRequest",
    "CandidateHypothesis",
    "ClinicalCase",
    # Evidence
    "EvidenceSourceEnum",
    "QueryStatusEnum",
    "RetrievalQuery",
    "EvidenceDocument",
    "EvidenceChunk",
    "RetrievedPassageResult",
    "RetrievedEvidenceSet",
    "ExternalEvidenceSet",
    # Reports & Verification
    "ClaimVerificationStatusEnum",
    "VerificationMethodEnum",
    "ReportStatusEnum",
    "ReviewActionEnum",
    "AbstentionReasonCode",
    "AbstentionRequirement",
    "CandidateClaim",
    "ClaimVerification",
    "CandidateClinicalReport",
    "AbstentionDecision",
    "HumanReview",
    # Executions & Workflow
    "AgentStatusEnum",
    "AgentExecutionRecord",
    "OptimizationIteration",
    "AuditEvent",
    "WorkflowResult",
    "WorkflowState",
]
