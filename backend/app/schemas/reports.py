"""Reports, Claims, Verifications, Abstentions, and Reviews Schemas.

Specification (SPEC 5.1, 5.2, 5.4, 6).
Models: CandidateClaim, ClaimVerification, CandidateClinicalReport, AbstentionDecision, HumanReview.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.schemas.clinical_case import SourceSpan


class ClaimVerificationStatusEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIABLE = "UNVERIFIABLE"


class VerificationMethodEnum(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"
    HYBRID = "HYBRID"


class ReportStatusEnum(str, Enum):
    DRAFT = "DRAFT"
    ANALYZING = "ANALYZING"
    EVIDENCE_REVIEW_REQUIRED = "EVIDENCE_REVIEW_REQUIRED"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    ABSTAINED = "ABSTAINED"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEWED = "REVIEWED"
    REVISION_REQUESTED = "REVISION_REQUESTED"
    FAILED = "FAILED"


class ReviewActionEnum(str, Enum):
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    REQUEST_EVIDENCE_REFRESH = "REQUEST_EVIDENCE_REFRESH"
    REQUEST_REVISION = "REQUEST_REVISION"
    MARK_REVIEWED = "MARK_REVIEWED"
    ADD_NOTE = "ADD_NOTE"


class AbstentionReasonCode(str, Enum):
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    UNSUPPORTED_MATERIAL_CLAIMS = "UNSUPPORTED_MATERIAL_CLAIMS"
    CRITICAL_SOURCE_UNAVAILABLE = "CRITICAL_SOURCE_UNAVAILABLE"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    INSUFFICIENT_PATIENT_INFO = "INSUFFICIENT_PATIENT_INFO"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ITERATION_BUDGET_EXHAUSTED = "ITERATION_BUDGET_EXHAUSTED"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class AbstentionRequirement(str, Enum):
    CLARIFICATION = "CLARIFICATION"
    RETRIEVAL = "RETRIEVAL"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class CandidateClaim(BaseModel):
    """An individual atomic claim extracted from the candidate report."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_version_id: str = Field(...)
    claim_text: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    is_material: bool = Field(default=True, description="Material to clinical decision support")
    is_safety_critical: bool = Field(default=False, description="Flagged for safety-sensitive content")
    cited_evidence_ids: List[str] = Field(default_factory=list)
    source_spans: List[SourceSpan] = Field(default_factory=list)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class ClaimVerification(BaseModel):
    """Verification record produced by Claim Verification Agent (EXT-A)."""
    claim_id: str = Field(...)
    claim_text: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1, description="Finding, Consideration, or Evidence fact")
    status: ClaimVerificationStatusEnum = Field(...)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    justification: str = Field(..., min_length=1)
    evidence_spans: List[SourceSpan] = Field(default_factory=list)
    method: VerificationMethodEnum = Field(...)
    verifier_version: str = Field(default="1.0")
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    limitations: Optional[str] = Field(None, description="e.g. CITATION_NOT_FOUND or NUMERIC_MISMATCH")
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class CandidateClinicalReport(BaseModel):
    """Immutable versioned clinical report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_number: int = Field(..., ge=1)
    case_id: str = Field(...)
    summary: str = Field(..., min_length=1)
    claims: List[CandidateClaim] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    status: ReportStatusEnum = Field(default=ReportStatusEnum.DRAFT)
    urgent_banner: Optional[str] = Field(None, description="Banner text for possible urgent presentation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class AbstentionDecision(BaseModel):
    """Explicit abstention decision record (EXT-B)."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    abstained: bool = Field(...)
    reason_codes: List[AbstentionReasonCode] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    affected_claim_ids: List[str] = Field(default_factory=list)
    affected_sections: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    requires: List[AbstentionRequirement] = Field(default_factory=list)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason_codes")
    @classmethod
    def validate_abstained_reasons(cls, v: List[AbstentionReasonCode], info) -> List[AbstentionReasonCode]:
        abstained = info.data.get("abstained")
        if abstained and len(v) == 0:
            raise ValueError("When abstained is True, at least one reason_code must be provided.")
        return v


class HumanReview(BaseModel):
    """Clinician review audit and note record (EXT-D)."""
    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str = Field(..., min_length=1)
    reviewer_role: str = Field(..., description="clinician_reviewer | admin")
    action: ReviewActionEnum = Field(...)
    report_version_id: str = Field(...)
    outcome: str = Field(..., min_length=1)
    note: Optional[str] = Field(None, description="Clinician note (labeled HUMAN)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")
