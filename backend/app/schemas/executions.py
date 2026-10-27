"""Execution, Optimization, Audit, and Workflow State Schemas.

Specification (SPEC 2, 3, 5.3, 6, Appendix A5).
Models: AgentExecutionRecord, OptimizationIteration, AuditEvent, WorkflowResult, WorkflowState.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field
from backend.app.schemas.clinical_case import (
    CandidateHypothesis,
    ClarificationRequest,
    ClinicalCase,
)
from backend.app.schemas.evidence import ExternalEvidenceSet, RetrievedEvidenceSet
from backend.app.schemas.reports import (
    AbstentionDecision,
    CandidateClinicalReport,
    ClaimVerification,
    ReportStatusEnum,
)


class AgentStatusEnum(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


class AgentExecutionRecord(BaseModel):
    """Execution lifecycle record for an individual agent run."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = Field(..., min_length=1)
    status: AgentStatusEnum = Field(default=AgentStatusEnum.RUNNING)
    error_code: Optional[str] = Field(None, description="Standard error code e.g. LLM_UNAVAILABLE, INVALID_LLM_OUTPUT")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None)
    duration_ms: Optional[int] = Field(None, ge=0)
    input_summary: str = Field(..., description="Summary of inputs (no patient PII)")
    output_summary: Optional[str] = Field(None, description="Summary of outputs")
    error_message: Optional[str] = Field(None)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class OptimizationIteration(BaseModel):
    """Feedback loop iteration record (AGT-5)."""
    iteration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = Field(...)
    iteration_number: int = Field(..., ge=1, le=3, description="Capped at MAX_OPTIMIZER_ITERATIONS=3")
    node_selected: str = Field(..., min_length=1)
    confidence_before: float = Field(..., ge=0.0, le=1.0)
    confidence_after: Optional[float] = Field(None, ge=0.0, le=1.0)
    action_taken: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class AuditEvent(BaseModel):
    """Append-only audit trail event (EXT-C). Never contains patient free text or secrets."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., min_length=1)
    actor_id: str = Field(..., min_length=1)
    actor_role: str = Field(..., min_length=1)
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class WorkflowResult(BaseModel):
    """End-to-end workflow execution outcome."""
    execution_id: str = Field(...)
    case_id: str = Field(...)
    final_state: ReportStatusEnum = Field(...)
    report: Optional[CandidateClinicalReport] = Field(None)
    abstention: Optional[AbstentionDecision] = Field(None)
    iterations_count: int = Field(default=0, ge=0, le=3)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class WorkflowState(BaseModel):
    """Typed state for the multi-agent LangGraph workflow."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = Field(...)
    case: Optional[ClinicalCase] = Field(None)
    clarification_needed: bool = Field(default=False)
    blocking_clarifications: List[ClarificationRequest] = Field(default_factory=list)
    non_blocking_clarifications: List[ClarificationRequest] = Field(default_factory=list)
    candidate_hypotheses: List[CandidateHypothesis] = Field(default_factory=list)
    retrieved_evidence: Optional[RetrievedEvidenceSet] = Field(None)
    external_evidence: Optional[ExternalEvidenceSet] = Field(None)
    report: Optional[CandidateClinicalReport] = Field(None)
    verifications: List[ClaimVerification] = Field(default_factory=list)
    abstention: Optional[AbstentionDecision] = Field(None)
    iterations: List[OptimizationIteration] = Field(default_factory=list)
    audit_trail: List[AuditEvent] = Field(default_factory=list)
    current_node: str = Field(default="validate_case")
    is_benchmark: bool = Field(default=False)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")
