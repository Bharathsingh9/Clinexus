"""Relational PostgreSQL ORM Models.

Specification Section 6 (PostgreSQL 16, SQLAlchemy 2.0).
Tables:
1. users
2. roles
3. clinical_cases
4. case_findings
5. workflow_executions
6. agent_executions
7. evidence_documents
8. evidence_passages
9. retrieval_runs
10. report_versions
11. candidate_claims
12. claim_verifications
13. provenance_edges
14. abstention_decisions
15. optimization_iterations
16. human_reviews
17. audit_events
18. evaluation_runs
19. evaluation_metrics
20. evaluation_labels
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from backend.app.persistence.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Role(Base):
    __tablename__ = "roles"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="viewer")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ClinicalCase(Base):
    __tablename__ = "clinical_cases"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    raw_text_hash = Column(String(64), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    age_missing_reason = Column(String(255), nullable=True)
    sex = Column(String(20), nullable=True)
    chief_complaint = Column(Text, nullable=False)
    symptom_duration = Column(String(100), nullable=True)
    duration_missing_reason = Column(String(255), nullable=True)
    status = Column(String(50), nullable=False, default="DRAFT")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    findings = relationship("CaseFinding", back_populates="case", cascade="all, delete-orphan")
    workflow_executions = relationship("WorkflowExecution", back_populates="case")


class CaseFinding(Base):
    __tablename__ = "case_findings"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    term = Column(Text, nullable=False)
    normalized_term = Column(String(255), nullable=True)
    vocabulary = Column(String(50), nullable=False, default="LOCAL")
    polarity = Column(String(20), nullable=False)  # PRESENT, ABSENT
    onset = Column(String(100), nullable=True)
    duration = Column(String(100), nullable=True)
    severity = Column(String(100), nullable=True)
    progression = Column(String(100), nullable=True)
    source_span_start = Column(Integer, nullable=False)
    source_span_end = Column(Integer, nullable=False)
    origin = Column(String(20), nullable=False)  # STATED, INFERRED
    uncertainty_note = Column(Text, nullable=True)

    case = relationship("ClinicalCase", back_populates="findings")


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="RUNNING")
    is_benchmark = Column(Boolean, nullable=False, default=False)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    failure_reason = Column(String(255), nullable=True)

    case = relationship("ClinicalCase", back_populates="workflow_executions")
    agent_executions = relationship("AgentExecution", back_populates="workflow_execution")


class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="RUNNING")
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    input_summary = Column(Text, nullable=False)
    output_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)
    input_hash = Column(String(64), nullable=True)
    output_hash = Column(String(64), nullable=True)

    workflow_execution = relationship("WorkflowExecution", back_populates="agent_executions")


class EvidenceDocument(Base):
    __tablename__ = "evidence_documents"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_evidence_documents_source_source_id"),
    )

    id = Column(String(36), primary_key=True, default=gen_uuid)
    source = Column(String(50), nullable=False, index=True)  # PUBMED, BIOASQ
    source_id = Column(String(100), nullable=False, index=True)
    title = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)
    journal = Column(String(255), nullable=True)
    publication_date = Column(String(50), nullable=True)
    pmid = Column(String(50), nullable=True, index=True)
    doi = Column(String(255), nullable=True)
    url = Column(Text, nullable=True)
    abstract = Column(Text, nullable=True)
    abstract_missing = Column(Boolean, nullable=False, default=False)
    evidence_level = Column(String(100), nullable=True, default="UNKNOWN")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    passages = relationship("EvidencePassage", back_populates="document", cascade="all, delete-orphan")


class EvidencePassage(Base):
    __tablename__ = "evidence_passages"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    document_id = Column(String(36), ForeignKey("evidence_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_id = Column(String(100), unique=True, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=False, index=True)
    short_passage = Column(Boolean, nullable=False, default=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    document = relationship("EvidenceDocument", back_populates="passages")


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    query_id = Column(String(36), nullable=False)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)
    top_k = Column(Integer, nullable=False)
    min_score = Column(Float, nullable=False)
    corpus_version = Column(String(128), nullable=False)
    embedding_model = Column(String(128), nullable=False)
    insufficient = Column(Boolean, nullable=False, default=False)
    reason_code = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class ReportVersion(Base):
    __tablename__ = "report_versions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False, default=1)
    summary = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    status = Column(String(50), nullable=False, default="DRAFT")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    claims = relationship("CandidateClaim", back_populates="report_version", cascade="all, delete-orphan")


class CandidateClaim(Base):
    __tablename__ = "candidate_claims"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    report_version_id = Column(String(36), ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    claim_text = Column(Text, nullable=False)
    section = Column(String(100), nullable=False)
    is_material = Column(Boolean, nullable=False, default=True)
    is_safety_critical = Column(Boolean, nullable=False, default=False)

    report_version = relationship("ReportVersion", back_populates="claims")
    verifications = relationship("ClaimVerification", back_populates="claim", cascade="all, delete-orphan")


class ClaimVerification(Base):
    __tablename__ = "claim_verifications"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    claim_id = Column(String(36), ForeignKey("candidate_claims.id", ondelete="CASCADE"), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, etc.
    justification = Column(Text, nullable=False)
    method = Column(String(50), nullable=False)  # DETERMINISTIC, MODEL, HYBRID
    verifier_version = Column(String(50), nullable=False, default="1.0")
    verified_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    limitations = Column(Text, nullable=True)

    claim = relationship("CandidateClaim", back_populates="verifications")


class ProvenanceEdge(Base):
    __tablename__ = "provenance_edges"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    report_version_id = Column(String(36), ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="SET NULL"), nullable=True)
    finding_id = Column(String(36), ForeignKey("case_findings.id", ondelete="SET NULL"), nullable=True)
    query_id = Column(String(36), nullable=True)
    evidence_chunk_id = Column(String(36), ForeignKey("evidence_passages.id", ondelete="SET NULL"), nullable=True)
    document_id = Column(String(36), ForeignKey("evidence_documents.id", ondelete="SET NULL"), nullable=True)
    claim_id = Column(String(36), ForeignKey("candidate_claims.id", ondelete="SET NULL"), nullable=True)
    verification_id = Column(String(36), ForeignKey("claim_verifications.id", ondelete="SET NULL"), nullable=True)
    report_section_id = Column(String(100), nullable=True)
    iteration_id = Column(String(36), nullable=True)
    abstention_id = Column(String(36), nullable=True)
    review_event_id = Column(String(36), nullable=True)
    relation = Column(String(50), nullable=False)  # SUPPORTS, PARTIALLY_SUPPORTS, CONTRADICTS, etc.
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class AbstentionDecision(Base):
    __tablename__ = "abstention_decisions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    report_version_id = Column(String(36), ForeignKey("report_versions.id", ondelete="SET NULL"), nullable=True)
    abstained = Column(Boolean, nullable=False)
    reason_codes = Column(Text, nullable=False)  # JSON-encoded array of codes
    explanation = Column(Text, nullable=False)
    affected_claim_ids = Column(Text, nullable=False, default="[]")
    affected_sections = Column(Text, nullable=False, default="[]")
    missing_evidence = Column(Text, nullable=False, default="[]")
    missing_information = Column(Text, nullable=False, default="[]")
    suggested_next_steps = Column(Text, nullable=False, default="[]")
    requires = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class OptimizationIteration(Base):
    __tablename__ = "optimization_iterations"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    execution_id = Column(String(36), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False, index=True)
    iteration_number = Column(Integer, nullable=False)
    node_selected = Column(String(100), nullable=False)
    confidence_before = Column(Float, nullable=False)
    confidence_after = Column(Float, nullable=True)
    action_taken = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    case_id = Column(String(36), ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False, index=True)
    report_version_id = Column(String(36), ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_role = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)  # REQUEST_CLARIFICATION, MARK_REVIEWED, etc.
    outcome = Column(Text, nullable=False)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    event_type = Column(String(100), nullable=False, index=True)
    actor_id = Column(String(100), nullable=False)
    actor_role = Column(String(50), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(100), nullable=False)
    payload = Column(Text, nullable=False, default="{}")  # Scrubbed JSON string (never patient text/secrets)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    dataset_name = Column(String(100), nullable=False)
    split_name = Column(String(50), nullable=False)
    sample_size = Column(Integer, nullable=False)
    corpus_version = Column(String(128), nullable=False)
    embedding_model = Column(String(128), nullable=False)
    run_metadata = Column(Text, nullable=False, default="{}")
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), nullable=False, default="RUNNING")

    metrics = relationship("EvaluationMetric", back_populates="evaluation_run", cascade="all, delete-orphan")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    evaluation_run_id = Column(String(36), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    question_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    is_unconfirmed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    evaluation_run = relationship("EvaluationRun", back_populates="metrics")


class EvaluationLabel(Base):
    __tablename__ = "evaluation_labels"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    sample_id = Column(String(100), nullable=False, index=True)
    corpus_id = Column(String(100), nullable=False, index=True)
    relevance_label = Column(Integer, nullable=False)  # 0 or 1
    is_derived = Column(Boolean, nullable=False, default=True)
    approved_by_human = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
