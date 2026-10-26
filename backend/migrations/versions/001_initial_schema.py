"""001 Initial Schema for All 20 Tables

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-26 13:30:00.000000

Specification Section 6 (PostgreSQL 16, SQLAlchemy 2.0).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. roles
    op.create_table(
        "roles",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=50), nullable=False, unique=True),
        sa.Column("description", sa.String(length=255), nullable=False),
    )
    op.create_index("ix_roles_name", "roles", ["name"])

    # 2. users
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # 3. clinical_cases
    op.create_table(
        "clinical_cases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("raw_text_hash", sa.String(length=64), nullable=False),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("age_missing_reason", sa.String(length=255), nullable=True),
        sa.Column("sex", sa.String(length=20), nullable=True),
        sa.Column("chief_complaint", sa.Text(), nullable=False),
        sa.Column("symptom_duration", sa.String(length=100), nullable=True),
        sa.Column("duration_missing_reason", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="DRAFT"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_clinical_cases_raw_text_hash", "clinical_cases", ["raw_text_hash"])

    # 4. case_findings
    op.create_table(
        "case_findings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("normalized_term", sa.String(length=255), nullable=True),
        sa.Column("vocabulary", sa.String(length=50), nullable=False, server_default="LOCAL"),
        sa.Column("polarity", sa.String(length=20), nullable=False),
        sa.Column("onset", sa.String(length=100), nullable=True),
        sa.Column("duration", sa.String(length=100), nullable=True),
        sa.Column("severity", sa.String(length=100), nullable=True),
        sa.Column("progression", sa.String(length=100), nullable=True),
        sa.Column("source_span_start", sa.Integer(), nullable=False),
        sa.Column("source_span_end", sa.Integer(), nullable=False),
        sa.Column("origin", sa.String(length=20), nullable=False),
        sa.Column("uncertainty_note", sa.Text(), nullable=True),
    )
    op.create_index("ix_case_findings_case_id", "case_findings", ["case_id"])

    # 5. workflow_executions
    op.create_table(
        "workflow_executions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="RUNNING"),
        sa.Column("is_benchmark", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
    )
    op.create_index("ix_workflow_executions_case_id", "workflow_executions", ["case_id"])

    # 6. agent_executions
    op.create_table(
        "agent_executions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="RUNNING"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("input_summary", sa.Text(), nullable=False),
        sa.Column("output_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("ix_agent_executions_execution_id", "agent_executions", ["execution_id"])

    # 7. evidence_documents
    op.create_table(
        "evidence_documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.String(length=100), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("journal", sa.String(length=255), nullable=True),
        sa.Column("publication_date", sa.String(length=50), nullable=True),
        sa.Column("pmid", sa.String(length=50), nullable=True),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("abstract_missing", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("evidence_level", sa.String(length=100), nullable=True, server_default="UNKNOWN"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source", "source_id", name="uq_evidence_documents_source_source_id"),
    )
    op.create_index("ix_evidence_documents_source", "evidence_documents", ["source"])
    op.create_index("ix_evidence_documents_source_id", "evidence_documents", ["source_id"])
    op.create_index("ix_evidence_documents_pmid", "evidence_documents", ["pmid"])

    # 8. evidence_passages
    op.create_table(
        "evidence_passages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("evidence_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("evidence_id", sa.String(length=100), nullable=False, unique=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("short_passage", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evidence_passages_document_id", "evidence_passages", ["document_id"])
    op.create_index("ix_evidence_passages_evidence_id", "evidence_passages", ["evidence_id"])
    op.create_index("ix_evidence_passages_text_hash", "evidence_passages", ["text_hash"])

    # 9. retrieval_runs
    op.create_table(
        "retrieval_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query_id", sa.String(length=36), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_type", sa.String(length=50), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("min_score", sa.Float(), nullable=False),
        sa.Column("corpus_version", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("insufficient", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reason_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retrieval_runs_execution_id", "retrieval_runs", ["execution_id"])

    # 10. report_versions
    op.create_table(
        "report_versions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="DRAFT"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_report_versions_case_id", "report_versions", ["case_id"])
    op.create_index("ix_report_versions_execution_id", "report_versions", ["execution_id"])

    # 11. candidate_claims
    op.create_table(
        "candidate_claims",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("report_version_id", sa.String(length=36), sa.ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(length=100), nullable=False),
        sa.Column("is_material", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_safety_critical", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_candidate_claims_report_version_id", "candidate_claims", ["report_version_id"])

    # 12. claim_verifications
    op.create_table(
        "claim_verifications",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("claim_id", sa.String(length=36), sa.ForeignKey("candidate_claims.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("method", sa.String(length=50), nullable=False),
        sa.Column("verifier_version", sa.String(length=50), nullable=False, server_default="1.0"),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("limitations", sa.Text(), nullable=True),
    )
    op.create_index("ix_claim_verifications_claim_id", "claim_verifications", ["claim_id"])

    # 13. provenance_edges
    op.create_table(
        "provenance_edges",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("report_version_id", sa.String(length=36), sa.ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="SET NULL"), nullable=True),
        sa.Column("finding_id", sa.String(length=36), sa.ForeignKey("case_findings.id", ondelete="SET NULL"), nullable=True),
        sa.Column("query_id", sa.String(length=36), nullable=True),
        sa.Column("evidence_chunk_id", sa.String(length=36), sa.ForeignKey("evidence_passages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("evidence_documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("claim_id", sa.String(length=36), sa.ForeignKey("candidate_claims.id", ondelete="SET NULL"), nullable=True),
        sa.Column("verification_id", sa.String(length=36), sa.ForeignKey("claim_verifications.id", ondelete="SET NULL"), nullable=True),
        sa.Column("report_section_id", sa.String(length=100), nullable=True),
        sa.Column("iteration_id", sa.String(length=36), nullable=True),
        sa.Column("abstention_id", sa.String(length=36), nullable=True),
        sa.Column("review_event_id", sa.String(length=36), nullable=True),
        sa.Column("relation", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_provenance_edges_report_version_id", "provenance_edges", ["report_version_id"])

    # 14. abstention_decisions
    op.create_table(
        "abstention_decisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_version_id", sa.String(length=36), sa.ForeignKey("report_versions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("reason_codes", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("affected_claim_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("affected_sections", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("missing_evidence", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("missing_information", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("suggested_next_steps", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("requires", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_abstention_decisions_execution_id", "abstention_decisions", ["execution_id"])
    op.create_index("ix_abstention_decisions_case_id", "abstention_decisions", ["case_id"])

    # 15. optimization_iterations
    op.create_table(
        "optimization_iterations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("execution_id", sa.String(length=36), sa.ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("iteration_number", sa.Integer(), nullable=False),
        sa.Column("node_selected", sa.String(length=100), nullable=False),
        sa.Column("confidence_before", sa.Float(), nullable=False),
        sa.Column("confidence_after", sa.Float(), nullable=True),
        sa.Column("action_taken", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_optimization_iterations_execution_id", "optimization_iterations", ["execution_id"])

    # 16. human_reviews
    op.create_table(
        "human_reviews",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("case_id", sa.String(length=36), sa.ForeignKey("clinical_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_version_id", sa.String(length=36), sa.ForeignKey("report_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer_role", sa.String(length=50), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_human_reviews_case_id", "human_reviews", ["case_id"])
    op.create_index("ix_human_reviews_report_version_id", "human_reviews", ["report_version_id"])
    op.create_index("ix_human_reviews_reviewer_id", "human_reviews", ["reviewer_id"])

    # 17. audit_events
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("actor_id", sa.String(length=100), nullable=False),
        sa.Column("actor_role", sa.String(length=50), nullable=False),
        sa.Column("resource_type", sa.String(length=100), nullable=False),
        sa.Column("resource_id", sa.String(length=100), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_timestamp", "audit_events", ["timestamp"])

    # 18. evaluation_runs
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_name", sa.String(length=100), nullable=False),
        sa.Column("split_name", sa.String(length=50), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("corpus_version", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("run_metadata", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="RUNNING"),
    )

    # 19. evaluation_metrics
    op.create_table(
        "evaluation_metrics",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("evaluation_run_id", sa.String(length=36), sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_type", sa.String(length=50), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("is_unconfirmed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluation_metrics_evaluation_run_id", "evaluation_metrics", ["evaluation_run_id"])

    # 20. evaluation_labels
    op.create_table(
        "evaluation_labels",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sample_id", sa.String(length=100), nullable=False),
        sa.Column("corpus_id", sa.String(length=100), nullable=False),
        sa.Column("relevance_label", sa.Integer(), nullable=False),
        sa.Column("is_derived", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("approved_by_human", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluation_labels_sample_id", "evaluation_labels", ["sample_id"])
    op.create_index("ix_evaluation_labels_corpus_id", "evaluation_labels", ["corpus_id"])


def downgrade() -> None:
    op.drop_table("evaluation_labels")
    op.drop_table("evaluation_metrics")
    op.drop_table("evaluation_runs")
    op.drop_table("audit_events")
    op.drop_table("human_reviews")
    op.drop_table("optimization_iterations")
    op.drop_table("abstention_decisions")
    op.drop_table("provenance_edges")
    op.drop_table("claim_verifications")
    op.drop_table("candidate_claims")
    op.drop_table("report_versions")
    op.drop_table("retrieval_runs")
    op.drop_table("evidence_passages")
    op.drop_table("evidence_documents")
    op.drop_table("agent_executions")
    op.drop_table("workflow_executions")
    op.drop_table("case_findings")
    op.drop_table("clinical_cases")
    op.drop_table("users")
    op.drop_table("roles")
