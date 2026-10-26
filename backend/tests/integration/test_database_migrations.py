"""Integration Tests for PostgreSQL Relational Migrations.

Specification Section 6 (PostgreSQL 16, Alembic, 20 Tables).
Runs against the real PostgreSQL container (must FAIL, not skip, if unreachable).
"""

import pytest
from sqlalchemy import create_engine, inspect
from backend.app.core.config import settings

EXPECTED_TABLES = {
    "roles",
    "users",
    "clinical_cases",
    "case_findings",
    "workflow_executions",
    "agent_executions",
    "evidence_documents",
    "evidence_passages",
    "retrieval_runs",
    "report_versions",
    "candidate_claims",
    "claim_verifications",
    "provenance_edges",
    "abstention_decisions",
    "optimization_iterations",
    "human_reviews",
    "audit_events",
    "evaluation_runs",
    "evaluation_metrics",
    "evaluation_labels",
}


@pytest.mark.integration
def test_postgres_migrations_all_20_tables_and_constraints():
    """Verify migrations applied to real Postgres container: all 20 tables exist with required constraints."""
    try:
        engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception as e:
        pytest.fail(f"PostgreSQL container unreachable or failed connection: {e}")

    # Confirm all 20 tables exist
    missing = EXPECTED_TABLES - existing_tables
    assert not missing, f"Missing tables in PostgreSQL container: {missing}"

    # Verify unique constraint on evidence_documents(source, source_id)
    doc_constraints = inspector.get_unique_constraints("evidence_documents")
    doc_unique_names = {c["name"] for c in doc_constraints}
    assert "uq_evidence_documents_source_source_id" in doc_unique_names, (
        f"Missing uq_evidence_documents_source_source_id in {doc_unique_names}"
    )

    # Verify unique constraint / index on evidence_passages(evidence_id)
    passage_indexes = inspector.get_indexes("evidence_passages")
    unique_cols = [
        idx["column_names"] for idx in passage_indexes if idx.get("unique")
    ]
    # Check if evidence_id is in any unique index or constraint
    passage_constraints = inspector.get_unique_constraints("evidence_passages")
    for c in passage_constraints:
        unique_cols.append(c["column_names"])

    evidence_id_unique = any(["evidence_id"] == cols for cols in unique_cols)
    assert evidence_id_unique, f"evidence_id is not uniquely constrained on evidence_passages. Unique indexes/constraints: {unique_cols}"
