"""Unit and Migration Schema Verification Tests.

Specification Section 6 (Relational DB Schema, 20 tables).
Tests that:
- All 20 tables are defined in metadata with required columns and constraints.
- Alembic migration scripts apply cleanly and idempotently.
"""

from alembic.config import Config
from alembic import command
import pytest
from sqlalchemy import create_engine, inspect
from backend.app.persistence.database import Base
import backend.app.persistence.models  # Register all models

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


def test_metadata_contains_all_20_tables():
    """Verify that SQLAlchemy metadata declares exactly the 20 tables from SPEC Section 6."""
    declared_tables = set(Base.metadata.tables.keys())
    assert declared_tables == EXPECTED_TABLES, (
        f"Missing tables: {EXPECTED_TABLES - declared_tables}; Extra tables: {declared_tables - EXPECTED_TABLES}"
    )


def test_unique_constraints_defined():
    """Verify unique constraints specified in SPEC Section 6."""
    # Unique (source, source_id) on evidence_documents
    doc_table = Base.metadata.tables["evidence_documents"]
    doc_constraints = [c.name for c in doc_table.constraints if hasattr(c, "columns")]
    assert "uq_evidence_documents_source_source_id" in doc_constraints

    # Unique evidence_id on evidence_passages
    passage_table = Base.metadata.tables["evidence_passages"]
    evidence_id_col = passage_table.columns["evidence_id"]
    assert evidence_id_col.unique is True


def test_alembic_upgrade_and_idempotence(tmp_path):
    """Verify alembic upgrade head applies on an empty database and repeat run is no-op."""
    db_file = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_file}"

    alembic_cfg = Config("backend/alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # First upgrade: applies migration 001_initial_schema
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    created_tables = set(inspector.get_table_names())
    assert EXPECTED_TABLES.issubset(created_tables)

    # Second upgrade: must be a clean no-op
    command.upgrade(alembic_cfg, "head")

    inspector_repeat = inspect(engine)
    assert set(inspector_repeat.get_table_names()) == created_tables
