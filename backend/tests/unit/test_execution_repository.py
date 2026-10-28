"""Unit Tests for Execution Persistence Repository and Interrupted Reconciler.

Specification Section 2, 3, 6 (Req ID: DB-3).
"""

from datetime import datetime, timezone
import uuid
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.persistence.database import Base
from backend.app.persistence.execution_repository import (
    ExecutionRepository,
    reconcile_interrupted_executions,
)
from backend.app.persistence.models import (
    AgentExecution as AgentExecutionModel,
    ReportVersion as ReportVersionModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.schemas.evidence import (
    EvidenceChunk as EvidenceChunkSchema,
    EvidenceDocument as EvidenceDocumentSchema,
    EvidenceSourceEnum,
)
from backend.app.schemas.reports import (
    CandidateClaim as CandidateClaimSchema,
)


@pytest.fixture
def sqlite_session():
    """In-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_reconcile_interrupted_executions(sqlite_session):
    """Verify that any workflow execution left in RUNNING status is marked FAILED(INTERRUPTED)."""
    case_id = str(uuid.uuid4())
    repo = ExecutionRepository(session=sqlite_session)
    repo.create_case(
        case_id=case_id,
        raw_text_hash="abc123hash",
        chief_complaint="Chest tightness",
        status="ANALYZING",
    )

    # 1. Create a stale RUNNING execution and an already COMPLETED execution
    running_id = str(uuid.uuid4())
    completed_id = str(uuid.uuid4())
    repo.create_workflow_execution(running_id, case_id=case_id, status="RUNNING")
    repo.create_workflow_execution(completed_id, case_id=case_id, status="COMPLETED")

    # Also add an agent execution under the running workflow
    repo.record_agent_execution_start(
        execution_id=running_id,
        agent_name="clinical_text_clarifier",
        input_summary="Input case",
    )

    # 2. Run reconciler
    reconciled_count = reconcile_interrupted_executions(session=sqlite_session)
    assert reconciled_count == 1

    # 3. Assert status updates
    running_row = sqlite_session.get(WorkflowExecutionModel, running_id)
    assert running_row.status == "FAILED"
    assert running_row.failure_reason == "INTERRUPTED"
    assert running_row.completed_at is not None

    completed_row = sqlite_session.get(WorkflowExecutionModel, completed_id)
    assert completed_row.status == "COMPLETED"
    assert completed_row.failure_reason is None

    # Agent execution under interrupted workflow is also marked FAILED
    agent_rows = sqlite_session.scalars(
        select(AgentExecutionModel).where(AgentExecutionModel.execution_id == running_id)
    ).all()
    assert len(agent_rows) == 1
    assert agent_rows[0].status == "FAILED"
    assert agent_rows[0].error_code == "INTERRUPTED"


def test_agent_execution_start_and_finish_lifecycle(sqlite_session):
    """Verify before-and-after agent execution records with input/output hashes and status."""
    case_id = str(uuid.uuid4())
    exec_id = str(uuid.uuid4())
    repo = ExecutionRepository(session=sqlite_session)
    repo.create_case(case_id=case_id, raw_text_hash="hash1", chief_complaint="Dyspnea")
    repo.create_workflow_execution(exec_id, case_id=case_id)

    # Before node: status RUNNING
    start_rec = repo.record_agent_execution_start(
        execution_id=exec_id,
        agent_name="symptom_rag_analyzer",
        input_summary="Case with 2 symptoms",
        input_hash="input_sha256_hash",
    )
    assert start_rec.status == "RUNNING"
    assert start_rec.input_hash == "input_sha256_hash"
    assert start_rec.completed_at is None

    # After node: status COMPLETED
    finish_rec = repo.record_agent_execution_finish(
        record_id=start_rec.id,
        status="COMPLETED",
        output_summary="Retrieved 5 passages",
        output_hash="output_sha256_hash",
    )
    assert finish_rec.status == "COMPLETED"
    assert finish_rec.output_hash == "output_sha256_hash"
    assert finish_rec.completed_at is not None
    assert finish_rec.duration_ms is not None
    assert finish_rec.duration_ms >= 0


def test_report_version_immutability(sqlite_session):
    """Verify that every report version revision creates an immutable new row."""
    case_id = str(uuid.uuid4())
    exec_id = str(uuid.uuid4())
    repo = ExecutionRepository(session=sqlite_session)
    repo.create_case(case_id=case_id, raw_text_hash="hash2", chief_complaint="Cough")
    repo.create_workflow_execution(exec_id, case_id=case_id)

    rep_v1_id = str(uuid.uuid4())
    claim1 = CandidateClaimSchema(
        claim_id=str(uuid.uuid4()),
        report_version_id=rep_v1_id,
        claim_text="Pneumonia is suspected.",
        section="differential",
        is_material=True,
        is_safety_critical=False,
    )
    v1 = repo.save_report_version(
        case_id=case_id,
        execution_id=exec_id,
        version_number=1,
        summary="Initial draft",
        confidence_score=0.55,
        status="DRAFT",
        claims=[claim1],
    )

    rep_v2_id = str(uuid.uuid4())
    claim2 = CandidateClaimSchema(
        claim_id=str(uuid.uuid4()),
        report_version_id=rep_v2_id,
        claim_text="Viral bronchitis considered.",
        section="differential",
        is_material=True,
        is_safety_critical=False,
    )
    v2 = repo.save_report_version(
        case_id=case_id,
        execution_id=exec_id,
        version_number=2,
        summary="Revised report",
        confidence_score=0.72,
        status="READY_FOR_REVIEW",
        claims=[claim1, claim2],
    )

    all_versions = sqlite_session.scalars(
        select(ReportVersionModel).where(ReportVersionModel.case_id == case_id)
    ).all()
    assert len(all_versions) == 2
    assert {v.version_number for v in all_versions} == {1, 2}
    assert {v.summary for v in all_versions} == {"Initial draft", "Revised report"}
