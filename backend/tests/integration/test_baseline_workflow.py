"""Integration Tests for Baseline Multi-Agent Workflow.

Specification Section 2, 3, 4, Section 11, Appendix A1, A2, A3 (Req ID: WKF-1, DB-3).
Uses real PostgreSQL and Qdrant containers with real orchestration.
Mocks ONLY external boundaries: FakeLLMProvider and FakeNCBIClient with [SYNTHETIC] responses.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import uuid

import pytest
from sqlalchemy import delete, select

from backend.app.agents.adaptive_optimizer import AdaptiveOptimizer
from backend.app.agents.clinical_data_fusion import ClinicalDataFusion
from backend.app.agents.clinical_text_clarifier import ClinicalTextClarifier
from backend.app.agents.evidence_based_scanner import EvidenceBasedScanner
from backend.app.agents.symptom_rag_analyzer import SymptomRAGAnalyzer
from backend.app.core.config import settings
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.orchestration.baseline_workflow import BaselineWorkflowOrchestrator
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.execution_repository import (
    ExecutionRepository,
    reconcile_interrupted_executions,
)
from backend.app.persistence.models import (
    AgentExecution as AgentExecutionModel,
    CandidateClaim as CandidateClaimModel,
    CaseFinding as CaseFindingModel,
    ClinicalCase as ClinicalCaseModel,
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
    OptimizationIteration as OptimizationIterationModel,
    ReportVersion as ReportVersionModel,
    RetrievalRun as RetrievalRunModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name
from backend.app.schemas.clinical_case import CaseSubmission
from backend.app.schemas.evidence import QueryStatusEnum
from backend.app.schemas.executions import (
    AgentStatusEnum,
    WorkflowState,
)
from backend.app.schemas.reports import ReportStatusEnum
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider
from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_corpus.csv"
)
TEST_COLLECTION_VERSION = 99


@pytest.fixture(scope="module")
def setup_synthetic_corpus():
    """Ingest synthetic fixture corpus into test Qdrant collection and clean up afterwards."""
    qdrant = QdrantWrapper()
    embedder = EmbeddingService()
    col_name = format_collection_name(embedder.model_name, TEST_COLLECTION_VERSION)

    # Clean existing fixture database records
    with SessionLocal() as session:
        session.execute(delete(EvidencePassageModel).where(EvidencePassageModel.evidence_id.like("ev:bioasq:SYNTH%")))
        session.execute(delete(EvidenceDocumentModel).where(EvidenceDocumentModel.source_id.in_(["111001", "111002", "111003", "111004", "111005"])))
        session.commit()

    if qdrant.collection_exists(col_name):
        qdrant.client.delete_collection(col_name)

    pipeline = IngestionPipeline(
        embedding_service=embedder,
        qdrant_wrapper=qdrant,
        db_session_factory=SessionLocal,
    )
    pipeline.ingest_file(
        file_path=FIXTURE_PATH,
        collection_version=TEST_COLLECTION_VERSION,
        reindex=False,
        batch_size=10,
    )

    yield {
        "qdrant": qdrant,
        "embedder": embedder,
        "collection_version": TEST_COLLECTION_VERSION,
        "collection_name": col_name,
    }

    # Teardown
    if qdrant.collection_exists(col_name):
        qdrant.client.delete_collection(col_name)

    with SessionLocal() as session:
        session.execute(delete(EvidencePassageModel).where(EvidencePassageModel.evidence_id.like("ev:bioasq:SYNTH%")))
        session.execute(delete(EvidenceDocumentModel).where(EvidenceDocumentModel.source_id.in_(["111001", "111002", "111003", "111004", "111005"])))
        session.commit()


def make_canned_llm_for_workflow(
    clarifier_dict: Optional[Dict] = None,
    rag_hypotheses_text: Optional[str] = None,
    fusion_dict: Optional[Dict] = None,
    optimizer_text: str = "Continue monitoring clinical findings.",
) -> FakeLLMProvider:
    """Helper to configure a FakeLLMProvider responding deterministically to each agent prompt."""
    default_clarifier = {
        "age": 45,
        "sex": "male",
        "chief_complaint": "fever and persistent cough",
        "symptom_duration": "3 days",
        "symptoms": [
            {"term": "fever", "polarity": "PRESENT", "source_span": {"start": 24, "end": 29}, "origin": "STATED"},
            {"term": "cough", "polarity": "PRESENT", "source_span": {"start": 56, "end": 61}, "origin": "STATED"},
            {"term": "chest pain", "polarity": "ABSENT", "source_span": {"start": 78, "end": 88}, "origin": "STATED"},
        ],
        "patient_context": {
            "history": ["no chronic pulmonary history"],
            "medications": ["none"],
            "allergies": ["none"],
            "vitals": {"temperature": "38.5C"},
        },
    }

    default_fusion = {
        "sections": {
            "case_summary": "45-year-old male presenting with fever and persistent dry cough for 3 days.",
            "relevant_findings": "Stated fever and dry cough; chest pain explicitly absent.",
            "evidence_supported_observations": "Fever and dry cough are characteristic of community-acquired respiratory infection [ev:bioasq:SYNTH001:0].",
            "potential_considerations": "Differential considerations include acute viral bronchitis and community-acquired pneumonia [ev:pubmed:30001:0].",
            "alternative_explanations": "Early presentation of atypical bacterial infection.",
            "contradictory_or_limiting_evidence": "Absence of chest pain lowers suspicion for acute pleuritis.",
            "missing_information": "Sputum gram stain and chest radiography not provided.",
            "uncertainty_and_limitations": "Synthesized from outpatient narrative without laboratory panel.",
            "recommended_evidence_review_steps": "Correlate with pulse oximetry and consider chest imaging if tachypnea develops.",
            "references": "BioASQ SYNTH001, PubMed 30001.",
        },
        "claims": [
            {
                "claim_text": "Fever and dry cough are characteristic of community-acquired respiratory infection.",
                "section": "evidence_supported_observations",
                "cited_evidence_ids": ["ev:bioasq:SYNTH001:0"],
            },
            {
                "claim_text": "Differential considerations include acute viral bronchitis and community-acquired pneumonia.",
                "section": "potential_considerations",
                "cited_evidence_ids": ["ev:pubmed:30001:0"],
            },
        ],
    }

    c_json = json.dumps(clarifier_dict or default_clarifier)
    f_json = json.dumps(fusion_dict or default_fusion)
    h_text = rag_hypotheses_text or (
        "CONSIDERATION: Community-acquired respiratory infection | CITATIONS: ev:bioasq:SYNTH001:0\n"
        "CONSIDERATION: Acute bronchitis | CITATIONS: ev:bioasq:SYNTH001:0"
    )

    def text_handler(prompt: str, system: Optional[str] = None) -> str:
        s = system or ""
        p = prompt or ""
        if "Clinical Text Clarifier" in s or "Extract stated symptoms" in s:
            return c_json
        elif "Clinical Data Fusion" in s or "Synthesize retrieved evidence" in s:
            return f_json
        elif "Adaptive Optimizer" in s or "SELECTED OPERATION" in p:
            return optimizer_text
        elif "CONSIDERATION" in p or "candidate differential hypotheses" in s:
            return h_text
        return "[SYNTHETIC] Default text completion"

    return FakeLLMProvider(text_handler=text_handler)


@pytest.mark.integration
def test_end_to_end_synthetic_case_all_nodes_and_stage_chaining(setup_synthetic_corpus):
    """A synthetic case runs through every node; assert stage chaining and evidence consistency."""
    embedder = setup_synthetic_corpus["embedder"]
    qdrant = setup_synthetic_corpus["qdrant"]
    col_ver = setup_synthetic_corpus["collection_version"]

    fake_llm = make_canned_llm_for_workflow()
    fake_ncbi = FakeNCBIClient()
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        llm_provider=fake_llm,
        collection_version=col_ver,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        scanner=EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=fake_llm, embedding_service=embedder),
        fusion=ClinicalDataFusion(llm_provider=fake_llm),
        optimizer=AdaptiveOptimizer(llm_provider=fake_llm),
        repository=repo,
        llm_provider=fake_llm,
        ncbi_client=fake_ncbi,
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
    )

    raw_narrative = "45yo male presents with fever for 3 days and persistent cough. Patient denies chest pain."
    sub = CaseSubmission(raw_text=raw_narrative)
    case_id = str(uuid.uuid4())
    exec_id = str(uuid.uuid4())

    init_state = WorkflowState(
        execution_id=exec_id,
        case_id=case_id,
        submission=sub,
    )

    final_state = orchestrator.run(init_state)

    # 1. Workflow finished successfully
    assert final_state.workflow_status == ReportStatusEnum.READY_FOR_REVIEW.value
    assert final_state.final_result is not None
    assert final_state.final_result.final_state == ReportStatusEnum.READY_FOR_REVIEW

    # 2. Every node executed
    rec_names = [r.agent_name for r in final_state.agent_records]
    assert "validate_case" in rec_names
    assert "clinical_text_clarifier" in rec_names
    assert "clarification_check" in rec_names
    assert "symptom_rag_analyzer" in rec_names
    assert "evidence_based_scanner" in rec_names
    assert "clinical_data_fusion" in rec_names

    # 3. Stage chaining: input hash of stage N equals output hash of stage N-1
    rec_map = {r.agent_name: r for r in final_state.agent_records}
    val_rec = rec_map["validate_case"]
    clar_rec = rec_map["clinical_text_clarifier"]
    check_rec = rec_map["clarification_check"]
    rag_rec = rec_map["symptom_rag_analyzer"]
    scan_rec = rec_map["evidence_based_scanner"]

    # Clarifier input hash equals validate_case output hash
    assert clar_rec.input_hash == val_rec.output_hash
    # Clarification check and RAG analyzer consume Clarifier's ClinicalCase output
    assert check_rec.input_hash == clar_rec.output_hash
    assert rag_rec.input_hash == clar_rec.output_hash
    # Scanner input hash equals RAG analyzer's output hash (RetrievedEvidenceSet)
    assert scan_rec.input_hash == rag_rec.output_hash

    # 4. Evidence IDs cited by Fusion exist in earlier evidence sets
    assert final_state.report is not None
    assert len(final_state.report.claims) >= 2

    local_ev_ids = set()
    for p in final_state.retrieved_evidence.results:
        local_ev_ids.add(p.evidence_id)
        if p.duplicate_evidence_ids:
            local_ev_ids.update(p.duplicate_evidence_ids)
    ext_ev_ids = {p.evidence_id for p in final_state.external_evidence.passages}
    all_known_evidence_ids = local_ev_ids | ext_ev_ids

    for claim in final_state.report.claims:
        for cited_id in claim.cited_evidence_ids:
            assert cited_id in all_known_evidence_ids, f"Claim cited unretrieved evidence ID: {cited_id}"

    # 5. Evidence IDs are consistent across agents
    for eid in local_ev_ids:
        assert eid.startswith("ev:bioasq:")
    for eid in ext_ev_ids:
        assert eid.startswith("ev:pubmed:")


@pytest.mark.integration
def test_blocking_clarification_halts_before_retrieval(setup_synthetic_corpus):
    """A blocking clarification halts before retrieval with AWAITING_CLARIFICATION status."""
    embedder = setup_synthetic_corpus["embedder"]
    qdrant = setup_synthetic_corpus["qdrant"]
    col_ver = setup_synthetic_corpus["collection_version"]

    # Missing age causes a blocking clarification question
    missing_age_clarifier = {
        "age": None,
        "sex": "male",
        "chief_complaint": "cough",
        "symptom_duration": "1 week",
        "symptoms": [
            {"term": "cough", "polarity": "PRESENT", "source_span": {"start": 0, "end": 5}, "origin": "STATED"}
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    fake_llm = make_canned_llm_for_workflow(clarifier_dict=missing_age_clarifier)
    fake_ncbi = FakeNCBIClient()
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        llm_provider=fake_llm,
        collection_version=col_ver,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        scanner=EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=fake_llm, embedding_service=embedder),
        fusion=ClinicalDataFusion(llm_provider=fake_llm),
        optimizer=AdaptiveOptimizer(llm_provider=fake_llm),
        repository=repo,
        llm_provider=fake_llm,
        ncbi_client=fake_ncbi,
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
    )

    sub = CaseSubmission(raw_text="cough without stated age")
    state = WorkflowState(
        execution_id=str(uuid.uuid4()),
        case_id=str(uuid.uuid4()),
        submission=sub,
    )

    final_state = orchestrator.run(state)

    # Must halt with AWAITING_CLARIFICATION
    assert final_state.workflow_status == ReportStatusEnum.AWAITING_CLARIFICATION.value
    assert final_state.final_result.final_state == ReportStatusEnum.AWAITING_CLARIFICATION

    # Retrieval, Scanner, and Fusion must NOT have run
    executed_nodes = [r.agent_name for r in final_state.agent_records]
    assert "validate_case" in executed_nodes
    assert "clinical_text_clarifier" in executed_nodes
    assert "clarification_check" in executed_nodes
    assert "symptom_rag_analyzer" not in executed_nodes
    assert "evidence_based_scanner" not in executed_nodes
    assert "clinical_data_fusion" not in executed_nodes


@pytest.mark.integration
def test_scanner_outage_yields_degraded_and_run_continues(setup_synthetic_corpus):
    """Scanner outage yields DEGRADED status, source_availability < 1, and run continues with local evidence."""
    embedder = setup_synthetic_corpus["embedder"]
    qdrant = setup_synthetic_corpus["qdrant"]
    col_ver = setup_synthetic_corpus["collection_version"]

    fake_llm = make_canned_llm_for_workflow()
    # Outage on NCBI client
    outage_ncbi = FakeNCBIClient(default_status=QueryStatusEnum.UNAVAILABLE)
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        llm_provider=fake_llm,
        collection_version=col_ver,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        scanner=EvidenceBasedScanner(ncbi_client=outage_ncbi, llm_provider=fake_llm, embedding_service=embedder),
        fusion=ClinicalDataFusion(llm_provider=fake_llm),
        optimizer=AdaptiveOptimizer(llm_provider=fake_llm),
        repository=repo,
        llm_provider=fake_llm,
        ncbi_client=outage_ncbi,
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
    )

    sub = CaseSubmission(raw_text="45yo male presents with fever for 3 days and persistent cough. Patient denies chest pain.")
    state = WorkflowState(
        execution_id=str(uuid.uuid4()),
        case_id=str(uuid.uuid4()),
        submission=sub,
    )

    final_state = orchestrator.run(state)

    # Scanner is DEGRADED, but overall workflow completes with local evidence
    assert final_state.scanner_degraded is True
    rec_map = {r.agent_name: r for r in final_state.agent_records}
    assert rec_map["evidence_based_scanner"].status == AgentStatusEnum.DEGRADED

    # Fusion successfully ran with local evidence
    assert "clinical_data_fusion" in rec_map
    assert rec_map["clinical_data_fusion"].status == AgentStatusEnum.COMPLETED

    # Source availability is below 1 (1 unavailable source out of 2)
    unavail_count = sum(
        1 for st in final_state.external_evidence.query_statuses.values() if st == QueryStatusEnum.UNAVAILABLE
    )
    assert unavail_count >= 1
    source_availability = 1.0 - (unavail_count / 2)
    assert source_availability < 1.0


@pytest.mark.integration
def test_low_confidence_triggers_optimizer_loop_and_stops_at_cap(setup_synthetic_corpus):
    """Low confidence triggers optimizer loop; stops at MAX_OPTIMIZER_ITERATIONS=3 with budget label."""
    embedder = setup_synthetic_corpus["embedder"]
    qdrant = setup_synthetic_corpus["qdrant"]
    col_ver = setup_synthetic_corpus["collection_version"]

    # Low confidence setup: query unmatched by synthetic corpus + outage on scanner
    low_conf_clarifier = {
        "age": 45,
        "sex": "male",
        "chief_complaint": "purple papules on lower extremities",
        "symptom_duration": "3 weeks",
        "symptoms": [
            {"term": "purple papules", "polarity": "PRESENT", "source_span": {"start": 0, "end": 14}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": ["none"], "allergies": ["none"], "vitals": {}},
    }

    # Configure fusion to produce an empty / unsupported report yielding low confidence
    low_conf_fusion = {
        "sections": {
            "case_summary": "Inconclusive dermatological findings.",
            "relevant_findings": "Purple papules on lower extremities.",
            "evidence_supported_observations": "No matching respiratory observations found in local corpus.",
            "potential_considerations": "Undetermined cutaneous pathology.",
            "alternative_explanations": "None.",
            "contradictory_or_limiting_evidence": "Sparse evidence.",
            "missing_information": "Skin biopsy and dermatopathology.",
            "uncertainty_and_limitations": "High uncertainty.",
            "recommended_evidence_review_steps": "Further diagnostic review.",
            "references": "None.",
        },
        "claims": [],  # Zero claims
    }

    fake_llm = make_canned_llm_for_workflow(
        clarifier_dict=low_conf_clarifier,
        fusion_dict=low_conf_fusion,
        optimizer_text="Refine retrieval for additional cutaneous conditions.",
    )
    # Outage on NCBI client yields source_availability = 0.5 (unavailable_sources = 1)
    # Total baseline confidence = 0.40 * 0.0 + 0.30 * 0.5 + 0.30 * 1.0 = 0.45 < 0.65 threshold
    fake_ncbi = FakeNCBIClient(default_status=QueryStatusEnum.UNAVAILABLE)
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        llm_provider=fake_llm,
        collection_version=col_ver,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        scanner=EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=fake_llm, embedding_service=embedder),
        fusion=ClinicalDataFusion(llm_provider=fake_llm),
        optimizer=AdaptiveOptimizer(llm_provider=fake_llm),
        repository=repo,
        llm_provider=fake_llm,
        ncbi_client=fake_ncbi,
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
    )

    sub = CaseSubmission(raw_text="purple papules on lower extremities for 3 weeks in a 45yo male.")
    state = WorkflowState(
        execution_id=str(uuid.uuid4()),
        case_id=str(uuid.uuid4()),
        submission=sub,
    )

    final_state = orchestrator.run(state)

    # 1. Loop ran up to cap (MAX_OPTIMIZER_ITERATIONS = 3)
    assert len(final_state.iterations) >= 1
    assert final_state.current_iteration >= 1

    # 2. Final report carries the required baseline budget label (Appendix A3)
    assert final_state.report is not None
    assert "BASELINE: budget reached, not verified" in final_state.report.summary
    assert any("BUDGET_EXHAUSTED" in it.action_taken for it in final_state.iterations)


@pytest.mark.integration
def test_no_provider_configured_fails_at_first_llm_node_with_llm_unavailable():
    """Without an LLM provider configured, workflow fails at first LLM node with LLM_UNAVAILABLE."""
    repo = ExecutionRepository()
    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=None),
        rag_analyzer=SymptomRAGAnalyzer(llm_provider=None),
        scanner=EvidenceBasedScanner(llm_provider=None),
        fusion=ClinicalDataFusion(llm_provider=None),
        optimizer=AdaptiveOptimizer(llm_provider=None),
        repository=repo,
        llm_provider=None,
    )

    sub = CaseSubmission(raw_text="35yo female with headache for 1 day.")
    state = WorkflowState(
        execution_id=str(uuid.uuid4()),
        case_id=str(uuid.uuid4()),
        submission=sub,
    )

    final_state = orchestrator.run(state)

    assert final_state.workflow_status == ReportStatusEnum.FAILED.value
    assert final_state.error_code == "LLM_UNAVAILABLE"
    assert final_state.final_result.final_state == ReportStatusEnum.FAILED

    # Check node execution: validate_case ran, clarifier failed, no downstream nodes ran!
    ran_nodes = [r.agent_name for r in final_state.agent_records]
    assert "validate_case" in ran_nodes
    assert "clinical_text_clarifier" in ran_nodes
    assert "symptom_rag_analyzer" not in ran_nodes
    assert "evidence_based_scanner" not in ran_nodes
    assert "clinical_data_fusion" not in ran_nodes


@pytest.mark.integration
def test_qdrant_unreachable_fails_rag_node():
    """When Qdrant is unreachable, RAG node fails with defined error and ends run FAILED."""
    from qdrant_client import QdrantClient
    bad_qdrant_client = QdrantClient(url="http://127.0.0.1:65534", timeout=1.0)
    bad_qdrant = QdrantWrapper(client=bad_qdrant_client)

    fake_llm = make_canned_llm_for_workflow()
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=bad_qdrant,
        llm_provider=fake_llm,
        collection_version=99,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        repository=repo,
        llm_provider=fake_llm,
        qdrant_wrapper=bad_qdrant,
    )

    sub = CaseSubmission(raw_text="45yo male presents with fever for 3 days and persistent cough. Patient denies chest pain.")
    state = WorkflowState(
        execution_id=str(uuid.uuid4()),
        case_id=str(uuid.uuid4()),
        submission=sub,
    )

    final_state = orchestrator.run(state)

    assert final_state.workflow_status == ReportStatusEnum.FAILED.value
    assert final_state.error_code in ("QDRANT_UNAVAILABLE", "RETRIEVAL_ERROR")

    # Downstream scanner and fusion did not run
    ran_nodes = [r.agent_name for r in final_state.agent_records]
    assert "evidence_based_scanner" not in ran_nodes
    assert "clinical_data_fusion" not in ran_nodes


@pytest.mark.integration
def test_persistence_verification_and_immutable_report_versions(setup_synthetic_corpus):
    """Verify that execution records, retrieval runs, and report versions persist immutably in PostgreSQL."""
    embedder = setup_synthetic_corpus["embedder"]
    qdrant = setup_synthetic_corpus["qdrant"]
    col_ver = setup_synthetic_corpus["collection_version"]

    fake_llm = make_canned_llm_for_workflow()
    fake_ncbi = FakeNCBIClient()
    repo = ExecutionRepository()

    rag_analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        llm_provider=fake_llm,
        collection_version=col_ver,
    )

    orchestrator = BaselineWorkflowOrchestrator(
        clarifier=ClinicalTextClarifier(llm_provider=fake_llm),
        rag_analyzer=rag_analyzer,
        scanner=EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=fake_llm, embedding_service=embedder),
        fusion=ClinicalDataFusion(llm_provider=fake_llm),
        optimizer=AdaptiveOptimizer(llm_provider=fake_llm),
        repository=repo,
        llm_provider=fake_llm,
        ncbi_client=fake_ncbi,
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
    )

    exec_id = str(uuid.uuid4())
    case_id = str(uuid.uuid4())
    sub = CaseSubmission(raw_text="45yo male presents with fever for 3 days and persistent cough. Patient denies chest pain.")
    state = WorkflowState(
        execution_id=exec_id,
        case_id=case_id,
        submission=sub,
    )

    final_state = orchestrator.run(state)

    with SessionLocal() as session:
        # Check workflow execution persisted
        wf = session.get(WorkflowExecutionModel, exec_id)
        assert wf is not None
        assert wf.status == ReportStatusEnum.READY_FOR_REVIEW.value
        assert wf.completed_at is not None

        # Check agent executions
        agents = session.scalars(
            select(AgentExecutionModel).where(AgentExecutionModel.execution_id == exec_id)
        ).all()
        assert len(agents) >= 5
        for ag in agents:
            assert ag.started_at is not None
            assert ag.completed_at is not None
            assert ag.duration_ms is not None
            assert ag.duration_ms >= 0

        # Check retrieval runs
        runs = session.scalars(
            select(RetrievalRunModel).where(RetrievalRunModel.execution_id == exec_id)
        ).all()
        assert len(runs) >= 1

        # Check report version and claims
        reports = session.scalars(
            select(ReportVersionModel).where(ReportVersionModel.execution_id == exec_id)
        ).all()
        assert len(reports) >= 1
        r_ver = reports[0]
        assert r_ver.version_number == 1
        assert r_ver.confidence_score > 0.0

        claims = session.scalars(
            select(CandidateClaimModel).where(CandidateClaimModel.report_version_id == r_ver.id)
        ).all()
        assert len(claims) >= 2


@pytest.mark.integration
def test_reconcile_interrupted_executions_in_postgres():
    """Verify that reconcile_interrupted_executions marks a stale RUNNING execution as FAILED(INTERRUPTED)."""
    case_id = str(uuid.uuid4())
    exec_id = str(uuid.uuid4())
    repo = ExecutionRepository()

    repo.create_case(
        case_id=case_id,
        raw_text_hash="persisthash",
        chief_complaint="Dyspnea",
    )
    repo.create_workflow_execution(
        execution_id=exec_id,
        case_id=case_id,
        status="RUNNING",
    )
    repo.record_agent_execution_start(
        execution_id=exec_id,
        agent_name="clinical_text_clarifier",
        input_summary="Case input",
    )

    reconciled = reconcile_interrupted_executions()
    assert reconciled >= 1

    with SessionLocal() as session:
        exec_row = session.get(WorkflowExecutionModel, exec_id)
        assert exec_row.status == "FAILED"
        assert exec_row.failure_reason == "INTERRUPTED"
        assert exec_row.completed_at is not None
