"""Unit Tests for Evidence-Level Provenance and Completeness Invariants (EXT-C).

Specification Section 5.3:
- Relational edges linking case, finding, query, evidence chunk, document,
  claim, verification, report section, iteration, abstention, and review event.
- Relations: SUPPORTS, PARTIALLY_SUPPORTS, CONTRADICTS, INSUFFICIENT,
  DERIVED_FROM_PATIENT_INPUT, RETRIEVED_BY, GENERATED_IN.
- Mark claims and summaries kind=INTERPRETATION and passages kind=SOURCE.
- assert_provenance_complete(report_version) fails if any material claim has no edge.
- Edges correctly link to report_version_id.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.persistence.database import Base
from backend.app.persistence.models import (
    CandidateClaim as CandidateClaimModel,
    ClinicalCase as ClinicalCaseModel,
    ReportVersion as ReportVersionModel,
)
from backend.app.provenance import (
    ProvenanceEdgeSchema,
    ProvenanceIncompleteError,
    ProvenanceNodeKindEnum,
    ProvenanceRelationEnum,
    assert_provenance_complete,
    build_provenance_edges_from_workflow,
    classify_provenance_node,
    save_provenance_edges,
)
from backend.app.schemas.clinical_case import ClinicalCase, StructuredSymptom
from backend.app.schemas.evidence import RetrievedEvidenceSet, RetrievedPassageResult
from backend.app.schemas.reports import (
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
    VerificationMethodEnum,
)


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for testing provenance persistence."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_classify_provenance_node_interpretation_vs_source():
    """Verify claims and summaries are INTERPRETATION, and passages/documents are SOURCE (SPEC 5.3)."""
    claim_node = classify_provenance_node("claim", "claim-1", "Observation text")
    assert claim_node.kind == ProvenanceNodeKindEnum.INTERPRETATION

    summary_node = classify_provenance_node("summary", "sum-1", "Report summary text")
    assert summary_node.kind == ProvenanceNodeKindEnum.INTERPRETATION

    passage_node = classify_provenance_node("passage", "ev:pubmed:1234:0", "Abstract excerpt")
    assert passage_node.kind == ProvenanceNodeKindEnum.SOURCE

    doc_node = classify_provenance_node("document", "doc-1", "PubMed 1234")
    assert doc_node.kind == ProvenanceNodeKindEnum.SOURCE


def test_build_provenance_edges_from_real_agent_outputs():
    """Verify edges are generated with correct relations and reference report_version_id."""
    version_id = "rv-100"
    case_id = "case-100"

    finding = StructuredSymptom(
        term="cough",
        polarity="PRESENT",
        source_span={"start": 0, "end": 5},
        origin="STATED",
    )
    case = ClinicalCase(
        case_id=case_id,
        raw_text_hash="c" * 64,
        chief_complaint="Persistent dry cough",
        age=45,
        symptom_duration="3 days",
        structured_symptoms=[finding],
    )

    claim1 = CandidateClaim(
        claim_id="claim-sup-1",
        report_version_id=version_id,
        claim_text="Symptom is consistent with viral bronchitis.",
        section="evidence_supported_observations",
        is_material=True,
        cited_evidence_ids=["ev:pubmed:1001:0"],
    )
    claim2 = CandidateClaim(
        claim_id="claim-contra-2",
        report_version_id=version_id,
        claim_text="Patient has active bacterial pneumonia.",
        section="alternative_explanations",
        is_material=True,
        cited_evidence_ids=["ev:pubmed:1002:0"],
    )

    report = CandidateClinicalReport(
        case_id=case_id,
        report_version_id=version_id,
        version_number=1,
        summary="Clinical synthesis candidate summary",
        claims=[claim1, claim2],
        confidence_score=0.82,
        status=ReportStatusEnum.READY_FOR_REVIEW,
    )

    verif1 = ClaimVerification(
        claim_id=claim1.claim_id,
        claim_text=claim1.claim_text,
        category="observation",
        status=ClaimVerificationStatusEnum.SUPPORTED,
        supporting_evidence_ids=["ev:pubmed:1001:0"],
        justification="Passage confirms viral etiology.",
        method=VerificationMethodEnum.HYBRID,
    )
    verif2 = ClaimVerification(
        claim_id=claim2.claim_id,
        claim_text=claim2.claim_text,
        category="consideration",
        status=ClaimVerificationStatusEnum.CONTRADICTED,
        contradicting_evidence_ids=["ev:pubmed:1002:0"],
        justification="Passage refutes bacterial infection in this presentation.",
        method=VerificationMethodEnum.HYBRID,
    )

    retrieved = RetrievedEvidenceSet(
        query_id="q-1",
        results=[
            RetrievedPassageResult(
                document_id="doc-1",
                evidence_id="ev:pubmed:1001:0",
                chunk_id="chunk-1",
                source="pubmed",
                retrieval_score=0.88,
                rank=1,
                corpus_version="v1",
                embedding_model="default",
                query_id="q-1",
                text="[SYNTHETIC] Viral bronchitis clinical overview.",
            )
        ],
    )

    edges = build_provenance_edges_from_workflow(
        report=report,
        case=case,
        verifications=[verif1, verif2],
        retrieved_evidence=retrieved,
    )

    # Check edges
    assert len(edges) >= 4
    for e in edges:
        assert e.report_version_id == version_id
        assert e.case_id == case_id

    relations = [e.relation for e in edges]
    assert ProvenanceRelationEnum.GENERATED_IN in relations
    assert ProvenanceRelationEnum.SUPPORTS in relations
    assert ProvenanceRelationEnum.CONTRADICTS in relations
    assert ProvenanceRelationEnum.RETRIEVED_BY in relations

    # assert_provenance_complete must pass since all material claims have edges
    assert assert_provenance_complete(report, edges=edges) is True


def test_assert_provenance_complete_fails_when_material_claim_lacks_edge():
    """Safety Invariant: assert_provenance_complete strictly fails if any material claim has no edge."""
    version_id = "rv-fail-1"
    claim_covered = CandidateClaim(
        claim_id="claim-covered",
        report_version_id=version_id,
        claim_text="Covered claim observation.",
        section="observations",
        is_material=True,
    )
    claim_uncovered = CandidateClaim(
        claim_id="claim-uncovered",
        report_version_id=version_id,
        claim_text="Ungrounded material claim observation.",
        section="observations",
        is_material=True,
    )

    report = CandidateClinicalReport(
        case_id="case-1",
        report_version_id=version_id,
        version_number=1,
        summary="Summary",
        claims=[claim_covered, claim_uncovered],
        confidence_score=0.75,
    )

    # Provide edge for only claim_covered
    edge = ProvenanceEdgeSchema(
        report_version_id=version_id,
        case_id="case-1",
        claim_id=claim_covered.claim_id,
        relation=ProvenanceRelationEnum.GENERATED_IN,
    )

    with pytest.raises(ProvenanceIncompleteError) as exc_info:
        assert_provenance_complete(report, edges=[edge])

    assert "claim-uncovered" in exc_info.value.missing_claim_ids
    assert exc_info.value.report_version_id == version_id


def test_provenance_persistence_in_database(sqlite_session):
    """Verify saving provenance edges in PostgreSQL/SQLite table."""
    case = ClinicalCaseModel(id="c-db", raw_text_hash="h"*64, chief_complaint="Chest pain")
    sqlite_session.add(case)
    sqlite_session.flush()

    report_row = ReportVersionModel(
        id="rv-db",
        case_id="c-db",
        execution_id="exec-db",
        version_number=1,
        summary="Report summary",
        confidence_score=0.85,
        status="READY_FOR_REVIEW",
    )
    sqlite_session.add(report_row)
    sqlite_session.flush()

    claim_row = CandidateClaimModel(
        id="cl-db",
        report_version_id="rv-db",
        claim_text="Finding is grounded.",
        section="observations",
        is_material=True,
    )
    sqlite_session.add(claim_row)
    sqlite_session.commit()

    edge_schema = ProvenanceEdgeSchema(
        report_version_id="rv-db",
        case_id="c-db",
        claim_id="cl-db",
        relation=ProvenanceRelationEnum.SUPPORTS,
    )

    saved = save_provenance_edges(sqlite_session, [edge_schema])
    assert len(saved) == 1
    assert saved[0].report_version_id == "rv-db"
    assert saved[0].claim_id == "cl-db"
    assert saved[0].relation == "SUPPORTS"

    # Completeness check against database
    assert assert_provenance_complete(report_row, session=sqlite_session) is True


def test_provenance_completeness_failure_diverts_to_abstained_system_error():
    """Safety Invariant: A provenance completeness failure ends in abstention with SYSTEM_ERROR and NEVER in ready."""
    from backend.app.orchestration.extended_workflow import ExtendedWorkflowOrchestrator
    from backend.app.persistence.execution_repository import ExecutionRepository
    from backend.app.schemas.executions import WorkflowState
    from backend.app.schemas.reports import AbstentionReasonCode
    from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider
    from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient

    orchestrator = ExtendedWorkflowOrchestrator(
        llm_provider=FakeLLMProvider(),
        ncbi_client=FakeNCBIClient(),
        repository=ExecutionRepository(),
    )

    uncovered_claim = CandidateClaim(
        claim_id="uncovered-claim-1",
        report_version_id="rv-uncovered",
        claim_text="Material claim citing evidence that lacks verification edges.",
        section="observations",
        is_material=True,
        cited_evidence_ids=["ev:bioasq:nonexistent:0"],
        source_spans=[],
    )
    report = CandidateClinicalReport(
        case_id="case-prov-fail",
        report_version_id="rv-uncovered",
        version_number=1,
        summary="Draft report",
        claims=[uncovered_claim],
        confidence_score=0.85,
    )

    state = WorkflowState(
        execution_id="exec-prov-fail",
        case_id="case-prov-fail",
        report=report,
        workflow_status="RUNNING",
    )

    # Calling finalize_ready must trigger the provenance completeness check
    result = orchestrator.node_finalize_ready(state)

    # Invariant: Must end in ABSTAINED, NEVER in READY_FOR_REVIEW
    assert result["workflow_status"] == ReportStatusEnum.ABSTAINED.value
    assert result["workflow_status"] != ReportStatusEnum.READY_FOR_REVIEW.value
    assert result["abstention"].reason_codes[0] == AbstentionReasonCode.SYSTEM_ERROR
    assert AbstentionReasonCode.SYSTEM_ERROR in result["abstention"].reason_codes
    assert "provenance completeness assertion failed" in result["abstention"].explanation
