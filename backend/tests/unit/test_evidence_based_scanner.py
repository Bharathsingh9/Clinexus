"""Unit tests for Evidence-Based Scanner (Agent 3 - AGT-3, App A2, A5, A9).

Specification Section 4.3, Appendix A2, A5, A9:
- Offline mode returns SKIPPED_OFFLINE without requiring LLM or network.
- Without LLM provider in online mode, returns FAILED with LLM_UNAVAILABLE.
- Formulates per-hypothesis PubMed queries and gap queries.
- Deduplicates against local evidence.
- Records without abstracts (abstract_missing=True) excluded from passages.
- Chunks abstracts into platform evidence IDs: ev:pubmed:<pmid>:<idx>.
- Query statuses tracked accurately.
"""

import pytest
from backend.app.agents.evidence_based_scanner import EvidenceBasedScanner
from backend.app.schemas.clinical_case import (
    CandidateHypothesis,
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)
from backend.app.schemas.evidence import (
    EvidenceDocument,
    QueryStatusEnum,
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)
from backend.app.schemas.executions import AgentStatusEnum
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider
from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient


def make_test_case():
    return ClinicalCase(
        raw_text_hash="a" * 64,
        age=52,
        sex="female",
        chief_complaint="Chest pain and shortness of breath",
        symptom_duration="3 days",
        structured_symptoms=[
            StructuredSymptom(
                term="chest pain",
                normalized_term="chest_pain",
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=0, end=10),
                origin=OriginEnum.STATED,
            ),
            StructuredSymptom(
                term="shortness of breath",
                normalized_term="dyspnea",
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=15, end=34),
                origin=OriginEnum.STATED,
            ),
        ],
        patient_context=PatientContext(),
    )


def test_scanner_offline_mode_returns_skipped_offline_without_llm():
    """In offline mode, scanner returns SKIPPED_OFFLINE and makes zero network or LLM calls."""
    fake_ncbi = FakeNCBIClient(offline=True)
    # Notice: llm_provider is None!
    scanner = EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=None)

    case = make_test_case()
    result = scanner.run(case)

    assert result.query_statuses.get("q:external:offline") == QueryStatusEnum.SKIPPED_OFFLINE
    assert "offline" in result.summary.lower()
    assert len(fake_ncbi.calls) == 0


def test_scanner_returns_llm_unavailable_when_online_without_provider():
    """In online mode, running without an LLM provider must return FAILED with LLM_UNAVAILABLE."""
    fake_ncbi = FakeNCBIClient(offline=False)
    scanner = EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=None)

    case = make_test_case()
    result = scanner.run(case)

    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == "LLM_UNAVAILABLE"


def test_scanner_builds_hypothesis_and_gap_queries_and_chunks():
    """Online success path: per-hypothesis queries + gap queries + abstract chunking."""
    fake_ncbi = FakeNCBIClient(offline=False)
    scanner = EvidenceBasedScanner(
        ncbi_client=fake_ncbi,
        llm_provider=FakeLLMProvider(),
    )

    case = make_test_case()
    local_retrieval = RetrievedEvidenceSet(
        query_id="q-local-1",
        results=[],
        candidate_hypotheses=[
            CandidateHypothesis(
                hypothesis_id="hyp-1",
                label="Acute Coronary Syndrome",
                supporting_evidence_ids=["ev:bioasq:1001:0"],
                retrieval_scores=[0.88],
            )
        ],
    )

    result = scanner.run(case, retrieved_evidence=local_retrieval, gap_hints=["atypical angina"])

    # Must execute both hypothesis query and gap query
    assert len(result.query_ids) >= 2
    assert any("hyp:1" in qid for qid in result.query_ids)
    assert any("gap:" in qid for qid in result.query_ids)

    # Documents and passages created
    assert len(result.documents) > 0
    assert len(result.passages) > 0
    # Evidence ID check: ev:pubmed:<pmid>:<idx>
    p = result.passages[0]
    assert p.evidence_id.startswith("ev:pubmed:")


def test_scanner_deduplicates_against_local_evidence():
    """External documents already retrieved locally are deduplicated."""
    # Local evidence has PMID 30001
    local_retrieval = RetrievedEvidenceSet(
        query_id="q-local-1",
        results=[
            RetrievedPassageResult(
                evidence_id="ev:bioasq:30001:0",
                document_id="doc-local-1",
                chunk_id="c-1",
                title="Local title",
                source="BIOASQ",
                retrieval_score=0.90,
                rank=1,
                corpus_version="v1",
                embedding_model="biobert",
                query_id="q-local-1",
                text="[SYNTHETIC] Detailed clinical abstract describing acute presentation and diagnostic considerations.",
            )
        ],
        candidate_hypotheses=[],
    )

    # Fake client returns document with pmid 30001
    fake_ncbi = FakeNCBIClient(offline=False)
    scanner = EvidenceBasedScanner(
        ncbi_client=fake_ncbi,
        llm_provider=FakeLLMProvider(),
    )

    case = make_test_case()
    result = scanner.run(case, retrieved_evidence=local_retrieval)

    # Document 30001 must be deduplicated
    pmids = [d.pmid for d in result.documents]
    assert "30001" not in pmids


def test_scanner_excludes_missing_abstract_from_passages():
    """Records with abstract_missing=True are preserved as documents but excluded from passage chunks."""
    doc_no_abstract = EvidenceDocument(
        source="PUBMED",
        source_id="40001",
        title="[SYNTHETIC] Brief letter",
        journal="Medical Journal",
        publication_date="2021",
        pmid="40001",
        abstract=None,
        abstract_missing=True,
        evidence_level="Letter",
    )
    fake_ncbi = FakeNCBIClient(
        canned_documents={"(differential diagnosis)": [doc_no_abstract]},
        default_status=QueryStatusEnum.OK,
    )
    # Force client to return doc_no_abstract for all queries
    fake_ncbi.search_and_fetch = lambda query, max_results=None: ([doc_no_abstract], QueryStatusEnum.OK)

    scanner = EvidenceBasedScanner(
        ncbi_client=fake_ncbi,
        llm_provider=FakeLLMProvider(),
    )

    case = make_test_case()
    result = scanner.run(case)

    # Document present
    assert len(result.documents) == 1
    assert result.documents[0].abstract_missing is True
    # Passages excluded
    assert len(result.passages) == 0
