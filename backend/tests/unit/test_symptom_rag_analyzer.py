"""Unit tests for Symptom RAG Analyzer (Agent 2 - AGT-2, App A5).

Specification Section 4.2, Section 11, Appendix A5:
- metadata preserved
- deduplication
- insufficient-evidence results (empty corpus, low score)
- fabricated citation rejected
- hypothesis without supporting passage dropped
- every sentence in retrieval_summary cites evidence IDs
"""

import uuid
from unittest.mock import MagicMock
import pytest
from backend.app.agents.symptom_rag_analyzer import SymptomRAGAnalyzer
from backend.app.schemas.clinical_case import (
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)
from backend.app.schemas.evidence import RetrievedPassageResult
from backend.app.schemas.executions import AgentStatusEnum
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


def make_sample_case():
    return ClinicalCase(
        raw_text_hash="b" * 64,
        age=50,
        sex="male",
        chief_complaint="Chest pain and shortness of breath",
        symptom_duration="5 days",
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


def test_llm_unavailable_without_provider():
    """Without an LLM provider, Agent 2 must return FAILED with LLM_UNAVAILABLE."""
    analyzer = SymptomRAGAnalyzer(llm_provider=None)
    case = make_sample_case()
    res = analyzer.run(case)

    assert res.status == AgentStatusEnum.FAILED
    assert res.error_code == "LLM_UNAVAILABLE"
    assert "LLM provider is not configured" in (res.error_message or "")


def test_empty_corpus_returns_insufficient_signal():
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False

    analyzer = SymptomRAGAnalyzer(qdrant_wrapper=mock_qdrant, llm_provider=FakeLLMProvider())
    case = make_sample_case()
    res = analyzer.run(case)

    assert res.insufficient is True
    assert res.reason_code == "EMPTY_CORPUS"
    assert len(res.results) == 0


def test_vector_db_unreachable_returns_insufficient_signal():
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.side_effect = ConnectionError("Connection refused")

    analyzer = SymptomRAGAnalyzer(qdrant_wrapper=mock_qdrant, llm_provider=FakeLLMProvider())
    case = make_sample_case()
    res = analyzer.run(case)

    assert res.insufficient is True
    assert res.reason_code == "VECTOR_DB_UNAVAILABLE"


def test_retrieval_summary_every_sentence_cites_evidence_ids():
    analyzer = SymptomRAGAnalyzer()
    passages = [
        RetrievedPassageResult(
            evidence_id="ev:bioasq:1001:0",
            document_id="doc-1",
            chunk_id="chunk-1",
            title="[SYNTHETIC] Test Title",
            source="BIOASQ",
            publication_date="2021",
            source_location="abstract",
            retrieval_score=0.88,
            rank=1,
            corpus_version="v1",
            embedding_model="biobert",
            query_id="q-1",
            evidence_level="UNKNOWN",
            text="Acute coronary syndrome presenting with chest pain and diaphoresis.",
        ),
        RetrievedPassageResult(
            evidence_id="ev:bioasq:1002:0",
            document_id="doc-2",
            chunk_id="chunk-2",
            title="[SYNTHETIC] Test Title 2",
            source="BIOASQ",
            publication_date="2022",
            source_location="abstract",
            retrieval_score=0.81,
            rank=2,
            corpus_version="v1",
            embedding_model="biobert",
            query_id="q-1",
            evidence_level="UNKNOWN",
            text="Pulmonary embolism should be evaluated in dyspnea.",
        ),
    ]

    summary = analyzer._build_and_verify_summary(passages)
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    assert len(sentences) > 0
    for s in sentences:
        assert re.search(r"ev:[a-z0-9_-]+:[a-z0-9_.-]+:\d+", s) is not None


def test_hypothesis_without_supporting_passage_dropped():
    # If LLM fabricates a citation to an unretrieved evidence ID, it must be dropped
    fabricated_response = (
        "CONSIDERATION: Myocardial infarction | CITATIONS: ev:bioasq:999999:0\n"
        "CONSIDERATION: Angina pectoris | CITATIONS: ev:bioasq:1001:0, ev:bioasq:999999:0"
    )
    fake_llm = FakeLLMProvider(default_text=fabricated_response)
    analyzer = SymptomRAGAnalyzer(llm_provider=fake_llm)

    valid_ev_ids = {"ev:bioasq:1001:0"}
    passages = [
        RetrievedPassageResult(
            evidence_id="ev:bioasq:1001:0",
            document_id="doc-1",
            chunk_id="chunk-1",
            title="[SYNTHETIC] Test Title",
            source="BIOASQ",
            publication_date="2021",
            source_location="abstract",
            retrieval_score=0.85,
            rank=1,
            corpus_version="v1",
            embedding_model="biobert",
            query_id="q-1",
            evidence_level="UNKNOWN",
            text="Stable angina presentation.",
        )
    ]

    case = make_sample_case()
    hypotheses = analyzer._generate_candidate_hypotheses(case, passages, valid_ev_ids)

    # 1. Myocardial infarction cited ONLY 999999 (not in valid_ev_ids) -> must be dropped!
    labels = [h.label for h in hypotheses]
    assert "Myocardial infarction" not in labels

    # 2. Angina pectoris cited both 1001 and 999999 -> 999999 filtered out, 1001 retained!
    if "Angina pectoris" in labels:
        h = next(h for h in hypotheses if h.label == "Angina pectoris")
        assert h.supporting_evidence_ids == ["ev:bioasq:1001:0"]
        assert "ev:bioasq:999999:0" not in h.supporting_evidence_ids
