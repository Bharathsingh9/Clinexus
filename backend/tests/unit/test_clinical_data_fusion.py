"""Unit tests for Clinical Data Fusion (Agent 4 - AGT-4, App A1, A8).

Specification Section 4.4, Section 8, Appendix A1, A8:
- Without LLM provider, returns FAILED with LLM_UNAVAILABLE.
- Untrusted data prompt block defense (<UNTRUSTED_EVIDENCE data-only="true">).
- 10 required report sections generated.
- Citations validated: unretrieved evidence ID re-asked once, dropped if still invalid.
- Forbidden language lint fails report containing imperative dosing or treatment directives.
- Strict safety: No treatment guidance.
"""

import json
import pytest
from backend.app.agents.clinical_data_fusion import ClinicalDataFusion
from backend.app.schemas.clinical_case import (
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)
from backend.app.schemas.evidence import (
    EvidenceChunk,
    EvidenceDocument,
    ExternalEvidenceSet,
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)
from backend.app.schemas.executions import AgentStatusEnum
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


def make_test_case():
    return ClinicalCase(
        raw_text_hash="c" * 64,
        age=45,
        sex="male",
        chief_complaint="Chest tightness and dyspnea",
        symptom_duration="2 days",
        structured_symptoms=[
            StructuredSymptom(
                term="chest tightness",
                normalized_term="chest_pain",
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=0, end=15),
                origin=OriginEnum.STATED,
            ),
        ],
        patient_context=PatientContext(),
    )


def make_test_evidence():
    local_ev = RetrievedEvidenceSet(
        query_id="q-local-1",
        results=[
            RetrievedPassageResult(
                evidence_id="ev:bioasq:1001:0",
                document_id="doc-1",
                chunk_id="chk-1",
                title="Angina presentation",
                source="BIOASQ",
                retrieval_score=0.88,
                rank=1,
                corpus_version="v1",
                embedding_model="biobert",
                query_id="q-local-1",
                text="Angina typically presents with retrosternal chest pressure and dyspnea.",
            )
        ],
    )
    ext_ev = ExternalEvidenceSet(
        query_ids=["q-ext-1"],
        documents=[
            EvidenceDocument(
                source="PUBMED",
                source_id="20001",
                title="Cardiac ischemia evaluation",
                pmid="20001",
                abstract="Clinical evaluation of cardiac ischemia in adults.",
            )
        ],
        passages=[
            EvidenceChunk(
                document_id="doc-ext-1",
                evidence_id="ev:pubmed:20001:0",
                chunk_index=0,
                text="Cardiac ischemia must be considered when chest tightness accompanies exertion.",
                text_hash="h" * 64,
            )
        ],
    )
    return local_ev, ext_ev


def test_fusion_returns_llm_unavailable_without_provider():
    """Without an LLM provider, Agent 4 must return FAILED with LLM_UNAVAILABLE."""
    fusion = ClinicalDataFusion(llm_provider=None)
    case = make_test_case()
    res = fusion.run(case)

    assert res.status == AgentStatusEnum.FAILED
    assert res.error_code == "LLM_UNAVAILABLE"


def test_fusion_untrusted_prompt_blocks():
    """Verify prompt surrounds evidence in untrusted data blocks (SPEC 8)."""
    fake_llm = FakeLLMProvider(default_text=json.dumps({"sections": {}, "claims": []}))
    fusion = ClinicalDataFusion(llm_provider=fake_llm)

    case = make_test_case()
    local_ev, ext_ev = make_test_evidence()

    prompt = fusion._build_untrusted_prompt(case, local_ev, ext_ev)
    assert '<UNTRUSTED_EVIDENCE source="LOCAL_RAG" data-only="true">' in prompt
    assert '<UNTRUSTED_EVIDENCE source="EXTERNAL_PUBMED" data-only="true">' in prompt
    assert "</UNTRUSTED_EVIDENCE>" in prompt
    assert "<PATIENT_INPUT>" in prompt
    assert "</PATIENT_INPUT>" in prompt


def test_fusion_generates_10_sections_and_valid_claims():
    """Success path: generates all 10 standard report sections and valid claims."""
    canned_report = {
        "sections": {
            "case_summary": "Patient presenting with acute chest tightness.",
            "relevant_findings": "Chest tightness and dyspnea present.",
            "evidence_supported_observations": "Angina presents with retrosternal chest pressure [ev:bioasq:1001:0].",
            "potential_considerations": "Considerations include angina and cardiac ischemia [ev:pubmed:20001:0].",
            "alternative_explanations": "Musculoskeletal strain is a possible non-cardiac etiology.",
            "contradictory_or_limiting_evidence": "No electrocardiographic findings provided.",
            "missing_information": "Cardiac enzyme panel and ECG are missing.",
            "uncertainty_and_limitations": "Evaluation is limited to narrative symptoms.",
            "recommended_evidence_review_steps": "Review troponin levels and 12-lead ECG.",
            "references": "BioASQ 1001, PubMed 20001.",
        },
        "claims": [
            {
                "claim_text": "Angina presents with retrosternal chest pressure.",
                "section": "evidence_supported_observations",
                "cited_evidence_ids": ["ev:bioasq:1001:0"],
            },
            {
                "claim_text": "Cardiac ischemia is a candidate differential consideration.",
                "section": "potential_considerations",
                "cited_evidence_ids": ["ev:pubmed:20001:0"],
            },
        ],
    }
    fake_llm = FakeLLMProvider(default_text=json.dumps(canned_report))
    fusion = ClinicalDataFusion(llm_provider=fake_llm)

    case = make_test_case()
    local_ev, ext_ev = make_test_evidence()
    report = fusion.run(case, local_evidence=local_ev, external_evidence=ext_ev)

    # All 10 sections present
    assert len(report.sections) == 10
    assert "case_summary" in report.sections
    assert "recommended_evidence_review_steps" in report.sections

    # Claims validated
    assert len(report.claims) == 2
    assert report.confidence_score > 0.0


def test_fusion_unretrieved_citation_reask_and_drop():
    """When a claim cites an unretrieved ID, re-ask once; if still unretrieved, drop claim."""
    # Attempt 1: cites unretrieved ev:bioasq:999999:0
    attempt1 = {
        "sections": {"case_summary": "Test summary"},
        "claims": [
            {
                "claim_text": "Unverified assertion.",
                "section": "evidence_supported_observations",
                "cited_evidence_ids": ["ev:bioasq:999999:0"],  # Unretrieved!
            }
        ],
    }
    # Attempt 2 (repair response): still cites unretrieved ID
    attempt2 = {
        "claims": [
            {
                "claim_text": "Still unverified assertion.",
                "section": "evidence_supported_observations",
                "cited_evidence_ids": ["ev:bioasq:999999:0"],
            }
        ]
    }

    call_count = 0
    fake_llm = FakeLLMProvider()

    def fake_generate_text(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps(attempt1)
        return json.dumps(attempt2)

    fake_llm.generate_text = fake_generate_text
    fusion = ClinicalDataFusion(llm_provider=fake_llm)

    case = make_test_case()
    local_ev, ext_ev = make_test_evidence()
    report = fusion.run(case, local_evidence=local_ev, external_evidence=ext_ev)

    # Claim should be DROPPED
    assert len(report.claims) == 0
    assert call_count == 2  # Proves re-ask occurred


def test_fusion_fails_on_forbidden_language():
    """If the LLM synthesizes forbidden dosing or prescription language, report fails."""
    violating_report = {
        "sections": {
            "case_summary": "Patient presenting with acute chest tightness.",
            "evidence_supported_observations": "Patient must take 500mg aspirin immediately.",  # FORBIDDEN!
        },
        "claims": [],
    }
    fake_llm = FakeLLMProvider(default_text=json.dumps(violating_report))
    fusion = ClinicalDataFusion(llm_provider=fake_llm)

    case = make_test_case()
    local_ev, ext_ev = make_test_evidence()
    result = fusion.run(case, local_evidence=local_ev, external_evidence=ext_ev)

    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == "FORBIDDEN_LANGUAGE_DETECTED"
