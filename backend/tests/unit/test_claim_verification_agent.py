"""Unit tests for Claim Verification Agent (EXT-A).

Specification Section 5.1 (Req ID: EXT-A).
Tests verify:
- Fabricated citation produces UNVERIFIABLE with CITATION_NOT_FOUND.
- A number absent from every cited passage CAN NEVER be SUPPORTED even when fake model says SUPPORTED.
- A judgment without a valid span CAN NEVER be SUPPORTED or CONTRADICTED.
- Each of the five statuses is produced in at least one test:
    * SUPPORTED
    * PARTIALLY_SUPPORTED
    * CONTRADICTED
    * INSUFFICIENT_EVIDENCE
    * UNVERIFIABLE
- Contradicting passages in other retrieved evidence are detected and listed.
- Verifier failure gets one bounded retry, then returns UNVERIFIABLE with VERIFIER_ERROR.
- Without an LLM provider, returns AgentExecutionRecord FAILED(LLM_UNAVAILABLE).
"""

from datetime import datetime, timezone
import json
import pytest

from backend.app.agents.claim_verification_agent import ClaimVerificationAgent
from backend.app.schemas.clinical_case import ClinicalCase, SourceSpan
from backend.app.schemas.evidence import EvidenceChunk
from backend.app.schemas.executions import AgentStatusEnum
from backend.app.schemas.reports import (
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    VerificationMethodEnum,
)
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


def _make_chunk(evidence_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        document_id="doc-test",
        evidence_id=evidence_id,
        chunk_index=0,
        text=text,
        text_hash="a" * 64,
    )


def _make_report(claims: list[CandidateClaim]) -> CandidateClinicalReport:
    return CandidateClinicalReport(
        version_number=1,
        case_id="case-101",
        summary="Clinical test report",
        claims=claims,
        confidence_score=0.75,
    )


def test_verifier_returns_llm_unavailable_without_provider():
    """Without an LLM provider, verifier returns AgentExecutionRecord with status FAILED and LLM_UNAVAILABLE."""
    agent = ClaimVerificationAgent(llm_provider=None)
    report = _make_report([])
    res = agent.run(report=report, evidence_passages=[])

    assert not isinstance(res, list)
    assert res.status == AgentStatusEnum.FAILED
    assert res.error_code == "LLM_UNAVAILABLE"


def test_fabricated_citation_gives_unverifiable():
    """Step 1 (Deterministic): A claim citing an evidence ID absent from this run's evidence set is UNVERIFIABLE with CITATION_NOT_FOUND."""
    fake_llm = FakeLLMProvider()
    agent = ClaimVerificationAgent(llm_provider=fake_llm)

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Troponin level was markedly elevated.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:99999999:0"],  # Not in evidence set!
    )
    report = _make_report([claim])
    real_chunks = [_make_chunk("ev:pubmed:12345678:0", "[SYNTHETIC] Serum troponin is a marker of cardiac injury.")]

    verifs = agent.run(report=report, evidence_passages=real_chunks)

    assert isinstance(verifs, list)
    assert len(verifs) == 1
    v = verifs[0]
    assert v.status == ClaimVerificationStatusEnum.UNVERIFIABLE
    assert v.limitations == "CITATION_NOT_FOUND"
    assert v.method == VerificationMethodEnum.DETERMINISTIC


def test_numeric_mismatch_vetoes_supported_status():
    """Step 2 (Deterministic): A claim with a number absent from every cited passage can NEVER be SUPPORTED, whatever the model says."""
    # Fake LLM attempts to return SUPPORTED with valid span
    passage_text = "[SYNTHETIC] Clinical studies indicate response rates of approximately 45 percent in treated adults."
    fake_output = json.dumps({
        "status": "SUPPORTED",
        "justification": "Evidence directly supports the claimed percentage.",
        "passage_index": 0,
        "span_start": 40,
        "span_end": 75,
    })
    fake_llm = FakeLLMProvider(default_text=fake_output)
    agent = ClaimVerificationAgent(llm_provider=fake_llm)

    # Claim contains number 85, which is absent from passage text (only 45 is present)
    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Clinical response rate was observed in 85 percent of treated patients.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:1111:0"],
    )
    report = _make_report([claim])
    chunks = [_make_chunk("ev:pubmed:1111:0", passage_text)]

    verifs = agent.run(report=report, evidence_passages=chunks)

    assert isinstance(verifs, list)
    assert len(verifs) == 1
    v = verifs[0]
    # Must NOT be SUPPORTED
    assert v.status != ClaimVerificationStatusEnum.SUPPORTED
    assert v.status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED
    assert v.limitations == "NUMERIC_MISMATCH"


def test_invalid_span_vetoes_supported_and_contradicted():
    """Step 3 (Model-based): A judgment without a valid span CAN NEVER be SUPPORTED or CONTRADICTED."""
    passage_text = "[SYNTHETIC] Bacterial pneumonia typically presents with localized consolidation."
    # Model returns SUPPORTED but invalid span offsets (-1, -1)
    fake_output = json.dumps({
        "status": "SUPPORTED",
        "justification": "General alignment with text.",
        "passage_index": 0,
        "span_start": -1,
        "span_end": -1,
    })
    fake_llm = FakeLLMProvider(default_text=fake_output)
    agent = ClaimVerificationAgent(llm_provider=fake_llm)

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Bacterial pneumonia presents with consolidation.",
        section="potential_considerations",
        cited_evidence_ids=["ev:pubmed:2222:0"],
    )
    report = _make_report([claim])
    chunks = [_make_chunk("ev:pubmed:2222:0", passage_text)]

    verifs = agent.run(report=report, evidence_passages=chunks)

    assert isinstance(verifs, list)
    v = verifs[0]
    # Since span was invalid, cannot be SUPPORTED
    assert v.status != ClaimVerificationStatusEnum.SUPPORTED
    assert v.status == ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE
    assert v.limitations == "INVALID_EVIDENCE_SPAN"


def test_all_five_claim_statuses_produced():
    """Verify that all five ClaimVerificationStatusEnum values can be generated."""
    passage_text = "[SYNTHETIC] Patient showed positive improvement with conservative fluid therapy."
    chunks = [_make_chunk("ev:pubmed:3333:0", passage_text)]

    # 1. SUPPORTED (with valid span and no numeric mismatch)
    resp_sup = json.dumps({
        "status": "SUPPORTED",
        "justification": "Fluid therapy showed positive improvement.",
        "passage_index": 0,
        "span_start": 26,
        "span_end": 79,
    })
    agent_sup = ClaimVerificationAgent(llm_provider=FakeLLMProvider(default_text=resp_sup))
    claim_sup = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Patient showed improvement with fluid therapy.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:3333:0"],
    )
    verifs_sup = agent_sup.run(_make_report([claim_sup]), chunks)
    assert verifs_sup[0].status == ClaimVerificationStatusEnum.SUPPORTED

    # 2. CONTRADICTED (with valid span)
    resp_contra = json.dumps({
        "status": "CONTRADICTED",
        "justification": "Evidence refutes fluid therapy as ineffective.",
        "passage_index": 0,
        "span_start": 0,
        "span_end": 35,
    })
    agent_contra = ClaimVerificationAgent(llm_provider=FakeLLMProvider(default_text=resp_contra))
    claim_contra = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Fluid therapy is completely ineffective.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:3333:0"],
    )
    verifs_contra = agent_contra.run(_make_report([claim_contra]), chunks)
    assert verifs_contra[0].status == ClaimVerificationStatusEnum.CONTRADICTED

    # 3. PARTIALLY_SUPPORTED
    resp_part = json.dumps({
        "status": "PARTIALLY_SUPPORTED",
        "justification": "Partially mentioned in fluid management context.",
        "passage_index": 0,
        "span_start": 10,
        "span_end": 40,
    })
    agent_part = ClaimVerificationAgent(llm_provider=FakeLLMProvider(default_text=resp_part))
    verifs_part = agent_part.run(_make_report([claim_sup]), chunks)
    assert verifs_part[0].status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED

    # 4. INSUFFICIENT_EVIDENCE
    resp_insuf = json.dumps({
        "status": "INSUFFICIENT_EVIDENCE",
        "justification": "Passage does not discuss pediatric cases.",
        "passage_index": 0,
        "span_start": 0,
        "span_end": 0,
    })
    agent_insuf = ClaimVerificationAgent(llm_provider=FakeLLMProvider(default_text=resp_insuf))
    verifs_insuf = agent_insuf.run(_make_report([claim_sup]), chunks)
    assert verifs_insuf[0].status == ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE

    # 5. UNVERIFIABLE (via fabricated ID)
    claim_unverif = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Unknown observation.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:nonexistent:0"],
    )
    verifs_unverif = agent_sup.run(_make_report([claim_unverif]), chunks)
    assert verifs_unverif[0].status == ClaimVerificationStatusEnum.UNVERIFIABLE


def test_contradicting_passages_listed():
    """Verify that opposing statements in other retrieved passages are listed in contradicting_evidence_ids."""
    cited_chunk = _make_chunk(
        "ev:pubmed:4444:0",
        "[SYNTHETIC] Azithromycin was considered as possible management strategy.",
    )
    opposing_chunk = _make_chunk(
        "ev:pubmed:5555:0",
        "[SYNTHETIC] Azithromycin is ineffective and not recommended for this condition.",
    )

    resp_sup = json.dumps({
        "status": "SUPPORTED",
        "justification": "Considered strategy.",
        "passage_index": 0,
        "span_start": 12,
        "span_end": 50,
    })
    agent = ClaimVerificationAgent(llm_provider=FakeLLMProvider(default_text=resp_sup))

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Azithromycin was considered for therapy.",
        section="potential_considerations",
        cited_evidence_ids=["ev:pubmed:4444:0"],
    )
    report = _make_report([claim])
    all_chunks = [cited_chunk, opposing_chunk]

    verifs = agent.run(report=report, evidence_passages=all_chunks)

    assert isinstance(verifs, list)
    v = verifs[0]
    assert "ev:pubmed:5555:0" in v.contradicting_evidence_ids


def test_verifier_bounded_retry_then_unverifiable():
    """Verifier failure gets one bounded retry, then returns UNVERIFIABLE with VERIFIER_ERROR."""
    class FailingFakeProvider(FakeLLMProvider):
        def generate_text(self, prompt: str, **kwargs):
            raise RuntimeError("API connection timeout")

    agent = ClaimVerificationAgent(llm_provider=FailingFakeProvider())
    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Serum lactate was elevated.",
        section="evidence_supported_observations",
        cited_evidence_ids=["ev:pubmed:6666:0"],
    )
    report = _make_report([claim])
    chunks = [_make_chunk("ev:pubmed:6666:0", "[SYNTHETIC] Lactate measurements.")]

    verifs = agent.run(report=report, evidence_passages=chunks)

    assert isinstance(verifs, list)
    v = verifs[0]
    assert v.status == ClaimVerificationStatusEnum.UNVERIFIABLE
    assert v.limitations == "VERIFIER_ERROR"
