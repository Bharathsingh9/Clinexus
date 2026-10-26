"""Unit Tests for Pydantic v2 Data Contracts.

Specification Section 6, Appendix A5.
Validates:
- extra="forbid" rejects unexpected fields
- Enum value enforcement
- Required fields enforcement
- UNKNOWN / null reason validation
- CandidateHypothesis contract validation
"""

import pytest
from pydantic import ValidationError
from backend.app.schemas.clinical_case import (
    CandidateHypothesis,
    ClarificationRequest,
    ClarificationStatusEnum,
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
    EvidenceSourceEnum,
    RetrievalQuery,
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)
from backend.app.schemas.reports import (
    AbstentionDecision,
    AbstentionReasonCode,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
    VerificationMethodEnum,
)


def test_extra_fields_forbidden():
    """Verify that models strictly forbid unknown attributes (extra='forbid')."""
    with pytest.raises(ValidationError):
        SourceSpan(start=0, end=10, unexpected_field="disallowed")

    with pytest.raises(ValidationError):
        PatientContext(history=["Asthma"], unknown_attribute=123)

    with pytest.raises(ValidationError):
        RetrievalQuery(query_text="fever", unknown="illegal")


def test_source_span_validation():
    """Verify source span start and end offsets."""
    span = SourceSpan(start=5, end=15)
    assert span.start == 5
    assert span.end == 15

    with pytest.raises(ValidationError, match="Span end offset must be >= start offset"):
        SourceSpan(start=20, end=10)


def test_structured_symptom_inferred_requires_note():
    """Verify that INFERRED symptoms must provide an uncertainty_note."""
    span = SourceSpan(start=0, end=5)
    # Stated without note is valid
    sym_stated = StructuredSymptom(
        term="cough",
        polarity=PolarityEnum.PRESENT,
        source_span=span,
        origin=OriginEnum.STATED,
    )
    assert sym_stated.origin == OriginEnum.STATED

    # Inferred without note must fail
    with pytest.raises(ValidationError, match="Inferred findings must carry an uncertainty_note"):
        StructuredSymptom(
            term="fatigue",
            polarity=PolarityEnum.PRESENT,
            source_span=span,
            origin=OriginEnum.INFERRED,
            uncertainty_note=None,
        )


def test_clinical_case_missing_data_reasons():
    """Verify that null age or duration requires an explicit reason."""
    dummy_hash = "a" * 64
    # Valid case with age and duration
    case = ClinicalCase(
        raw_text_hash=dummy_hash,
        age=45,
        chief_complaint="Chest tightness",
        symptom_duration="3 days",
    )
    assert case.age == 45

    # Age is None without reason must fail
    with pytest.raises(ValidationError, match="When age is null, age_missing_reason must be provided"):
        ClinicalCase(
            raw_text_hash=dummy_hash,
            age=None,
            age_missing_reason=None,
            chief_complaint="Chest tightness",
            symptom_duration="3 days",
        )

    # Valid with explicit missing reason
    case_with_reason = ClinicalCase(
        raw_text_hash=dummy_hash,
        age=None,
        age_missing_reason="Patient refused to provide age",
        chief_complaint="Headache",
        symptom_duration="1 day",
    )
    assert case_with_reason.age is None
    assert case_with_reason.age_missing_reason == "Patient refused to provide age"


def test_candidate_hypothesis_validation():
    """Verify CandidateHypothesis scores count must match supporting IDs count."""
    hyp = CandidateHypothesis(
        label="Community-Acquired Pneumonia",
        supporting_evidence_ids=["ev:pubmed:1111:0", "ev:pubmed:2222:1"],
        retrieval_scores=[0.85, 0.78],
    )
    assert hyp.label == "Community-Acquired Pneumonia"
    assert len(hyp.supporting_evidence_ids) == 2

    with pytest.raises(ValidationError, match="retrieval_scores count must match supporting_evidence_ids count"):
        CandidateHypothesis(
            label="Incomplete scores",
            supporting_evidence_ids=["ev:pubmed:1111:0", "ev:pubmed:2222:1"],
            retrieval_scores=[0.85],
        )


def test_evidence_chunk_validates_evidence_id():
    """Verify EvidenceChunk rejects invalid platform evidence IDs."""
    with pytest.raises(ValidationError, match="Invalid platform evidence ID format"):
        EvidenceChunk(
            document_id="doc-123",
            evidence_id="invalid_format",
            chunk_index=0,
            text="Passage text",
            text_hash="b" * 64,
        )


def test_abstention_decision_validation():
    """Verify that abstained=True requires at least one reason code."""
    with pytest.raises(ValidationError, match="When abstained is True, at least one reason_code must be provided"):
        AbstentionDecision(
            abstained=True,
            reason_codes=[],
            explanation="Abstaining without code",
        )

    decision = AbstentionDecision(
        abstained=True,
        reason_codes=[AbstentionReasonCode.NO_RELEVANT_EVIDENCE],
        explanation="No relevant literature found",
    )
    assert decision.abstained is True
    assert decision.reason_codes == [AbstentionReasonCode.NO_RELEVANT_EVIDENCE]


def test_enum_statuses_enforced():
    """Verify invalid enum strings raise validation error."""
    with pytest.raises(ValidationError):
        ClarificationRequest(
            question_text="Clarify onset?",
            is_blocking=True,
            reason="Unspecified",
            status="INVALID_STATUS",
        )

    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            claim_text="Sample claim",
            category="Finding",
            status="NOT_A_VALID_STATUS",
            justification="Because",
            method=VerificationMethodEnum.DETERMINISTIC,
        )
