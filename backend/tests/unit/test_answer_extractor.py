"""Unit Tests for AnswerExtractor Agent (Phase 5 Stage C).

Specification Section 9.9, Appendix A7 (Req ID: EVL-6 / AGT-6):
- Tests use FakeLLMProvider and synthetic clinical/biomedical questions only; no real BioASQ text.
- Validates:
    * Each question type's valid output parses (yesno, factoid, list, summary).
    * Invalid shapes trigger one retry then FAILED with error_code='LLM_INVALID_RESPONSE'.
    * Successful repair on attempt 2 succeeds.
    * 'ABSTAIN' from LLM is carried through cleanly.
    * Upstream workflow abstention or empty supported claims makes ZERO LLM calls (verified via ExplodingLLMProvider).
    * Extractor receives ONLY SUPPORTED/PARTIALLY_SUPPORTED claims (rejection of CONTRADICTED/UNVERIFIABLE).
    * Provider prompt payload never contains sample_ids, gold data, or ungrounded snippets.
    * Typed LLM errors map to FAILED execution records with corresponding error codes.
    * extractor_version and model_name are recorded on every output.
"""

import json
from typing import Any, List, Optional
import pytest
from pydantic import ValidationError

from backend.app.agents.answer_extractor import (
    EXTRACTOR_VERSION,
    AnswerExtractionResult,
    AnswerExtractor,
    AnswerExtractorInput,
    ExtractedReportClaim,
    validate_answer_shape,
)
from backend.app.core.llm_exceptions import (
    LLMAuthError,
    LLMInvalidResponse,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from backend.app.schemas.executions import AgentStatusEnum
from backend.app.schemas.reports import ClaimVerificationStatusEnum


class ExplodingLLMProvider:
    """Mock provider that strictly raises if any completion method is invoked."""

    async def generate_text(self, *args, **kwargs) -> str:
        raise AssertionError("ExplodingLLMProvider was unexpectedly called!")

    async def generate_json(self, *args, **kwargs):
        raise AssertionError("ExplodingLLMProvider was unexpectedly called!")


class ScriptedLLMProvider:
    """Mock provider returning scripted sequence of string completions."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.recorded_calls: List[dict] = []

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        self.recorded_calls.append({
            "prompt": prompt,
            "system": system_instruction,
            "model": model,
        })
        if not self.responses:
            raise RuntimeError("ScriptedLLMProvider has no remaining responses configured.")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp if isinstance(resp, str) else json.dumps(resp)


def test_extracted_report_claim_enforces_supported_statuses():
    """Verify ExtractedReportClaim only allows SUPPORTED or PARTIALLY_SUPPORTED."""
    # Valid
    c1 = ExtractedReportClaim(
        claim_text="[SYNTHETIC] Target protein kinase is inhibited by compound X.",
        status=ClaimVerificationStatusEnum.SUPPORTED,
    )
    assert c1.status == ClaimVerificationStatusEnum.SUPPORTED

    c2 = ExtractedReportClaim(
        claim_text="[SYNTHETIC] Moderate symptom reduction noted.",
        status=ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
    )
    assert c2.status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED

    # Invalid statuses must raise ValidationError
    with pytest.raises(ValidationError):
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Invalid claim",
            status=ClaimVerificationStatusEnum.CONTRADICTED,
        )

    with pytest.raises(ValidationError):
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Invalid claim",
            status=ClaimVerificationStatusEnum.UNVERIFIABLE,
        )

    with pytest.raises(ValidationError):
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Invalid claim",
            status=ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE,
        )


def test_answer_extractor_input_forbids_extra_fields():
    """Verify AnswerExtractorInput rejects forbidden metadata (sample_id, gold data)."""
    # Reject sample_id
    with pytest.raises(ValidationError):
        AnswerExtractorInput(
            question_type="yesno",
            question="[SYNTHETIC] Is drug X safe?",
            sample_id="SAMPLE_001",  # Extra forbidden!
        )

    # Reject gold_answers
    with pytest.raises(ValidationError):
        AnswerExtractorInput(
            question_type="factoid",
            question="[SYNTHETIC] Target gene?",
            gold_answer="BRCA1",  # Extra forbidden!
        )


def test_validate_answer_shape_pure_rules():
    """Test pure answer shape validation and normalization across all types."""
    # yesno
    assert validate_answer_shape("yesno", "yes") == "yes"
    assert validate_answer_shape("yesno", "YES") == "yes"
    assert validate_answer_shape("yesno", "no") == "no"
    assert validate_answer_shape("yesno", "ABSTAIN") == "ABSTAIN"
    with pytest.raises(ValueError):
        validate_answer_shape("yesno", "maybe")
    with pytest.raises(ValueError):
        validate_answer_shape("yesno", ["yes"])

    # factoid
    assert validate_answer_shape("factoid", ["Entity1", "Entity2"]) == ["Entity1", "Entity2"]
    assert validate_answer_shape("factoid", "SingleEntity") == ["SingleEntity"]
    assert validate_answer_shape("factoid", "ABSTAIN") == "ABSTAIN"
    assert validate_answer_shape("factoid", ["e1", "e2", "e3", "e4", "e5", "e6"]) == ["e1", "e2", "e3", "e4", "e5"]
    with pytest.raises(ValueError):
        validate_answer_shape("factoid", [])

    # list
    assert validate_answer_shape("list", ["Entity1", "Entity2"]) == ["Entity1", "Entity2"]
    assert validate_answer_shape("list", "ABSTAIN") == "ABSTAIN"
    with pytest.raises(ValueError):
        validate_answer_shape("list", [])
    with pytest.raises(ValueError):
        validate_answer_shape("list", 123)

    # summary
    assert validate_answer_shape("summary", "Concise summary.") == "Concise summary."
    assert validate_answer_shape("summary", "ABSTAIN") == "ABSTAIN"
    with pytest.raises(ValueError):
        validate_answer_shape("summary", "")
    with pytest.raises(ValueError):
        validate_answer_shape("summary", ["summary"])


def test_extractor_valid_outputs_for_each_type():
    """Verify AnswerExtractor parses valid JSON outputs for each question type."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Compound A is a selective antagonist of receptor B.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]

    # 1. yesno
    prov_yesno = ScriptedLLMProvider([{"answer": "yes"}])
    ext_yesno = AnswerExtractor(llm_provider=prov_yesno, model_name="test-model")
    res_yesno = ext_yesno.extract(
        AnswerExtractorInput(question_type="yesno", question="Is compound A an antagonist?", claims=claims)
    )
    assert res_yesno.extracted_answer == "yes"
    assert res_yesno.extractor_version == EXTRACTOR_VERSION
    assert res_yesno.model_name == "test-model"
    assert res_yesno.execution_record.status == AgentStatusEnum.COMPLETED

    # 2. factoid
    prov_factoid = ScriptedLLMProvider([{"answer": ["Receptor B", "Receptor B1"]}])
    ext_factoid = AnswerExtractor(llm_provider=prov_factoid, model_name="test-model")
    res_factoid = ext_factoid.extract(
        AnswerExtractorInput(question_type="factoid", question="Which receptor is targeted?", claims=claims)
    )
    assert res_factoid.extracted_answer == ["Receptor B", "Receptor B1"]
    assert res_factoid.execution_record.status == AgentStatusEnum.COMPLETED

    # 3. list
    prov_list = ScriptedLLMProvider([{"answer": ["Enzyme X", "Enzyme Y"]}])
    ext_list = AnswerExtractor(llm_provider=prov_list, model_name="test-model")
    res_list = ext_list.extract(
        AnswerExtractorInput(question_type="list", question="List involved enzymes?", claims=claims)
    )
    assert res_list.extracted_answer == ["Enzyme X", "Enzyme Y"]

    # 4. summary
    prov_summary = ScriptedLLMProvider([{"answer": "Compound A selectively inhibits receptor B."}])
    ext_summary = AnswerExtractor(llm_provider=prov_summary, model_name="test-model")
    res_summary = ext_summary.extract(
        AnswerExtractorInput(question_type="summary", question="Summarize findings.", claims=claims)
    )
    assert res_summary.extracted_answer == "Compound A selectively inhibits receptor B."


def test_extractor_invalid_shape_retried_once_then_fails():
    """Verify invalid answer shape is retried once, then returns FAILED(LLM_INVALID_RESPONSE)."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Verified observation.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]

    # Both attempt 1 and repair attempt 2 return invalid yesno answers (not yes/no/ABSTAIN)
    provider = ScriptedLLMProvider([
        {"answer": "uncertain"},   # Attempt 1
        {"answer": "undecided"},   # Attempt 2 (repair retry)
    ])
    extractor = AnswerExtractor(llm_provider=provider, model_name="test-model")

    res = extractor.extract(
        AnswerExtractorInput(question_type="yesno", question="Is this verified?", claims=claims)
    )

    assert len(provider.recorded_calls) == 2  # Proves exactly one repair retry
    assert res.extracted_answer == "ABSTAIN"
    assert res.execution_record is not None
    assert res.execution_record.status == AgentStatusEnum.FAILED
    assert res.execution_record.error_code == "LLM_INVALID_RESPONSE"


def test_extractor_repair_retry_succeeds_on_second_attempt():
    """Verify that if attempt 1 is malformed JSON, repair retry on attempt 2 can succeed."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Confirmed treatment effect.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]

    provider = ScriptedLLMProvider([
        "NOT_JSON_AT_ALL",           # Attempt 1 (bad JSON)
        {"answer": "yes"},           # Attempt 2 (repair succeeds)
    ])
    extractor = AnswerExtractor(llm_provider=provider, model_name="test-model")

    res = extractor.extract(
        AnswerExtractorInput(question_type="yesno", question="Was treatment effective?", claims=claims)
    )

    assert len(provider.recorded_calls) == 2
    assert res.extracted_answer == "yes"
    assert res.execution_record.status == AgentStatusEnum.COMPLETED


def test_extractor_abstain_response_from_llm():
    """Verify that an explicit 'ABSTAIN' from the LLM is preserved."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Inconclusive peripheral finding.",
            status=ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
        )
    ]
    provider = ScriptedLLMProvider([{"answer": "ABSTAIN"}])
    extractor = AnswerExtractor(llm_provider=provider)

    res = extractor.extract(
        AnswerExtractorInput(question_type="factoid", question="What is the active dosage?", claims=claims)
    )
    assert res.extracted_answer == "ABSTAIN"
    assert res.execution_record.status == AgentStatusEnum.COMPLETED


def test_extractor_workflow_abstained_makes_zero_llm_calls():
    """Verify that when workflow_abstained is True, extractor makes zero LLM calls."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Claim from abstained workflow.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]
    exploding_provider = ExplodingLLMProvider()
    extractor = AnswerExtractor(llm_provider=exploding_provider)

    inp = AnswerExtractorInput(
        question_type="yesno",
        question="[SYNTHETIC] Clinical question",
        claims=claims,
        workflow_abstained=True,
    )

    res = extractor.extract(inp)
    assert res.workflow_abstained is True
    assert res.extracted_answer == "ABSTAIN"
    assert res.execution_record is None  # Extractor was not invoked


def test_extractor_empty_claims_makes_zero_llm_calls():
    """Verify that when no SUPPORTED/PARTIALLY_SUPPORTED claims exist, extractor makes zero LLM calls."""
    exploding_provider = ExplodingLLMProvider()
    extractor = AnswerExtractor(llm_provider=exploding_provider)

    inp = AnswerExtractorInput(
        question_type="yesno",
        question="[SYNTHETIC] Question with zero supported claims",
        claims=[],
        workflow_abstained=False,
    )

    res = extractor.extract(inp)
    assert res.workflow_abstained is True
    assert res.extracted_answer == "ABSTAIN"
    assert res.execution_record is None


def test_extractor_prompt_contains_no_gold_data_or_sample_ids():
    """Assert the exact payload sent to the LLM provider contains no gold answers or sample_ids."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Silybin demonstrates antioxidant efficacy in vitro.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]
    provider = ScriptedLLMProvider([{"answer": "yes"}])
    extractor = AnswerExtractor(llm_provider=provider)

    extractor.extract(
        AnswerExtractorInput(
            question_type="yesno",
            question="[SYNTHETIC] Does silybin show antioxidant activity?",
            claims=claims,
        )
    )

    assert len(provider.recorded_calls) == 1
    call_prompt = provider.recorded_calls[0]["prompt"]
    call_system = provider.recorded_calls[0]["system"]
    combined = f"{call_system}\n{call_prompt}"

    # Verify absence of gold terms / identifiers
    forbidden_tokens = ["sample_id", "gold", "exact_answer", "ideal_answer", "evaluation_labels", "SYNTH_GOLD_999"]
    for token in forbidden_tokens:
        assert token not in combined.lower()


def test_extractor_typed_llm_errors_mapped_to_failed_record():
    """Verify typed LLM exceptions return FAILED execution records with exact error codes."""
    claims = [
        ExtractedReportClaim(
            claim_text="[SYNTHETIC] Target claim.",
            status=ClaimVerificationStatusEnum.SUPPORTED,
        )
    ]

    error_cases = [
        (LLMTimeout("Request timed out"), "LLM_TIMEOUT"),
        (LLMRateLimited("Rate limit hit"), "LLM_RATE_LIMITED"),
        (LLMAuthError("Unauthorized key"), "LLM_AUTH_ERROR"),
        (LLMUnavailable("Service unavailable"), "LLM_UNAVAILABLE"),
    ]

    for exc, expected_code in error_cases:
        provider = ScriptedLLMProvider([exc])
        extractor = AnswerExtractor(llm_provider=provider)
        res = extractor.extract(
            AnswerExtractorInput(question_type="yesno", question="Question?", claims=claims)
        )

        assert res.extracted_answer == "ABSTAIN"
        assert res.execution_record is not None
        assert res.execution_record.status == AgentStatusEnum.FAILED
        assert res.execution_record.error_code == expected_code
        assert res.extractor_version == EXTRACTOR_VERSION
