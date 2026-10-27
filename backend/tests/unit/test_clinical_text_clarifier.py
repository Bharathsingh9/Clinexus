"""Unit tests for Clinical Text Clarifier (Agent 1 - AGT-1).

Specification Section 4.1, Section 11:
- negations kept as polarity=ABSENT
- unknown never turned into absent
- source spans valid and verified against original text
- terminology mapped via symptom_vocab.json; unmapped keeps normalized_term=null
- required fields missing raise blocking clarification
- contradiction detected by rules raises blocking clarification
- at most 5 clarification questions
- invalid LLM JSON raises FAILED(INVALID_LLM_OUTPUT) after repair retry
- without LLM provider, returns FAILED with error_code LLM_UNAVAILABLE
"""

import json
import pytest
from backend.app.agents.clinical_text_clarifier import ClinicalTextClarifier
from backend.app.schemas.clinical_case import (
    CaseSubmission,
    PolarityEnum,
)
from backend.app.schemas.executions import AgentStatusEnum
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


def test_llm_unavailable_without_provider():
    """Without an LLM provider, Agent 1 must return FAILED with LLM_UNAVAILABLE."""
    clarifier = ClinicalTextClarifier(llm_provider=None)
    submission = CaseSubmission(raw_text="35yo female with headache for 1 day.")
    result = clarifier.run(submission)

    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == "LLM_UNAVAILABLE"
    assert "LLM provider is not configured" in (result.error_message or "")


def test_negations_kept_as_absent():
    text = "45yo male presents with fever for 3 days. Patient denies headache and has no nausea."
    llm_payload = {
        "age": 45,
        "sex": "male",
        "chief_complaint": "fever",
        "symptom_duration": "3 days",
        "symptoms": [
            {"term": "fever", "polarity": "PRESENT", "source_span": {"start": 24, "end": 29}, "origin": "STATED"},
            {"term": "headache", "polarity": "ABSENT", "source_span": {"start": 57, "end": 65}, "origin": "STATED"},
            {"term": "nausea", "polarity": "ABSENT", "source_span": {"start": 77, "end": 83}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    terms = {s.term.lower(): s.polarity for s in case.structured_symptoms}
    # Present
    assert terms.get("fever") == PolarityEnum.PRESENT
    # Negations kept as ABSENT, never turned into UNKNOWN or omitted
    assert terms.get("headache") == PolarityEnum.ABSENT
    assert terms.get("nausea") == PolarityEnum.ABSENT


def test_unknown_never_turned_into_absent():
    text = "Patient presents with persistent cough."
    # Age and duration missing
    llm_payload = {
        "age": None,
        "sex": "UNKNOWN",
        "chief_complaint": "cough",
        "symptom_duration": None,
        "symptoms": [
            {"term": "cough", "polarity": "PRESENT", "source_span": {"start": 33, "end": 38}, "origin": "STATED"}
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    assert case.age is None
    assert case.age_missing_reason is not None
    assert case.symptom_duration is None
    assert case.duration_missing_reason is not None
    assert case.sex == "UNKNOWN"

    # Verify that unknown fields trigger blocking clarifications, not ABSENT
    blocking_questions = [q for q in case.clarification_questions if q.is_blocking]
    assert len(blocking_questions) >= 3


def test_source_span_validation():
    text = "A 60yo female presents with chest pain and dyspnea for 2 days."
    llm_payload = {
        "age": 60,
        "sex": "female",
        "chief_complaint": "chest pain",
        "symptom_duration": "2 days",
        "symptoms": [
            {"term": "chest pain", "polarity": "PRESENT", "source_span": {"start": 28, "end": 38}, "origin": "STATED"},
            {"term": "dyspnea", "polarity": "PRESENT", "source_span": {"start": 43, "end": 50}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    for s in case.structured_symptoms:
        start = s.source_span.start
        end = s.source_span.end
        assert 0 <= start < end <= len(text)
        sub = text[start:end]
        assert s.term.lower() in sub.lower() or sub.lower() in s.term.lower()


def test_symptom_vocab_normalization():
    text = "A 30yo man with shortness of breath and high fever for 1 week."
    llm_payload = {
        "age": 30,
        "sex": "male",
        "chief_complaint": "shortness of breath",
        "symptom_duration": "1 week",
        "symptoms": [
            {"term": "shortness of breath", "polarity": "PRESENT", "source_span": {"start": 16, "end": 35}, "origin": "STATED"},
            {"term": "high fever", "polarity": "PRESENT", "source_span": {"start": 40, "end": 50}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    norm_map = {s.term.lower(): s.normalized_term for s in case.structured_symptoms}
    # Check normalized mappings
    if "shortness of breath" in norm_map:
        assert norm_map["shortness of breath"] == "dyspnea"
    if "high fever" in norm_map:
        assert norm_map["high fever"] == "pyrexia"


def test_contradiction_detection_raises_blocking_clarification():
    text = "55yo male with fever for 4 days. Patient denies fever on admission."
    llm_payload = {
        "age": 55,
        "sex": "male",
        "chief_complaint": "fever",
        "symptom_duration": "4 days",
        "symptoms": [
            {"term": "fever", "polarity": "PRESENT", "source_span": {"start": 15, "end": 20}, "origin": "STATED"},
            {"term": "fever", "polarity": "ABSENT", "source_span": {"start": 48, "end": 53}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    blocking = [q for q in case.clarification_questions if q.is_blocking and "clarify" in q.question_text.lower()]
    assert len(blocking) >= 1
    assert "fever" in blocking[0].question_text.lower()


def test_clarification_questions_cap():
    text = "Vague fatigue."
    # Narrative missing everything
    llm_payload = {
        "age": None,
        "sex": "UNKNOWN",
        "chief_complaint": "",
        "symptom_duration": None,
        "symptoms": [
            {"term": "fatigue", "polarity": "PRESENT", "source_span": {"start": 6, "end": 13}, "origin": "STATED"},
            {"term": "weakness", "polarity": "PRESENT", "source_span": {"start": 6, "end": 13}, "origin": "STATED"},
            {"term": "malaise", "polarity": "PRESENT", "source_span": {"start": 6, "end": 13}, "origin": "STATED"},
            {"term": "lethargy", "polarity": "PRESENT", "source_span": {"start": 6, "end": 13}, "origin": "STATED"},
        ],
        "patient_context": {"history": [], "medications": [], "allergies": [], "vitals": {}},
    }
    clarifier = ClinicalTextClarifier(llm_provider=FakeLLMProvider(default_text=json.dumps(llm_payload)))
    submission = CaseSubmission(raw_text=text)
    case = clarifier.run(submission)

    assert len(case.clarification_questions) <= 5


def test_invalid_llm_json_repair_failure_raises_value_error():
    # Configure FakeLLMProvider to return invalid non-JSON string
    bad_provider = FakeLLMProvider(default_text="NOT A JSON STRING {{{")
    clarifier = ClinicalTextClarifier(llm_provider=bad_provider)

    submission = CaseSubmission(raw_text="35yo female with headache for 1 day.")
    with pytest.raises(ValueError, match="FAILED: INVALID_LLM_OUTPUT"):
        clarifier.run(submission)
