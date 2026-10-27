"""Unit tests for Retrieval Query Builder (Req ID: RET-4)."""

from backend.app.retrieval.query_builder import build_retrieval_queries
from backend.app.schemas.clinical_case import (
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)


def make_test_case(present_terms, absent_terms):
    symptoms = []
    for term in present_terms:
        symptoms.append(
            StructuredSymptom(
                term=term,
                normalized_term=term,
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=0, end=len(term)),
                origin=OriginEnum.STATED,
            )
        )
    for term in absent_terms:
        symptoms.append(
            StructuredSymptom(
                term=term,
                normalized_term=term,
                vocabulary="LOCAL",
                polarity=PolarityEnum.ABSENT,
                source_span=SourceSpan(start=0, end=len(term)),
                origin=OriginEnum.STATED,
            )
        )

    return ClinicalCase(
        raw_text_hash="a" * 64,
        age=45,
        sex="female",
        chief_complaint="Chest pain on exertion",
        symptom_duration="3 days",
        structured_symptoms=symptoms,
        patient_context=PatientContext(history=["hypertension"]),
    )


def test_build_queries_includes_present_and_excludes_absent():
    case = make_test_case(present_terms=["fever", "dyspnea"], absent_terms=["nausea", "headache"])
    queries = build_retrieval_queries(case)

    assert 1 <= len(queries) <= 3
    q1_text = queries[0].query_text.lower()
    assert "chest pain" in q1_text
    assert "fever" in q1_text or "dyspnea" in q1_text
    # Absent terms must NOT be in primary query
    assert "nausea" not in q1_text
    assert "headache" not in q1_text


def test_build_queries_with_alternative_hints():
    case = make_test_case(present_terms=["cough"], absent_terms=[])
    hints = ["acute bronchitis differential"]
    queries = build_retrieval_queries(case, alternative_hints=hints)

    assert len(queries) <= 3
    found_hint = any("acute bronchitis differential" in q.query_text for q in queries)
    assert found_hint
