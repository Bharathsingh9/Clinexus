"""Unit Tests for Urgent-Symptom Pattern Detection and Alerting (SAF-3).

Specification Section 8:
- Documented pattern set:
  1. chest pain with dyspnea
  2. sudden focal neurological deficit
  3. suicidal ideation statements
  4. severe bleeding
  5. anaphylaxis signs
- Non-urgent text does not match.
- When matched, returns banner text:
  "Possible urgent presentation: do not delay appropriate urgent human assessment"
- Writes an immutable AuditEvent.
- Matches never block or delay workflow analysis.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import Base
from backend.app.safety.urgent import (
    URGENT_BANNER_TEXT,
    evaluate_urgent_rules,
    load_urgent_rules,
)
from backend.app.schemas.clinical_case import StructuredSymptom


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for audit events."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_load_urgent_rules_contains_five_patterns():
    """Verify urgent ruleset loads all 5 required pattern definitions."""
    rules = load_urgent_rules()
    assert len(rules) == 5

    names = [r.name for r in rules]
    assert "chest_pain_with_dyspnea" in names
    assert "sudden_focal_neurological_deficit" in names
    assert "suicidal_ideation_statements" in names
    assert "severe_bleeding" in names
    assert "anaphylaxis_signs" in names


@pytest.mark.parametrize(
    ("pattern_name", "synthetic_phrase"),
    [
        ("chest_pain_with_dyspnea", "Patient reports sudden onset of retrosternal pain and severe shortness of breath."),
        ("sudden_focal_neurological_deficit", "Family noticed sudden facial droop and right arm weakness 1 hour ago."),
        ("suicidal_ideation_statements", "Patient explicitly stated thoughts of suicide and wanting to end my life."),
        ("severe_bleeding", "Emergency intake for massive hemorrhage and vomiting blood."),
        ("anaphylaxis_signs", "Patient developed acute anaphylaxis with severe swelling of lips and tongue after peanut exposure."),
    ],
)
def test_urgent_patterns_match_synthetic_phrases(pattern_name, synthetic_phrase, sqlite_session):
    """Verify each documented pattern matches its representative synthetic clinical phrase."""
    audit_repo = AuditRepository(sqlite_session)

    res = evaluate_urgent_rules(
        raw_text=synthetic_phrase,
        case_id="case-urg-1",
        audit_repo=audit_repo,
    )

    assert res.is_urgent is True
    assert pattern_name in res.matched_rule_names
    assert res.banner_text == URGENT_BANNER_TEXT

    # Verify audit event written
    events = audit_repo.list_by_resource("clinical_case", "case-urg-1")
    assert len(events) >= 1
    assert events[-1].event_type == "URGENT_SYMPTOM_MATCHED"
    assert URGENT_BANNER_TEXT in events[-1].payload


def test_urgent_patterns_match_structured_findings(sqlite_session):
    """Verify matching operates on structured findings terms as well as narrative text."""
    audit_repo = AuditRepository(sqlite_session)
    finding1 = StructuredSymptom(
        term="chest tightness",
        polarity="PRESENT",
        source_span={"start": 0, "end": 15},
        origin="STATED",
    )
    finding2 = StructuredSymptom(
        term="dyspnea on exertion",
        polarity="PRESENT",
        source_span={"start": 16, "end": 35},
        origin="STATED",
    )

    res = evaluate_urgent_rules(
        raw_text="Routine clinical review of vitals.",
        findings=[finding1, finding2],
        case_id="case-urg-findings",
        audit_repo=audit_repo,
    )

    assert res.is_urgent is True
    assert "chest_pain_with_dyspnea" in res.matched_rule_names
    assert res.banner_text == URGENT_BANNER_TEXT


def test_non_urgent_text_does_not_trigger_alert(sqlite_session):
    """Verify routine, non-urgent clinical text does not trigger urgent banner or audit event."""
    audit_repo = AuditRepository(sqlite_session)
    text = "Patient presents for annual physical exam. Reports mild seasonal allergies and intermittent itchy eyes."

    res = evaluate_urgent_rules(
        raw_text=text,
        case_id="case-routine-1",
        audit_repo=audit_repo,
    )

    assert res.is_urgent is False
    assert res.banner_text is None
    assert len(res.matched_rule_ids) == 0

    events = audit_repo.list_by_resource("clinical_case", "case-routine-1")
    assert len(events) == 0
