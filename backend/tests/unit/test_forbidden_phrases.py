"""Unit tests for Forbidden Phrases Linter (Req ID: SAF-5, App A8).

Specification Section 4.4, Section 8, Appendix A8:
- Imperative dosing commands detected and failed.
- Prescription directives detected and failed.
- Mandatory treatment directives detected and failed.
- Forced single diagnoses detected and failed.
- Safe differential diagnostic language passes cleanly.
"""

import pytest
from backend.app.safety.forbidden_phrases import assert_no_forbidden_phrases, lint_forbidden_phrases


def test_clean_differential_language_passes():
    text = (
        "Considerations include viral upper respiratory tract infection and acute bronchitis. "
        "Further clinical evaluation is recommended based on presenting symptoms of cough and fever."
    )
    violations = lint_forbidden_phrases(text)
    assert len(violations) == 0
    # Should not raise
    assert_no_forbidden_phrases(text)


def test_imperative_dosing_flagged():
    samples = [
        "Take 500mg amoxicillin three times daily.",
        "Administer 10 ml of oral suspension.",
        "Give 25 mcg levothyroxine daily.",
        "Inject 100 units of subcutaneous heparin.",
    ]
    for s in samples:
        violations = lint_forbidden_phrases(s)
        assert len(violations) >= 1
        assert any(v[0] == "IMPERATIVE_DOSING" for v in violations)
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            assert_no_forbidden_phrases(s)


def test_prescription_directives_flagged():
    samples = [
        "Prescribe azithromycin for the infection.",
        "Write prescription for lisinopril immediately.",
        "Prescribing ampicillin for prophylaxis.",
    ]
    for s in samples:
        violations = lint_forbidden_phrases(s)
        assert len(violations) >= 1
        assert any(v[0] == "PRESCRIPTION_DIRECTIVE" for v in violations)


def test_treatment_mandates_flagged():
    samples = [
        "Patient must be prescribed beta-blockers.",
        "Initiate treatment with corticosteroids without delay.",
        "Start patient on intravenous hydration.",
        "Titrate dose to therapeutic level.",
    ]
    for s in samples:
        violations = lint_forbidden_phrases(s)
        assert len(violations) >= 1
        assert any(v[0] == "TREATMENT_MANDATE" for v in violations)


def test_forced_single_diagnosis_flagged():
    samples = [
        "The patient is definitively diagnosed with pulmonary embolism.",
        "It is confirmed that the patient definitively has acute appendicitis.",
        "The conclusive diagnosis is myocardial infarction.",
    ]
    for s in samples:
        violations = lint_forbidden_phrases(s)
        assert len(violations) >= 1
        assert any(v[0] == "FORCED_SINGLE_DIAGNOSIS" for v in violations)
