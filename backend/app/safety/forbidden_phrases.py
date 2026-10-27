"""Forbidden Phrases Linter.

Specification Section 4.4, Section 8, Appendix A8 (Req ID: SAF-5):
- Scans report text for forbidden clinical language:
  * Imperative treatment instructions
  * Prescription directives
  * Specific medication dosing orders
  * Definitive, forced autonomous diagnoses
- Pure deterministic regex linter.
"""

import re
from typing import List, Tuple

FORBIDDEN_RULES = [
    (
        "IMPERATIVE_DOSING",
        re.compile(
            r"\b(take|administer|inject|infuse|give|prescribe)\s+\d+[\d\.]*\s*(mg|g|mcg|ml|tablets?|capsules?|units?|iu|drops?)\b",
            re.IGNORECASE,
        ),
        "Contains direct dosing command or specific dosage quantity instruction",
    ),
    (
        "PRESCRIPTION_DIRECTIVE",
        re.compile(
            r"\b(prescribe|prescription for|order prescription|prescribing)\s+[a-zA-Z]{3,}\b",
            re.IGNORECASE,
        ),
        "Contains autonomous prescription command",
    ),
    (
        "TREATMENT_MANDATE",
        re.compile(
            r"\b(patient (must|should) be prescribed|patient (must|should) take|initiate treatment with|start (the )?patient on|titrate (dose )?to)\b",
            re.IGNORECASE,
        ),
        "Contains mandatory treatment directive",
    ),
    (
        "FORCED_SINGLE_DIAGNOSIS",
        re.compile(
            r"\b(definitively diagnosed with|patient definitively has|diagnosed with absolute certainty|conclusive diagnosis is)\b",
            re.IGNORECASE,
        ),
        "Contains definitive autonomous diagnosis instead of candidate differential considerations",
    ),
]


def lint_forbidden_phrases(text: str) -> List[Tuple[str, str, str]]:
    """Scan text against forbidden safety rules.

    Returns list of tuples: (rule_name, matched_substring, description).
    """
    violations: List[Tuple[str, str, str]] = []
    if not text:
        return violations

    for rule_name, pattern, description in FORBIDDEN_RULES:
        for match in pattern.finditer(text):
            violations.append((rule_name, match.group(0), description))

    return violations


def assert_no_forbidden_phrases(text: str) -> None:
    """Raise ValueError if any forbidden language rule is violated."""
    violations = lint_forbidden_phrases(text)
    if violations:
        details = "; ".join([f"[{rule}]: '{matched}' ({desc})" for rule, matched, desc in violations])
        raise ValueError(f"SAFETY VIOLATION (Forbidden language detected): {details}")
