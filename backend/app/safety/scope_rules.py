"""Deterministic Clinical Scope Checker.

Specification Section 1, Section 3 (Rule 1), Section 9.8 (Req ID: SAF-1, EXT-B).
[ENGINEERING-DECISION: SCOPE-REGEX]
Rule 1 of evidence_safety_gate requires abstention on out-of-scope requests:
treatment, prescription, dosing, or direct clinical orders.
This module implements deterministic regex-based detection of clinical action requests.
NOTE: As documented in docs/known_gaps.md, this module detects treatment/dosing requests,
but does not attempt broad non-biomedical topic classification.
In benchmark mode (SPEC 9.8), biomedical factual questions are counted as in-scope.
"""

import re
from typing import Optional, Tuple

# Treatment recommendation requests
_TREATMENT_PATTERNS = [
    r"\b(?:how\s+to\s+treat|treatment\s+(?:plan|regimen|option|order)s?|prescribe|prescriptions?)\b",
    r"\b(?:recommend\s+(?:medication|drug|therapy|treatment)|cure\s+for|what\s+should\s+i\s+take)\b",
    r"\b(?:start\s+on\s+[a-z]+|management\s+strategy\s+for)\b",
]

# Prescription, dosing, titration requests
_DOSING_PATTERNS = [
    r"\b(?:dosage|dosing|dose\s+calculation|how\s+much\s+[a-z]+\s+to\s+take)\b",
    r"\b(?:\d+\s*(?:mg|mcg|g|ml|units?)\s*(?:daily|bid|tid|qid|prn|po|iv))\b",
    r"\b(?:titrat(?:e|ion)\s+to|calculate\s+dose|pediatric\s+dosing)\b",
]

# Direct clinical action orders
_ORDER_PATTERNS = [
    r"\b(?:order\s+(?:iv|infusion|bolus|injection)|administer\s+\d+)\b",
    r"\b(?:give\s+(?:patient\s+)?\d+\s*(?:mg|ml))\b",
]

_ALL_SCOPE_REGEXES = [
    re.compile(p, re.IGNORECASE)
    for p in (_TREATMENT_PATTERNS + _DOSING_PATTERNS + _ORDER_PATTERNS)
]


def check_clinical_scope(
    text: Optional[str],
    is_benchmark: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Determine whether a case text is in scope for clinical decision support.

    Returns:
        (is_in_scope, violation_reason)
        is_in_scope: True if request is valid diagnostic decision support;
                     False if request seeks treatment, prescription, or dosing.
    """
    if is_benchmark:
        # SPEC 9.8: In benchmark mode a biomedical factual question is in scope
        return True, None

    if not text or not text.strip():
        return True, None

    normalized_text = text.strip()

    for pattern in _ALL_SCOPE_REGEXES:
        match = pattern.search(normalized_text)
        if match:
            matched_phrase = match.group(0)
            return False, f"Out-of-scope request detected: matches prohibited treatment/dosing pattern '{matched_phrase}'"

    return True, None
