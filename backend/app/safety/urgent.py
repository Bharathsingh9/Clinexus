"""Urgent-Symptom Pattern Matching and Alerting Service.

Specification Section 8 (SAF-3, Req ID: SAF-3):
- Matches explicit, human-readable patterns (chest pain with dyspnea, sudden focal
  neurological deficit, suicidal ideation statements, severe bleeding, anaphylaxis signs)
  against the raw narrative text and structured findings at submission BEFORE analysis.
- When matched, writes an immutable AuditEvent and returns banner text:
  "Possible urgent presentation: do not delay appropriate urgent human assessment"
- A match MUST NEVER block or delay anything else (workflow proceeds normally).
- Attached to every report generated for that case.
- LIMITATION: Prototype ruleset. NOT clinically validated (docs/known_gaps.md).
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import yaml
from pydantic import BaseModel, Field

from backend.app.persistence.audit_repository import AuditRepository
from backend.app.schemas.executions import AuditEvent as AuditEventSchema

logger = logging.getLogger(__name__)

URGENT_BANNER_TEXT = "Possible urgent presentation: do not delay appropriate urgent human assessment"

DEFAULT_RULES_PATH = Path(__file__).parent / "urgent_rules.yaml"


class UrgentRule(BaseModel):
    id: str
    name: str
    category: str
    description: str
    require_both_groups: bool = False
    primary_terms: List[str]
    secondary_terms: List[str] = Field(default_factory=list)


class UrgentMatchResult(BaseModel):
    is_urgent: bool = False
    matched_rule_ids: List[str] = Field(default_factory=list)
    matched_rule_names: List[str] = Field(default_factory=list)
    banner_text: Optional[str] = None


def load_urgent_rules(rules_path: Optional[Path] = None) -> List[UrgentRule]:
    """Load urgent symptom rules from YAML configuration file."""
    path = rules_path or DEFAULT_RULES_PATH
    if not path.exists():
        logger.warning("Urgent rules file not found at %s; returning empty rule set", path)
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules_raw = data.get("rules", [])
    return [UrgentRule(**r) for r in rules_raw]


def evaluate_urgent_rules(
    raw_text: str,
    findings: Optional[List[Any]] = None,
    rules: Optional[List[UrgentRule]] = None,
    case_id: Optional[str] = None,
    audit_repo: Optional[AuditRepository] = None,
) -> UrgentMatchResult:
    """Evaluate raw clinical text and structured findings against urgent rules.

    Safety Invariant: A match NEVER blocks or halts workflow analysis.
    """
    if rules is None:
        rules = load_urgent_rules()

    # Normalize narrative text for case-insensitive matching
    text_lower = (raw_text or "").lower()

    # Extract finding terms
    finding_terms_lower: List[str] = []
    if findings:
        for f in findings:
            if hasattr(f, "term") and f.term:
                finding_terms_lower.append(str(f.term).lower())
            if hasattr(f, "normalized_term") and f.normalized_term:
                finding_terms_lower.append(str(f.normalized_term).lower())

    combined_search_corpus = text_lower + " " + " ".join(finding_terms_lower)

    matched_ids: List[str] = []
    matched_names: List[str] = []

    for rule in rules:
        # Check primary terms
        primary_match = any(term.lower() in combined_search_corpus for term in rule.primary_terms)

        if rule.require_both_groups:
            # Check secondary terms
            secondary_match = any(term.lower() in combined_search_corpus for term in rule.secondary_terms)
            if primary_match and secondary_match:
                matched_ids.append(rule.id)
                matched_names.append(rule.name)
        else:
            if primary_match:
                matched_ids.append(rule.id)
                matched_names.append(rule.name)

    if matched_ids:
        # Emit an audit event for urgent pattern detection
        if audit_repo is not None and case_id:
            event = AuditEventSchema(
                event_id=str(uuid.uuid4()),
                event_type="URGENT_SYMPTOM_MATCHED",
                actor_id="system_urgent_filter",
                actor_role="system",
                resource_type="clinical_case",
                resource_id=case_id,
                payload={
                    "case_id": case_id,
                    "matched_rule_ids": matched_ids,
                    "matched_rule_names": matched_names,
                    "banner_text": URGENT_BANNER_TEXT,
                },
                timestamp=datetime.now(timezone.utc),
            )
            try:
                audit_repo.append_event(event)
            except Exception as e:
                logger.error("Failed to append urgent rule audit event: %s", e)

        return UrgentMatchResult(
            is_urgent=True,
            matched_rule_ids=matched_ids,
            matched_rule_names=matched_names,
            banner_text=URGENT_BANNER_TEXT,
        )

    return UrgentMatchResult(is_urgent=False, banner_text=None)
