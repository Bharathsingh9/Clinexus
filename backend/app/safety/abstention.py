"""Explicit Abstention Mechanism (Extension B - EXT-B).

Specification Section 3, Section 5.2, Appendix A3 (Req ID: EXT-B):
- Produces AbstentionDecision with strict reason code mappings:
    * Rule 1 -> OUT_OF_SCOPE
    * Rule 2 -> VERIFICATION_FAILED
    * Rule 3 -> UNRESOLVED_CONTRADICTION (if material claim CONTRADICTED) else UNSUPPORTED_MATERIAL_CLAIMS
    * Rule 4 -> NO_RELEVANT_EVIDENCE
    * Rule 5 -> LOW_CONFIDENCE
    * Additional reason codes appended on abstention:
        + ITERATION_BUDGET_EXHAUSTED: if abstaining with iteration counter at maximum
        + CRITICAL_SOURCE_UNAVAILABLE: if abstaining and Scanner is UNAVAILABLE
        + INSUFFICIENT_PATIENT_INFO: if abstaining and required case fields remain unknown
        + SYSTEM_ERROR: on gate-internal or pipeline exception
- Produces sanitized CandidateClinicalReport containing ONLY:
    * The abstention decision
    * Verified-SUPPORTED observations (explicitly marked)
    * Patient-fact summary
    * Missing information
    * Suggested next steps
- Prohibits all unverified considerations, speculative diagnoses, or filler medical content.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from backend.app.schemas.clinical_case import ClinicalCase
from backend.app.schemas.reports import (
    AbstentionDecision,
    AbstentionReasonCode,
    AbstentionRequirement,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
)


def map_gate_rule_to_primary_reason(
    rule_id: int,
    has_contradicted_material_claim: bool = False,
) -> AbstentionReasonCode:
    """Map gate rule trigger to primary AbstentionReasonCode per SPEC Section 3."""
    if rule_id == 1:
        return AbstentionReasonCode.OUT_OF_SCOPE
    elif rule_id == 2:
        return AbstentionReasonCode.VERIFICATION_FAILED
    elif rule_id == 3:
        if has_contradicted_material_claim:
            return AbstentionReasonCode.UNRESOLVED_CONTRADICTION
        return AbstentionReasonCode.UNSUPPORTED_MATERIAL_CLAIMS
    elif rule_id == 4:
        return AbstentionReasonCode.NO_RELEVANT_EVIDENCE
    elif rule_id == 5:
        return AbstentionReasonCode.LOW_CONFIDENCE
    elif rule_id == 0 or rule_id == 99:
        return AbstentionReasonCode.SYSTEM_ERROR
    else:
        return AbstentionReasonCode.SYSTEM_ERROR


def build_abstention_decision(
    primary_reason: AbstentionReasonCode,
    explanation: str,
    at_iteration_budget: bool = False,
    scanner_unavailable: bool = False,
    missing_required_patient_info: bool = False,
    affected_claim_ids: Optional[List[str]] = None,
    affected_sections: Optional[List[str]] = None,
    missing_evidence: Optional[List[str]] = None,
    missing_information: Optional[List[str]] = None,
    suggested_next_steps: Optional[List[str]] = None,
    requires: Optional[List[AbstentionRequirement]] = None,
) -> AbstentionDecision:
    """Build structured AbstentionDecision including primary and secondary reason codes."""
    reason_codes: List[AbstentionReasonCode] = [primary_reason]

    # Additional contextual reason codes required by SPEC 5.2 and Phase 4 Stage A instructions
    if at_iteration_budget and AbstentionReasonCode.ITERATION_BUDGET_EXHAUSTED not in reason_codes:
        reason_codes.append(AbstentionReasonCode.ITERATION_BUDGET_EXHAUSTED)

    if scanner_unavailable and AbstentionReasonCode.CRITICAL_SOURCE_UNAVAILABLE not in reason_codes:
        reason_codes.append(AbstentionReasonCode.CRITICAL_SOURCE_UNAVAILABLE)

    if missing_required_patient_info and AbstentionReasonCode.INSUFFICIENT_PATIENT_INFO not in reason_codes:
        reason_codes.append(AbstentionReasonCode.INSUFFICIENT_PATIENT_INFO)

    # Determine default requirements if not provided
    inferred_requires = list(requires or [])
    if not inferred_requires:
        if primary_reason in (AbstentionReasonCode.NO_RELEVANT_EVIDENCE, AbstentionReasonCode.LOW_CONFIDENCE):
            inferred_requires.append(AbstentionRequirement.RETRIEVAL)
        if primary_reason == AbstentionReasonCode.INSUFFICIENT_PATIENT_INFO or missing_required_patient_info:
            inferred_requires.append(AbstentionRequirement.CLARIFICATION)
        inferred_requires.append(AbstentionRequirement.HUMAN_REVIEW)

    # Deduplicate requirements preserving order
    deduped_requires: List[AbstentionRequirement] = []
    for req in inferred_requires:
        if req not in deduped_requires:
            deduped_requires.append(req)

    default_steps = suggested_next_steps or [
        "Conduct thorough clinical assessment; this decision-support tool has abstained from providing differential considerations.",
        "Verify documented patient history and physical examination findings.",
        "Consult specialist literature or institutional clinical protocols independently.",
    ]

    return AbstentionDecision(
        decision_id=str(uuid.uuid4()),
        abstained=True,
        reason_codes=reason_codes,
        explanation=explanation,
        affected_claim_ids=affected_claim_ids or [],
        affected_sections=affected_sections or [],
        missing_evidence=missing_evidence or [],
        missing_information=missing_information or [],
        suggested_next_steps=default_steps,
        requires=deduped_requires,
    )


def create_abstained_report(
    original_report: Optional[CandidateClinicalReport],
    abstention_decision: AbstentionDecision,
    case: Optional[ClinicalCase] = None,
    verifications: Optional[List[ClaimVerification]] = None,
) -> CandidateClinicalReport:
    """Sanitize report on explicit abstention per SPEC 5.2.

    An abstained report contains ONLY:
    - the abstention decision
    - verified-SUPPORTED observations (marked)
    - patient-fact summary
    - missing information
    - suggested next steps
    NO unverified considerations, speculative diagnoses, or filler medical content.
    """
    case_id = (
        original_report.case_id
        if original_report
        else (case.case_id if case else str(uuid.uuid4()))
    )
    version_num = (original_report.version_number + 1) if original_report else 1
    report_version_id = str(uuid.uuid4())

    # 1. Identify verified SUPPORTED claims ONLY
    verif_map: Dict[str, ClaimVerification] = {
        v.claim_id: v for v in (verifications or [])
    }

    supported_claims: List[CandidateClaim] = []
    supported_obs_lines: List[str] = []

    if original_report:
        for claim in original_report.claims:
            v_record = verif_map.get(claim.claim_id)
            if v_record and v_record.status == ClaimVerificationStatusEnum.SUPPORTED:
                # Mark as verified SUPPORTED observation
                supported_claims.append(
                    CandidateClaim(
                        claim_id=claim.claim_id,
                        report_version_id=report_version_id,
                        claim_text=f"[SUPPORTED] {claim.claim_text}",
                        section="evidence_supported_observations",
                        is_material=claim.is_material,
                        is_safety_critical=claim.is_safety_critical,
                        cited_evidence_ids=claim.cited_evidence_ids,
                        source_spans=claim.source_spans,
                    )
                )
                supported_obs_lines.append(f"- [SUPPORTED] {claim.claim_text}")

    # 2. Extract patient-fact summary
    patient_facts: List[str] = []
    if case:
        if case.chief_complaint:
            patient_facts.append(f"Chief Complaint: {case.chief_complaint}")
        if case.age is not None:
            patient_facts.append(f"Age: {case.age}")
        if case.sex:
            patient_facts.append(f"Sex: {case.sex}")
        if case.symptom_duration:
            patient_facts.append(f"Duration: {case.symptom_duration}")
        for symptom in case.structured_symptoms:
            polarity_str = "Present" if symptom.polarity.value == "PRESENT" else "Absent"
            patient_facts.append(f"- {symptom.term} ({polarity_str})")
    elif original_report and "case_summary" in original_report.sections:
        patient_facts.append(original_report.sections["case_summary"])

    summary_text = (
        "CLINICAL DECISION SUPPORT ABSTAINED.\n"
        f"Reason(s): {', '.join(rc.value for rc in abstention_decision.reason_codes)}.\n"
        f"Explanation: {abstention_decision.explanation}\n"
        "All speculative or unverified considerations have been suppressed to prevent ungrounded clinical guidance."
    )

    # 3. Assemble strictly sanitized sections
    sections: Dict[str, str] = {
        "case_summary": (
            "Patient Facts Summary:\n" + "\n".join(patient_facts)
            if patient_facts
            else "Patient presentation recorded."
        ),
        "evidence_supported_observations": (
            "Verified Supported Observations:\n" + "\n".join(supported_obs_lines)
            if supported_obs_lines
            else "No observations verified as fully supported by retrieved evidence."
        ),
        "potential_considerations": (
            "[OMITTED DUE TO ABSTENTION] Unverified diagnostic considerations suppressed for safety."
        ),
        "alternative_explanations": (
            "[OMITTED DUE TO ABSTENTION] Suppressed for safety."
        ),
        "missing_information": (
            "Missing Clinical Information:\n"
            + (
                "\n".join(f"- {m}" for m in abstention_decision.missing_information)
                if abstention_decision.missing_information
                else "- No specific missing fields documented."
            )
        ),
        "recommended_evidence_review_steps": (
            "Suggested Next Steps:\n"
            + "\n".join(f"- {s}" for s in abstention_decision.suggested_next_steps)
        ),
        "abstention_decision": (
            f"Abstained: True\n"
            f"Reason Codes: {', '.join(rc.value for rc in abstention_decision.reason_codes)}\n"
            f"Explanation: {abstention_decision.explanation}"
        ),
    }

    return CandidateClinicalReport(
        report_id=original_report.report_id if original_report else str(uuid.uuid4()),
        report_version_id=report_version_id,
        version_number=version_num,
        case_id=case_id,
        summary=summary_text,
        claims=supported_claims,
        sections=sections,
        confidence_score=0.0,
        status=ReportStatusEnum.ABSTAINED,
        created_at=datetime.now(timezone.utc),
    )
