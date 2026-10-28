"""Evidence Safety Gate (Extension B - EXT-B).

Specification Section 3 (Rules 1 to 6), Section 5.2, Appendix A3 (Req ID: EXT-B, SAF-1).
Pure function evaluate_gate(state, settings) -> GateDecision.

Gate Rules evaluated strictly in order:
1. Out-of-scope request (treatment, prescription, dosing, non-biomedical)
   -> finalize_abstain, OUT_OF_SCOPE
2. Verifier error on any material or safety-critical claim (after one bounded verifier retry)
   -> finalize_abstain, VERIFICATION_FAILED
3. Any safety-critical claim not SUPPORTED, or any material claim CONTRADICTED
   -> optimizer if iterations remain, else finalize_abstain
      (UNRESOLVED_CONTRADICTION if any material claim CONTRADICTED, else UNSUPPORTED_MATERIAL_CLAIMS)
4. Fewer than MIN_RELEVANT_PASSAGES relevant passages
   -> optimizer if iterations remain and an alternative query exists, else finalize_abstain, NO_RELEVANT_EVIDENCE
5. Confidence below threshold
   -> optimizer if iterations remain, else finalize_abstain, LOW_CONFIDENCE
6. Otherwise
   -> finalize_ready

Safety Invariants:
- High confidence NEVER overrides rules 1 to 4; confidence can only ADD an abstention reason.
- Gate-internal exception terminates in finalize_abstain with SYSTEM_ERROR, NEVER ready.
- Material claim = category other than 'background'.
- Safety-critical claim = category in {diagnostic_consideration, red_flag, contraindication, urgency, drug_interaction}.
- Non-safety-critical material claims that are not SUPPORTED stay in the ready report, shown with verification status, and lower claim_support.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings
from backend.app.safety.abstention import (
    build_abstention_decision,
    map_gate_rule_to_primary_reason,
)
from backend.app.safety.confidence import compute_confidence
from backend.app.safety.scope_rules import check_clinical_scope
from backend.app.schemas.evidence import QueryStatusEnum
from backend.app.schemas.executions import AgentStatusEnum, WorkflowState
from backend.app.schemas.reports import (
    AbstentionDecision,
    AbstentionReasonCode,
    CandidateClaim,
    ClaimVerification,
    ClaimVerificationStatusEnum,
)

logger = logging.getLogger(__name__)

SAFETY_CRITICAL_CATEGORIES: Set[str] = {
    "diagnostic_consideration",
    "red_flag",
    "contraindication",
    "urgency",
    "drug_interaction",
}


class GateOutcome(str, Enum):
    FINALIZE_READY = "finalize_ready"
    FINALIZE_ABSTAIN = "finalize_abstain"
    OPTIMIZER = "adaptive_optimizer"


class GateDecision(BaseModel):
    """Immutable evaluation outcome from the evidence safety gate."""
    outcome: GateOutcome
    rule_triggered: int = Field(..., ge=0, le=6, description="Rule 1..6, or 0 for SYSTEM_ERROR")
    primary_reason: Optional[AbstentionReasonCode] = None
    reason_codes: List[AbstentionReasonCode] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)
    target_node: str = Field(..., description="adaptive_optimizer | finalize_ready | finalize_abstain")
    affected_claim_ids: List[str] = Field(default_factory=list)
    abstention_decision: Optional[AbstentionDecision] = None
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


def is_safety_critical(claim: CandidateClaim, verif: Optional[ClaimVerification] = None) -> bool:
    """Determine whether a claim is safety-critical per SPEC 3."""
    if claim.is_safety_critical:
        return True
    category = (verif.category if verif and verif.category else getattr(claim, "category", None) or "").strip().lower()
    if category in SAFETY_CRITICAL_CATEGORIES:
        return True
    # If section is potential_considerations, treat as diagnostic consideration
    if claim.section in ("potential_considerations", "diagnostic_considerations"):
        return True
    return False


def is_material(claim: CandidateClaim, verif: Optional[ClaimVerification] = None) -> bool:
    """Determine whether a claim is material per SPEC 3 (category != 'background')."""
    category = (verif.category if verif and verif.category else getattr(claim, "category", None) or "").strip().lower()
    if category == "background":
        return False
    if claim.is_material is False:
        return False
    return True


def evaluate_gate(state: WorkflowState, settings: Settings) -> GateDecision:
    """Pure function evaluating safety gate rules 1 through 6 in exact order."""
    try:
        current_iteration = state.current_iteration
        max_iterations = settings.MAX_OPTIMIZER_ITERATIONS
        iterations_remain = current_iteration < max_iterations

        # Check scanner unavailable and missing patient info context
        scanner_unavailable = False
        if state.external_evidence and state.external_evidence.query_statuses:
            if any(st == QueryStatusEnum.UNAVAILABLE for st in state.external_evidence.query_statuses.values()):
                scanner_unavailable = True
        elif state.scanner_degraded:
            scanner_unavailable = True

        missing_patient_info = False
        if not state.is_benchmark and state.case:
            if state.case.age is None or state.case.symptom_duration is None:
                missing_patient_info = True

        # ---------------------------------------------------------------------
        # Rule 1: Out-of-scope request (treatment, prescription, dosing)
        # ---------------------------------------------------------------------
        raw_submission_text = ""
        if state.submission and state.submission.raw_text:
            raw_submission_text = state.submission.raw_text
        elif state.submission_payload and "raw_text" in state.submission_payload:
            raw_submission_text = str(state.submission_payload["raw_text"])
        elif state.case and state.case.chief_complaint:
            raw_submission_text = state.case.chief_complaint

        in_scope, scope_violation = check_clinical_scope(
            text=raw_submission_text,
            is_benchmark=state.is_benchmark,
        )
        if not in_scope:
            expl = scope_violation or "Out-of-scope request detected: treatment or dosing guidance requested."
            abst_dec = build_abstention_decision(
                primary_reason=AbstentionReasonCode.OUT_OF_SCOPE,
                explanation=expl,
                at_iteration_budget=(not iterations_remain),
                scanner_unavailable=scanner_unavailable,
                missing_required_patient_info=missing_patient_info,
            )
            return GateDecision(
                outcome=GateOutcome.FINALIZE_ABSTAIN,
                rule_triggered=1,
                primary_reason=AbstentionReasonCode.OUT_OF_SCOPE,
                reason_codes=abst_dec.reason_codes,
                explanation=expl,
                target_node="finalize_abstain",
                abstention_decision=abst_dec,
            )

        # ---------------------------------------------------------------------
        # Rule 2: Verifier error on any material or safety-critical claim
        # ---------------------------------------------------------------------
        # Check if Claim Verification Agent encountered an unrecoverable error
        has_verifier_error = False
        verifier_error_msg = ""
        for rec in state.agent_records:
            if rec.agent_name == "claim_verification" and rec.status == AgentStatusEnum.FAILED:
                has_verifier_error = True
                verifier_error_msg = rec.error_message or "Claim verification agent execution failed."
                break

        # Check verifications for error limitations
        verif_by_id: Dict[str, ClaimVerification] = {v.claim_id: v for v in state.verifications}
        report_claims = state.report.claims if state.report else []

        if not has_verifier_error:
            for claim in report_claims:
                v = verif_by_id.get(claim.claim_id)
                if v and v.status == ClaimVerificationStatusEnum.UNVERIFIABLE:
                    if v.limitations in ("VERIFIER_ERROR", "LLM_UNAVAILABLE"):
                        if is_material(claim, v) or is_safety_critical(claim, v):
                            has_verifier_error = True
                            verifier_error_msg = f"Verifier error on material claim {claim.claim_id}: {v.justification}"
                            break

        if has_verifier_error:
            expl = f"Claim verification failure: {verifier_error_msg}"
            abst_dec = build_abstention_decision(
                primary_reason=AbstentionReasonCode.VERIFICATION_FAILED,
                explanation=expl,
                at_iteration_budget=(not iterations_remain),
                scanner_unavailable=scanner_unavailable,
                missing_required_patient_info=missing_patient_info,
            )
            return GateDecision(
                outcome=GateOutcome.FINALIZE_ABSTAIN,
                rule_triggered=2,
                primary_reason=AbstentionReasonCode.VERIFICATION_FAILED,
                reason_codes=abst_dec.reason_codes,
                explanation=expl,
                target_node="finalize_abstain",
                abstention_decision=abst_dec,
            )

        # ---------------------------------------------------------------------
        # Rule 3: Any safety-critical claim not SUPPORTED, or material claim CONTRADICTED
        # ---------------------------------------------------------------------
        unsupported_safety_critical_ids: List[str] = []
        contradicted_material_ids: List[str] = []

        for claim in report_claims:
            v = verif_by_id.get(claim.claim_id)
            c_mat = is_material(claim, v)
            c_safe = is_safety_critical(claim, v)

            if c_safe:
                # Must be fully SUPPORTED
                if not v or v.status != ClaimVerificationStatusEnum.SUPPORTED:
                    unsupported_safety_critical_ids.append(claim.claim_id)

            if c_mat:
                # Can never be CONTRADICTED
                if v and v.status == ClaimVerificationStatusEnum.CONTRADICTED:
                    contradicted_material_ids.append(claim.claim_id)

        if unsupported_safety_critical_ids or contradicted_material_ids:
            has_contra = len(contradicted_material_ids) > 0
            if iterations_remain:
                expl = (
                    f"Optimizer iteration requested: {len(unsupported_safety_critical_ids)} unsupported safety-critical "
                    f"claims and {len(contradicted_material_ids)} contradicted material claims."
                )
                return GateDecision(
                    outcome=GateOutcome.OPTIMIZER,
                    rule_triggered=3,
                    primary_reason=None,
                    reason_codes=[],
                    explanation=expl,
                    target_node="adaptive_optimizer",
                    affected_claim_ids=unsupported_safety_critical_ids + contradicted_material_ids,
                )
            else:
                primary = (
                    AbstentionReasonCode.UNRESOLVED_CONTRADICTION
                    if has_contra
                    else AbstentionReasonCode.UNSUPPORTED_MATERIAL_CLAIMS
                )
                expl = (
                    f"Abstention triggered (budget exhausted): {len(unsupported_safety_critical_ids)} safety-critical "
                    f"claims not supported, {len(contradicted_material_ids)} contradicted material claims."
                )
                abst_dec = build_abstention_decision(
                    primary_reason=primary,
                    explanation=expl,
                    at_iteration_budget=True,
                    scanner_unavailable=scanner_unavailable,
                    missing_required_patient_info=missing_patient_info,
                    affected_claim_ids=unsupported_safety_critical_ids + contradicted_material_ids,
                )
                return GateDecision(
                    outcome=GateOutcome.FINALIZE_ABSTAIN,
                    rule_triggered=3,
                    primary_reason=primary,
                    reason_codes=abst_dec.reason_codes,
                    explanation=expl,
                    target_node="finalize_abstain",
                    affected_claim_ids=unsupported_safety_critical_ids + contradicted_material_ids,
                    abstention_decision=abst_dec,
                )

        # ---------------------------------------------------------------------
        # Rule 4: Fewer than MIN_RELEVANT_PASSAGES relevant passages
        # ---------------------------------------------------------------------
        retrieved_passages = (
            state.retrieved_evidence.results
            if state.retrieved_evidence and state.retrieved_evidence.results
            else []
        )
        relevant_passages = [
            p for p in retrieved_passages
            if p.retrieval_score >= settings.RETRIEVAL_MIN_SCORE
        ]

        if len(relevant_passages) < settings.MIN_RELEVANT_PASSAGES:
            alternative_query_exists = (
                bool(state.gap_hints)
                or (len(state.candidate_hypotheses) > 1)
                or (current_iteration == 0)
            )
            if iterations_remain and alternative_query_exists:
                expl = (
                    f"Optimizer iteration requested: only {len(relevant_passages)} relevant passages "
                    f"(minimum required: {settings.MIN_RELEVANT_PASSAGES}). Alternative queries available."
                )
                return GateDecision(
                    outcome=GateOutcome.OPTIMIZER,
                    rule_triggered=4,
                    primary_reason=None,
                    reason_codes=[],
                    explanation=expl,
                    target_node="adaptive_optimizer",
                )
            else:
                expl = (
                    f"Abstention triggered: only {len(relevant_passages)} relevant passages retrieved "
                    f"(minimum required: {settings.MIN_RELEVANT_PASSAGES}). No further queries available."
                )
                abst_dec = build_abstention_decision(
                    primary_reason=AbstentionReasonCode.NO_RELEVANT_EVIDENCE,
                    explanation=expl,
                    at_iteration_budget=(not iterations_remain),
                    scanner_unavailable=scanner_unavailable,
                    missing_required_patient_info=missing_patient_info,
                )
                return GateDecision(
                    outcome=GateOutcome.FINALIZE_ABSTAIN,
                    rule_triggered=4,
                    primary_reason=AbstentionReasonCode.NO_RELEVANT_EVIDENCE,
                    reason_codes=abst_dec.reason_codes,
                    explanation=expl,
                    target_node="finalize_abstain",
                    abstention_decision=abst_dec,
                )

        # ---------------------------------------------------------------------
        # Rule 5: Confidence below threshold
        # ---------------------------------------------------------------------
        # Calculate full confidence score with claim support
        material_claims = [c for c in report_claims if is_material(c, verif_by_id.get(c.claim_id))]
        if material_claims:
            sup_count = 0.0
            for c in material_claims:
                v = verif_by_id.get(c.claim_id)
                if v:
                    if v.status == ClaimVerificationStatusEnum.SUPPORTED:
                        sup_count += 1.0
                    elif v.status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED:
                        sup_count += 0.5
            claim_support_score = sup_count / float(len(material_claims))
        else:
            claim_support_score = 1.0

        retrieval_scores = [p.retrieval_score for p in retrieved_passages]
        unavailable_sources = 1 if scanner_unavailable else 0
        missing_fields = 0
        if state.case and not state.is_benchmark:
            if state.case.age is None:
                missing_fields += 1
            if state.case.symptom_duration is None:
                missing_fields += 1

        conf_res = compute_confidence(
            retrieval_scores=retrieval_scores,
            claim_support_score=claim_support_score,
            unavailable_sources=unavailable_sources,
            configured_sources=2,
            missing_case_fields=missing_fields,
            total_case_fields=4,
            is_baseline=False,
        )
        current_conf = conf_res.confidence_score

        if current_conf < settings.CONFIDENCE_THRESHOLD:
            if iterations_remain:
                expl = (
                    f"Optimizer iteration requested: confidence {current_conf:.4f} is below "
                    f"threshold {settings.CONFIDENCE_THRESHOLD:.4f}."
                )
                return GateDecision(
                    outcome=GateOutcome.OPTIMIZER,
                    rule_triggered=5,
                    primary_reason=None,
                    reason_codes=[],
                    explanation=expl,
                    target_node="adaptive_optimizer",
                )
            else:
                expl = (
                    f"Abstention triggered (budget exhausted): confidence {current_conf:.4f} is below "
                    f"threshold {settings.CONFIDENCE_THRESHOLD:.4f}."
                )
                abst_dec = build_abstention_decision(
                    primary_reason=AbstentionReasonCode.LOW_CONFIDENCE,
                    explanation=expl,
                    at_iteration_budget=True,
                    scanner_unavailable=scanner_unavailable,
                    missing_required_patient_info=missing_patient_info,
                )
                return GateDecision(
                    outcome=GateOutcome.FINALIZE_ABSTAIN,
                    rule_triggered=5,
                    primary_reason=AbstentionReasonCode.LOW_CONFIDENCE,
                    reason_codes=abst_dec.reason_codes,
                    explanation=expl,
                    target_node="finalize_abstain",
                    abstention_decision=abst_dec,
                )

        # ---------------------------------------------------------------------
        # Rule 6: Otherwise -> finalize_ready
        # ---------------------------------------------------------------------
        return GateDecision(
            outcome=GateOutcome.FINALIZE_READY,
            rule_triggered=6,
            primary_reason=None,
            reason_codes=[],
            explanation=(
                f"All gate safety rules satisfied. Confidence {current_conf:.4f} meets or exceeds "
                f"threshold {settings.CONFIDENCE_THRESHOLD:.4f} with no contradicted material claims "
                f"and all safety-critical claims supported."
            ),
            target_node="finalize_ready",
        )

    except Exception as exc:
        # Safety Invariant: Gate-internal exception ends in finalize_abstain with SYSTEM_ERROR, never ready
        logger.exception("Unexpected exception inside evaluate_gate: %s", exc)
        err_expl = f"Internal gate evaluation failure: {str(exc)}"
        abst_dec = build_abstention_decision(
            primary_reason=AbstentionReasonCode.SYSTEM_ERROR,
            explanation=err_expl,
            at_iteration_budget=True,
        )
        return GateDecision(
            outcome=GateOutcome.FINALIZE_ABSTAIN,
            rule_triggered=0,
            primary_reason=AbstentionReasonCode.SYSTEM_ERROR,
            reason_codes=abst_dec.reason_codes,
            explanation=err_expl,
            target_node="finalize_abstain",
            abstention_decision=abst_dec,
        )
