"""Unit tests and truth-table verification for Evidence Safety Gate (EXT-B).

Specification Section 3 (Rules 1 to 6), Section 5.2, Appendix A3 (Req ID: EXT-B, SAF-1).
Tests include:
- Parametrized truth-table over the input space proving finalize_ready is returned ONLY
  when NONE of rules 1 to 5 trigger.
- Confidence 0.99 with a CONTRADICTED material claim is NEVER ready.
- Exact mapping of all reason codes:
    * Rule 1 -> OUT_OF_SCOPE
    * Rule 2 -> VERIFICATION_FAILED
    * Rule 3 -> UNRESOLVED_CONTRADICTION / UNSUPPORTED_MATERIAL_CLAIMS
    * Rule 4 -> NO_RELEVANT_EVIDENCE
    * Rule 5 -> LOW_CONFIDENCE
    * Contextual codes: ITERATION_BUDGET_EXHAUSTED, CRITICAL_SOURCE_UNAVAILABLE, INSUFFICIENT_PATIENT_INFO
    * System exception -> SYSTEM_ERROR
"""

from datetime import datetime, timezone
import pytest

from backend.app.core.config import Settings
from backend.app.safety.gate import (
    GateDecision,
    GateOutcome,
    evaluate_gate,
)
from backend.app.schemas.clinical_case import CaseSubmission, ClinicalCase
from backend.app.schemas.evidence import (
    EvidenceChunk,
    ExternalEvidenceSet,
    QueryStatusEnum,
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)
from backend.app.schemas.executions import (
    AgentExecutionRecord,
    AgentStatusEnum,
    WorkflowState,
)
from backend.app.schemas.reports import (
    AbstentionReasonCode,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    VerificationMethodEnum,
)


def _make_dummy_case(with_patient_info: bool = True) -> ClinicalCase:
    return ClinicalCase(
        raw_text_hash="c" * 64,
        age=50 if with_patient_info else None,
        age_missing_reason=None if with_patient_info else "Unknown",
        sex="M" if with_patient_info else None,
        chief_complaint="Chest discomfort and dyspnea on exertion",
        symptom_duration="2 weeks" if with_patient_info else None,
        duration_missing_reason=None if with_patient_info else "Unknown",
    )


def _make_dummy_evidence(count: int = 5, score: float = 0.85) -> RetrievedEvidenceSet:
    results = []
    for i in range(count):
        results.append(
            RetrievedPassageResult(
                document_id=f"doc-{i}",
                evidence_id=f"ev:pubmed:1000{i}:0",
                chunk_id=f"chunk-{i}",
                source="pubmed",
                retrieval_score=score,
                rank=i + 1,
                corpus_version="v1",
                embedding_model="default",
                query_id="q-1",
                text=f"[SYNTHETIC] Clinical evidence statement {i} confirming finding.",
            )
        )
    return RetrievedEvidenceSet(
        query_id="q-1",
        results=results,
    )


def _make_dummy_report(
    claims: list[CandidateClaim],
    confidence: float = 0.80,
) -> CandidateClinicalReport:
    return CandidateClinicalReport(
        version_number=1,
        case_id="case-100",
        summary="Clinical synthesis candidate report",
        claims=claims,
        confidence_score=confidence,
    )


def test_rule_1_out_of_scope_request():
    """Rule 1: Prohibited treatment, prescription, or dosing request triggers finalize_abstain with OUT_OF_SCOPE."""
    sub = CaseSubmission(raw_text="Patient has chest pain. How to treat with medications and what is the dosage?")
    state = WorkflowState(
        case_id="case-1",
        submission=sub,
        current_iteration=0,
    )
    settings = Settings()

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 1
    assert decision.primary_reason == AbstentionReasonCode.OUT_OF_SCOPE
    assert AbstentionReasonCode.OUT_OF_SCOPE in decision.reason_codes


def test_rule_1_bypassed_in_benchmark_mode():
    """Rule 1: In benchmark mode (SPEC 9.8), biomedical factual questions are strictly in-scope."""
    sub = CaseSubmission(raw_text="What is the recommended treatment for myocardial infarction?")
    state = WorkflowState(
        case_id="case-bench-1",
        submission=sub,
        is_benchmark=True,
        current_iteration=3,  # at max budget so it falls to later rules
    )
    settings = Settings()

    decision = evaluate_gate(state, settings)
    # Rule 1 must NOT trigger in benchmark mode
    assert decision.rule_triggered != 1


def test_rule_2_verifier_execution_failure():
    """Rule 2: Unrecoverable ClaimVerificationAgent failure triggers finalize_abstain with VERIFICATION_FAILED."""
    state = WorkflowState(
        case_id="case-2",
        submission=CaseSubmission(raw_text="Patient presents with persistent cough."),
        case=_make_dummy_case(),
        agent_records=[
            AgentExecutionRecord(
                agent_name="claim_verification",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="Verifier LLM call failed after retry.",
                input_summary="Verifying claims",
            )
        ],
    )
    settings = Settings()

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 2
    assert decision.primary_reason == AbstentionReasonCode.VERIFICATION_FAILED
    assert AbstentionReasonCode.VERIFICATION_FAILED in decision.reason_codes


def test_rule_3_unsupported_safety_critical_claim():
    """Rule 3: Safety-critical claim not SUPPORTED routes to optimizer if budget remains, else abstains."""
    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Consider pulmonary embolism as high risk diagnostic consideration.",
        section="potential_considerations",
        is_material=True,
        is_safety_critical=True,
    )
    verif = ClaimVerification(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        category="diagnostic_consideration",
        status=ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE,
        justification="Insufficient evidence found in passages.",
        method=VerificationMethodEnum.HYBRID,
    )
    report = _make_dummy_report([claim])
    settings = Settings()

    # Case A: Iterations remain -> routes to adaptive_optimizer
    state_rem = WorkflowState(
        case_id="case-3a",
        submission=CaseSubmission(raw_text="Patient presents with acute pleuritic pain."),
        case=_make_dummy_case(),
        report=report,
        verifications=[verif],
        current_iteration=0,
    )
    dec_rem = evaluate_gate(state_rem, settings)
    assert dec_rem.outcome == GateOutcome.OPTIMIZER
    assert dec_rem.rule_triggered == 3

    # Case B: Budget exhausted -> routes to finalize_abstain with UNSUPPORTED_MATERIAL_CLAIMS and ITERATION_BUDGET_EXHAUSTED
    state_cap = WorkflowState(
        case_id="case-3b",
        submission=CaseSubmission(raw_text="Patient presents with acute pleuritic pain."),
        case=_make_dummy_case(),
        report=report,
        verifications=[verif],
        current_iteration=3,
    )
    dec_cap = evaluate_gate(state_cap, settings)
    assert dec_cap.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert dec_cap.rule_triggered == 3
    assert dec_cap.primary_reason == AbstentionReasonCode.UNSUPPORTED_MATERIAL_CLAIMS
    assert AbstentionReasonCode.ITERATION_BUDGET_EXHAUSTED in dec_cap.reason_codes


def test_rule_3_contradicted_material_claim_high_confidence_never_ready():
    """Rule 3 Invariant: Confidence 0.99 with a CONTRADICTED material claim is NEVER ready."""
    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Troponin elevation indicates acute pericarditis.",
        section="evidence_supported_observations",
        is_material=True,
        is_safety_critical=False,
    )
    verif = ClaimVerification(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        category="evidence_observation",
        status=ClaimVerificationStatusEnum.CONTRADICTED,
        justification="Literature refutes troponin as characteristic for pericarditis alone.",
        method=VerificationMethodEnum.HYBRID,
    )
    report = _make_dummy_report([claim], confidence=0.99)
    settings = Settings()

    # Even at iteration budget cap with 0.99 confidence, must ABSTAIN with UNRESOLVED_CONTRADICTION
    state = WorkflowState(
        case_id="case-contra",
        submission=CaseSubmission(raw_text="Patient with retrosternal chest pain."),
        case=_make_dummy_case(),
        report=report,
        verifications=[verif],
        current_iteration=3,
    )

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 3
    assert decision.primary_reason == AbstentionReasonCode.UNRESOLVED_CONTRADICTION
    assert AbstentionReasonCode.UNRESOLVED_CONTRADICTION in decision.reason_codes
    assert AbstentionReasonCode.ITERATION_BUDGET_EXHAUSTED in decision.reason_codes


def test_rule_4_insufficient_relevant_passages():
    """Rule 4: Fewer than MIN_RELEVANT_PASSAGES relevant passages routes to optimizer or abstains with NO_RELEVANT_EVIDENCE."""
    settings = Settings(MIN_RELEVANT_PASSAGES=3)
    # Only 1 passage passing threshold
    evidence = _make_dummy_evidence(count=1, score=0.85)

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="General finding observation.",
        section="evidence_supported_observations",
        is_material=True,
    )
    verif = ClaimVerification(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        category="background",
        status=ClaimVerificationStatusEnum.SUPPORTED,
        justification="Supported by finding.",
        method=VerificationMethodEnum.DETERMINISTIC,
    )
    report = _make_dummy_report([claim])

    # Iteration budget exhausted -> finalize_abstain with NO_RELEVANT_EVIDENCE
    state = WorkflowState(
        case_id="case-4",
        submission=CaseSubmission(raw_text="Rare clinical presentation."),
        case=_make_dummy_case(),
        retrieved_evidence=evidence,
        report=report,
        verifications=[verif],
        current_iteration=3,
    )

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 4
    assert decision.primary_reason == AbstentionReasonCode.NO_RELEVANT_EVIDENCE
    assert AbstentionReasonCode.NO_RELEVANT_EVIDENCE in decision.reason_codes


def test_rule_5_low_confidence():
    """Rule 5: Confidence below CONFIDENCE_THRESHOLD routes to optimizer or abstains with LOW_CONFIDENCE."""
    settings = Settings(CONFIDENCE_THRESHOLD=0.65)
    evidence = _make_dummy_evidence(count=5, score=0.35)  # Low scores yield low retrieval strength

    # Material claim is only PARTIALLY supported, pulling down claim support
    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Mildly elevated ESR indicates inflammatory process.",
        section="evidence_supported_observations",
        is_material=True,
        is_safety_critical=False,
    )
    verif = ClaimVerification(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        category="evidence_observation",
        status=ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
        justification="Weak partial evidence.",
        method=VerificationMethodEnum.HYBRID,
    )
    report = _make_dummy_report([claim], confidence=0.45)

    state = WorkflowState(
        case_id="case-5",
        submission=CaseSubmission(raw_text="Vague joint pain and fatigue."),
        case=_make_dummy_case(with_patient_info=False),
        retrieved_evidence=evidence,
        report=report,
        verifications=[verif],
        current_iteration=3,
    )

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 5
    assert decision.primary_reason == AbstentionReasonCode.LOW_CONFIDENCE
    assert AbstentionReasonCode.LOW_CONFIDENCE in decision.reason_codes
    assert AbstentionReasonCode.INSUFFICIENT_PATIENT_INFO in decision.reason_codes


def test_rule_6_finalize_ready_when_all_rules_pass():
    """Rule 6: finalize_ready is returned ONLY when rules 1 to 5 are strictly satisfied."""
    settings = Settings(CONFIDENCE_THRESHOLD=0.65, MIN_RELEVANT_PASSAGES=2)
    evidence = _make_dummy_evidence(count=5, score=0.90)

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Consistent with typical viral bronchitis presentation.",
        section="potential_considerations",
        is_material=True,
        is_safety_critical=True,
    )
    verif = ClaimVerification(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        category="diagnostic_consideration",
        status=ClaimVerificationStatusEnum.SUPPORTED,
        justification="Directly supported by literature citations.",
        method=VerificationMethodEnum.HYBRID,
    )
    report = _make_dummy_report([claim], confidence=0.88)

    state = WorkflowState(
        case_id="case-ready",
        submission=CaseSubmission(raw_text="Patient presents with rhinorrhea and low-grade fever for 4 days."),
        case=_make_dummy_case(with_patient_info=True),
        retrieved_evidence=evidence,
        report=report,
        verifications=[verif],
        current_iteration=0,
    )

    decision = evaluate_gate(state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_READY
    assert decision.rule_triggered == 6
    assert decision.target_node == "finalize_ready"
    assert len(decision.reason_codes) == 0


def test_system_error_on_gate_exception():
    """Safety Invariant: Any unexpected internal exception terminates in finalize_abstain with SYSTEM_ERROR, NEVER ready."""
    settings = Settings()
    # Pass an invalid state that triggers an attribute error or type error
    broken_state = None  # None state triggers exception in evaluate_gate

    decision = evaluate_gate(broken_state, settings)

    assert decision.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert decision.rule_triggered == 0
    assert decision.primary_reason == AbstentionReasonCode.SYSTEM_ERROR
    assert AbstentionReasonCode.SYSTEM_ERROR in decision.reason_codes


@pytest.mark.parametrize(
    (
        "test_id",
        "raw_text",
        "verifier_failed",
        "claim_status",
        "claim_is_safety_critical",
        "claim_is_material",
        "relevant_passages_count",
        "passage_score",
        "iteration",
        "expected_outcome",
        "expected_rule",
        "expected_primary_reason",
    ),
    [
        # Rule 1: Out of scope request -> Abstain regardless of other inputs
        (
            "rule_1_out_of_scope",
            "Please prescribe 500mg amoxicillin three times daily.",
            False,
            ClaimVerificationStatusEnum.SUPPORTED,
            False,
            False,
            5,
            0.99,
            0,
            GateOutcome.FINALIZE_ABSTAIN,
            1,
            AbstentionReasonCode.OUT_OF_SCOPE,
        ),
        # Rule 2: Verifier agent failed -> Abstain regardless of confidence
        (
            "rule_2_verifier_failed",
            "Patient presents with fever and cough.",
            True,
            ClaimVerificationStatusEnum.SUPPORTED,
            False,
            False,
            5,
            0.99,
            0,
            GateOutcome.FINALIZE_ABSTAIN,
            2,
            AbstentionReasonCode.VERIFICATION_FAILED,
        ),
        # Rule 3: Unsupported safety-critical claim, iteration budget remaining -> Optimizer
        (
            "rule_3_unsupported_sc_iterations_remain",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE,
            True,
            True,
            5,
            0.85,
            0,
            GateOutcome.OPTIMIZER,
            3,
            None,
        ),
        # Rule 3: Unsupported safety-critical claim, iteration budget exhausted -> Abstain
        (
            "rule_3_unsupported_sc_budget_exhausted",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE,
            True,
            True,
            5,
            0.85,
            3,
            GateOutcome.FINALIZE_ABSTAIN,
            3,
            AbstentionReasonCode.UNSUPPORTED_MATERIAL_CLAIMS,
        ),
        # Rule 3: Contradicted material claim with extreme confidence (0.99), iterations remain -> Optimizer
        (
            "rule_3_contradicted_material_high_conf_iter_remains",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.CONTRADICTED,
            False,
            True,
            5,
            0.99,
            1,
            GateOutcome.OPTIMIZER,
            3,
            None,
        ),
        # Rule 3: Contradicted material claim with extreme confidence (0.99), budget exhausted -> Abstain (NEVER ready)
        (
            "rule_3_contradicted_material_high_conf_budget_exhausted",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.CONTRADICTED,
            False,
            True,
            5,
            0.99,
            3,
            GateOutcome.FINALIZE_ABSTAIN,
            3,
            AbstentionReasonCode.UNRESOLVED_CONTRADICTION,
        ),
        # Rule 4: Insufficient passages (< MIN_RELEVANT_PASSAGES=2), iterations remain -> Optimizer
        (
            "rule_4_insufficient_passages_iter_remains",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.SUPPORTED,
            True,
            True,
            1,  # below minimum 2
            0.85,
            0,
            GateOutcome.OPTIMIZER,
            4,
            None,
        ),
        # Rule 4: Insufficient passages (< 2), budget exhausted -> Abstain
        (
            "rule_4_insufficient_passages_budget_exhausted",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.SUPPORTED,
            True,
            True,
            1,
            0.85,
            3,
            GateOutcome.FINALIZE_ABSTAIN,
            4,
            AbstentionReasonCode.NO_RELEVANT_EVIDENCE,
        ),
        # Rule 5: Low confidence (< CONFIDENCE_THRESHOLD=0.65), iterations remain -> Optimizer
        (
            "rule_5_low_confidence_iter_remains",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
            False,
            True,
            5,
            0.35,  # low passage score pulls retrieval strength down
            1,
            GateOutcome.OPTIMIZER,
            5,
            None,
        ),
        # Rule 5: Low confidence (< 0.65), budget exhausted -> Abstain
        (
            "rule_5_low_confidence_budget_exhausted",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
            False,
            True,
            5,
            0.35,  # low passage score pulls retrieval strength down
            3,
            GateOutcome.FINALIZE_ABSTAIN,
            5,
            AbstentionReasonCode.LOW_CONFIDENCE,
        ),
        # Rule 6: All rules pass, budget remaining -> Finalize Ready (ONLY way to reach ready)
        (
            "rule_6_all_pass_iter_0",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.SUPPORTED,
            True,
            True,
            5,
            0.85,
            0,
            GateOutcome.FINALIZE_READY,
            6,
            None,
        ),
        # Rule 6: All rules pass, at iteration cap -> Finalize Ready
        (
            "rule_6_all_pass_iter_cap",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.SUPPORTED,
            True,
            True,
            5,
            0.85,
            3,
            GateOutcome.FINALIZE_READY,
            6,
            None,
        ),
        # Rule 6: Non-safety-critical material claim that is unverified stays in ready report and doesn't block ready
        (
            "rule_6_non_safety_critical_unverified_allows_ready",
            "Patient presents with fever and cough.",
            False,
            ClaimVerificationStatusEnum.UNVERIFIABLE,
            False,  # is_safety_critical=False
            True,   # is_material=True
            5,
            0.85,
            0,
            GateOutcome.FINALIZE_READY,
            6,
            None,
        ),
    ],
)
def test_gate_truth_table_parametrized_matrix(
    test_id: str,
    raw_text: str,
    verifier_failed: bool,
    claim_status: ClaimVerificationStatusEnum,
    claim_is_safety_critical: bool,
    claim_is_material: bool,
    relevant_passages_count: int,
    passage_score: float,
    iteration: int,
    expected_outcome: GateOutcome,
    expected_rule: int,
    expected_primary_reason: AbstentionReasonCode,
):
    """Parametrized truth-table matrix covering the gate input space and reason code mappings."""
    settings = Settings(CONFIDENCE_THRESHOLD=0.65, MIN_RELEVANT_PASSAGES=2)
    sub = CaseSubmission(raw_text=raw_text)

    agent_records = []
    if verifier_failed:
        agent_records.append(
            AgentExecutionRecord(
                agent_name="claim_verification",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="Verifier failure",
                input_summary="Verifying claims",
            )
        )

    claims = []
    verifs = []

    if test_id == "rule_6_non_safety_critical_unverified_allows_ready":
        # 2 supported material claims + 1 unverified non-safety-critical material claim
        # claim_support = 2/3 = 0.667 -> total confidence = 0.817 >= 0.65
        for i in range(2):
            c = CandidateClaim(
                report_version_id="rv-p-sup",
                claim_text=f"Supported observation {i}.",
                section="evidence_supported_observations",
                is_material=True,
                is_safety_critical=False,
            )
            v = ClaimVerification(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                category="evidence_observation",
                status=ClaimVerificationStatusEnum.SUPPORTED,
                justification="Supported by literature.",
                method=VerificationMethodEnum.HYBRID,
            )
            claims.append(c)
            verifs.append(v)

        c_unv = CandidateClaim(
            report_version_id="rv-p-unv",
            claim_text="Unverified non-safety-critical observation.",
            section="evidence_supported_observations",
            is_material=True,
            is_safety_critical=False,
        )
        v_unv = ClaimVerification(
            claim_id=c_unv.claim_id,
            claim_text=c_unv.claim_text,
            category="evidence_observation",
            status=ClaimVerificationStatusEnum.UNVERIFIABLE,
            justification="Passage text does not mention this detail.",
            method=VerificationMethodEnum.HYBRID,
        )
        claims.append(c_unv)
        verifs.append(v_unv)
    else:
        claim = CandidateClaim(
            report_version_id="rv-param-1",
            claim_text="Parametrized evaluated claim statement.",
            section="potential_considerations" if claim_is_safety_critical else "evidence_supported_observations",
            is_material=claim_is_material,
            is_safety_critical=claim_is_safety_critical,
        )
        cat = "diagnostic_consideration" if claim_is_safety_critical else "evidence_observation"
        verif = ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            category=cat,
            status=claim_status,
            justification="Truth table justification text.",
            method=VerificationMethodEnum.HYBRID,
        )
        claims.append(claim)
        verifs.append(verif)

    report = _make_dummy_report(claims)
    evidence = _make_dummy_evidence(count=relevant_passages_count, score=passage_score)

    has_patient_info = "low_confidence" not in test_id
    state = WorkflowState(
        case_id=f"case-tt-{test_id}",
        submission=sub,
        case=_make_dummy_case(with_patient_info=has_patient_info),
        retrieved_evidence=evidence,
        report=report,
        verifications=verifs,
        agent_records=agent_records,
        current_iteration=iteration,
    )

    decision = evaluate_gate(state, settings)

    assert decision.outcome == expected_outcome, f"Failed on {test_id}: expected outcome {expected_outcome}, got {decision.outcome}"
    assert decision.rule_triggered == expected_rule, f"Failed on {test_id}: expected rule {expected_rule}, got {decision.rule_triggered}"
    if expected_primary_reason is not None:
        assert decision.primary_reason == expected_primary_reason, f"Failed on {test_id}: expected primary reason {expected_primary_reason}, got {decision.primary_reason}"
        assert expected_primary_reason in decision.reason_codes

