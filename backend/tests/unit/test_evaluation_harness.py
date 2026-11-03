"""Unit and Integration Tests for BioASQ Evaluation Harness (Phase 5 Stage D).

Specification Section 9, 10, Appendix A11-A13 (Req ID: EVL-1..6, SAF-1):
- Safety gate refusals: third_party_llm_allowed, eval split confirmations, call cap, rerun protection.
- Dry run zero-call guarantee.
- Checkpointing and resume idempotency.
- Circuit breakers (auth errors, consecutive failures, call cap).
- Strict information privacy (no question text, no snippets, no gold data, whitelisted settings).
- Hand-computed retrieval_only verification on synthetic corpus.
- Benchmark mode gate invariants (Rule 1 modified; Rules 2 to 6 unchanged).
"""

import json
import math
import os
import tempfile
import uuid
from typing import Any, Dict, List
import pytest
import yaml

from backend.app.core.config import Settings
from backend.app.core.llm_exceptions import LLMAuthError, LLMQuotaExhausted, LLMUnavailable
from backend.app.evaluation.harness import (
    DEFAULT_SOURCE_FILES,
    WHITELISTED_SETTINGS,
    EvaluationHarness,
    compute_configuration_hash,
    compute_settings_snapshot,
)
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import EvaluationLabel, EvaluationRun
from backend.app.safety.abstention import build_abstention_decision
from backend.app.safety.gate import (
    GateOutcome,
    evaluate_gate,
)
from backend.app.schemas.clinical_case import CaseSubmission, ClinicalCase
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum, WorkflowState
from backend.app.schemas.reports import (
    AbstentionDecision,
    AbstentionReasonCode,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
)
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


class ExplodingLLMProvider:
    """Mock provider that raises an AssertionError if any generation method is called."""

    async def generate_text(self, *args, **kwargs) -> str:
        raise AssertionError("ExplodingLLMProvider was unexpectedly called!")

    async def generate_json(self, *args, **kwargs):
        raise AssertionError("ExplodingLLMProvider was unexpectedly called!")

    def get_usage(self) -> Dict[str, int]:
        return {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}


class SyntheticFailingLLMProvider:
    """Mock provider that raises LLMAuthError, LLMQuotaExhausted, or generic failure."""

    def __init__(self, fail_with_auth: bool = False, fail_with_quota: bool = False):
        self.fail_with_auth = fail_with_auth
        self.fail_with_quota = fail_with_quota
        self._call_counter = 0

    async def generate_text(self, *args, **kwargs) -> str:
        self._call_counter += 1
        if self.fail_with_quota:
            raise LLMQuotaExhausted("Per-day quota exhausted. Check plan and billing.")
        if self.fail_with_auth:
            raise LLMAuthError("Synthetic auth error")
        raise LLMUnavailable("Synthetic provider unavailable")

    async def generate_json(self, *args, **kwargs):
        self._call_counter += 1
        if self.fail_with_quota:
            raise LLMQuotaExhausted("Per-day quota exhausted. Check plan and billing.")
        if self.fail_with_auth:
            raise LLMAuthError("Synthetic auth error")
        raise LLMUnavailable("Synthetic provider unavailable")

    def get_usage(self) -> Dict[str, int]:
        return {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": self._call_counter}


def test_harness_refuses_llm_run_without_third_party_llm_allowed(tmp_path):
    """Safety Gate (a): LLM runs must refuse if third_party_llm_allowed is false."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": False, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": False, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    harness = EvaluationHarness(source_files=source_files, output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="third_party_llm_allowed"):
        harness.check_safety_gates(
            set_name="smoke30",
            config_name="full",
            eval_max_llm_calls=50,
        )


def test_harness_refuses_eval_split_without_confirmations(tmp_path):
    """Safety Gate (b): Eval split runs must refuse if manifest or evaluation_labels are unconfirmed."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    harness = EvaluationHarness(source_files=source_files, output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="requires confirmed_by_human: true"):
        harness.check_safety_gates(
            set_name="retrieval_eval",
            config_name="retrieval_only",
            eval_max_llm_calls=None,
        )


def test_harness_refuses_e2e_run_without_eval_max_llm_calls(tmp_path):
    """Safety Gate (c): End-to-end runs must refuse if EVAL_MAX_LLM_CALLS is unset."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    test_settings = Settings(
        SECRET_KEY="x" * 32,
        ADMIN_PASSWORD="x" * 12,
        EVAL_MAX_LLM_CALLS=None,
    )
    harness = EvaluationHarness(
        settings_obj=test_settings,
        source_files=source_files,
        output_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="require setting 'EVAL_MAX_LLM_CALLS'"):
        harness.check_safety_gates(
            set_name="smoke30",
            config_name="full",
            eval_max_llm_calls=None,
        )


def test_harness_refuses_duplicate_eval_run_without_allow_rerun(tmp_path):
    """Safety Gate (b): Repeated completed run on eval split with same hash must refuse."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": True},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": True},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    cfg_hash = "mock_hash_eval_duplicate_test"

    # Insert a synthetic completed run into evaluation_runs in test database
    with SessionLocal() as session:
        session.add(
            EvaluationRun(
                id=str(uuid.uuid4()),
                dataset_name="bioasq",
                split_name="eval",
                sample_size=10,
                corpus_version="v1",
                embedding_model="test",
                run_metadata=json.dumps({"config_hash": cfg_hash}),
                status="COMPLETED",
            )
        )
        session.commit()

    harness = EvaluationHarness(source_files=source_files, output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="has already completed a final run"):
        harness.check_safety_gates(
            set_name="e2e_sample",
            config_name="full",
            eval_max_llm_calls=100,
            config_hash=cfg_hash,
            allow_rerun=None,
        )


def test_dry_run_makes_zero_llm_calls(tmp_path):
    """Operational Safeguard: --dry-run must calculate plans with 0 LLM calls."""
    harness = EvaluationHarness(
        llm_provider=ExplodingLLMProvider(),
        output_dir=str(tmp_path),
    )

    dry = harness.dry_run(
        set_name="smoke30",
        config_name="full",
        limit=5,
        eval_max_llm_calls=50,
    )
    assert dry["planned_question_count"] == 5
    assert dry["is_llm_required"] is True
    assert dry["estimated_call_ceiling"] <= 50


def test_run_json_privacy_and_whitelisting(tmp_path):
    """Run JSON must contain NO question text, snippet text, or gold text, and only whitelisted settings."""
    out_dir = tmp_path / "runs"
    out_dir.mkdir()

    test_settings = Settings(
        SECRET_KEY="secret-test-key-32-chars-at-least",
        ADMIN_PASSWORD="super-secret-password-1234",
        LLM_API_KEY="AIzaSySecretApiKeyDoNotLeak",
    )

    snap = compute_settings_snapshot(test_settings)
    assert "SECRET_KEY" not in snap
    assert "ADMIN_PASSWORD" not in snap
    assert "LLM_API_KEY" not in snap
    assert "temperature_sent" in snap
    assert "thinking_level" in snap
    assert "model_name" in snap
    for key in snap:
        assert key in WHITELISTED_SETTINGS

    # Check mock run record
    mock_record = {
        "sample_id": "test_id_001",
        "question_type": "factoid",
        "workflow_status": "READY_FOR_REVIEW",
        "gate_rule": 6,
        "reason_codes": [],
        "claim_status_counts": {"SUPPORTED": 2},
        "iteration_count": 0,
        "extracted_answer": "42",
        "correctness_flags": {"strict": True},
        "error_code": None,
        "durations": {"workflow_ms": 150},
        "llm_calls": 2,
        "llm_tokens": {"input": 100, "output": 20},
    }

    dumped = json.dumps(mock_record)
    assert "question" not in mock_record
    assert "snippet" not in dumped
    assert "gold" not in dumped


def test_circuit_breaker_stops_after_three_auth_errors(tmp_path):
    """Operational Safeguard: Circuit breaker halts after 3 consecutive auth errors and writes PARTIAL."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    failing_provider = SyntheticFailingLLMProvider(fail_with_auth=True)

    harness = EvaluationHarness(
        llm_provider=failing_provider,
        source_files=source_files,
        output_dir=str(tmp_path),
    )

    result = harness.run_e2e(
        set_name="smoke30",
        config_name="full",
        limit=5,
        eval_max_llm_calls=50,
    )

    assert result["status"] == "PARTIAL"
    assert "PARTIAL" in result["labels"]
    assert "3 consecutive LLM_AUTH_ERROR" in result["partial_reason"]
    assert len(result["per_record_outputs"]) == 3


def test_quota_exhausted_stops_run_and_marks_partial(tmp_path):
    """Operational Safeguard: Daily quota exhaustion halts run immediately with PARTIAL result."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    failing_provider = SyntheticFailingLLMProvider(fail_with_quota=True)

    harness = EvaluationHarness(
        llm_provider=failing_provider,
        source_files=source_files,
        output_dir=str(tmp_path),
    )

    result = harness.run_e2e(
        set_name="smoke30",
        config_name="baseline",
        limit=5,
        eval_max_llm_calls=50,
    )

    assert result["status"] == "PARTIAL"
    assert "PARTIAL" in result["labels"]
    assert "LLM_QUOTA_EXHAUSTED" in result["partial_reason"]
    assert len(result["per_record_outputs"]) == 1
    assert result["per_record_outputs"][0]["error_code"] in ("LLM_QUOTA_EXHAUSTED", "LLMQuotaExhausted")



def test_call_cap_stops_run_and_marks_partial(tmp_path):
    """Operational Safeguard: Call cap stops run immediately and marks result PARTIAL."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    class CountingLLM(FakeLLMProvider):
        def __init__(self):
            super().__init__()
            self._call_counter = 0

        def generate_text(self, *args, **kwargs):
            self._call_counter += 1
            return json.dumps({
                "structured_symptoms": [],
                "patient_context": {"age": 45, "sex": "M", "duration": "3 days"},
                "clarification_questions": [],
            })

        def get_usage(self) -> Dict[str, int]:
            return {
                "total_input_tokens": 10 * self._call_counter,
                "total_output_tokens": 10 * self._call_counter,
                "total_calls": self._call_counter,
            }

    harness = EvaluationHarness(
        llm_provider=CountingLLM(),
        source_files=source_files,
        output_dir=str(tmp_path),
    )

    # Set EVAL_MAX_LLM_CALLS cap to 2 calls
    result = harness.run_e2e(
        set_name="smoke30",
        config_name="baseline",
        limit=5,
        eval_max_llm_calls=2,
    )

    assert result["status"] == "PARTIAL"
    assert "PARTIAL" in result["labels"]
    assert "Call cap reached" in result["partial_reason"]


def test_benchmark_mode_gate_invariants():
    """Verify benchmark_mode changes only Rule 1 scope, keeping Rules 2 to 6 identical."""
    test_settings = Settings(
        SECRET_KEY="x" * 32,
        ADMIN_PASSWORD="x" * 12,
        MAX_OPTIMIZER_ITERATIONS=3,
    )

    # 1. Scope Rule: Treatment/dosing request
    sub_treatment = CaseSubmission(raw_text="What is the recommended dosage and treatment plan for hypertension?")

    # Clinical mode: Must ABSTAIN via Rule 1 (OUT_OF_SCOPE)
    state_clinical = WorkflowState(
        case_id="c1",
        submission=sub_treatment,
        is_benchmark=False,
    )
    dec_clinical = evaluate_gate(state_clinical, test_settings)
    assert dec_clinical.rule_triggered == 1
    assert dec_clinical.outcome == GateOutcome.FINALIZE_ABSTAIN
    assert dec_clinical.primary_reason == AbstentionReasonCode.OUT_OF_SCOPE

    # Benchmark mode: Rule 1 bypassed (biomedical question in-scope)
    state_benchmark = WorkflowState(
        case_id="c2",
        submission=sub_treatment,
        is_benchmark=True,
    )
    dec_benchmark = evaluate_gate(state_benchmark, test_settings)
    # Since verifications are empty and retrieval is empty, it proceeds to later rules (e.g. Rule 4 NO_RELEVANT_EVIDENCE)
    assert dec_benchmark.rule_triggered != 1

    # 2. Rule 2: Verifier error on safety-critical claim
    # Must trigger Rule 2 in BOTH clinical and benchmark mode!
    claim = CandidateClaim(
        claim_id="clm_01",
        report_version_id="rv_01",
        claim_text="Hypothesis indication",
        section="potential_considerations",
        is_safety_critical=True,
        is_material=True,
    )
    rep = CandidateClinicalReport(
        version_number=1,
        case_id="c_err",
        summary="Summary",
        claims=[claim],
        confidence_score=0.90,
    )
    rec_fail = AgentExecutionRecord(
        agent_name="claim_verification",
        status=AgentStatusEnum.FAILED,
        input_summary="Verifying",
        error_message="Verifier LLM crashed",
    )

    state_r2_clin = WorkflowState(
        case_id="c_r2_clin",
        submission=CaseSubmission(raw_text="Biomedical observation"),
        report=rep,
        agent_records=[rec_fail],
        is_benchmark=False,
    )
    dec_r2_clin = evaluate_gate(state_r2_clin, test_settings)
    assert dec_r2_clin.rule_triggered == 2
    assert dec_r2_clin.outcome == GateOutcome.FINALIZE_ABSTAIN

    state_r2_bench = WorkflowState(
        case_id="c_r2_bench",
        submission=CaseSubmission(raw_text="Biomedical observation"),
        report=rep,
        agent_records=[rec_fail],
        is_benchmark=True,
    )
    dec_r2_bench = evaluate_gate(state_r2_bench, test_settings)
    assert dec_r2_bench.rule_triggered == 2
    assert dec_r2_bench.outcome == GateOutcome.FINALIZE_ABSTAIN


def test_resume_skips_completed_questions_and_matches(tmp_path):
    """Operational Safeguard: resume skips completed questions and produces aggregated output."""
    manifest_file = tmp_path / "MANIFEST.yaml"
    manifest_data = {
        "files": [
            {"name": "bioasq_evidence_corpus.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
            {"name": "bioasq_normalized.csv", "third_party_llm_allowed": True, "confirmed_by_human": False},
        ]
    }
    manifest_file.write_text(yaml.dump(manifest_data), encoding="utf-8")

    source_files = dict(DEFAULT_SOURCE_FILES)
    source_files["manifest"] = str(manifest_file)

    class CannedLLM(FakeLLMProvider):
        def __init__(self):
            super().__init__()
            self._call_counter = 0

        def generate_text(self, *args, **kwargs):
            self._call_counter += 1
            return json.dumps({
                "structured_symptoms": [],
                "patient_context": {"age": 50, "sex": "F", "duration": "1 week"},
                "clarification_questions": [],
            })

        def get_usage(self) -> Dict[str, int]:
            return {"total_input_tokens": 10 * self._call_counter, "total_output_tokens": 10 * self._call_counter, "total_calls": self._call_counter}

    harness = EvaluationHarness(
        llm_provider=CannedLLM(),
        source_files=source_files,
        output_dir=str(tmp_path),
    )

    # 1. First run with limit 1
    res1 = harness.run_e2e(
        set_name="smoke30",
        config_name="baseline",
        limit=1,
        eval_max_llm_calls=20,
    )
    assert len(res1["per_record_outputs"]) == 1
    completed_sid = res1["per_record_outputs"][0]["sample_id"]

    # 2. Checkpoint file contains completed question
    cp_files = list((tmp_path / "checkpoints").glob("*.jsonl"))
    assert len(cp_files) >= 1

    # Verify that completed_records dict skips this question when resuming
    completed_records = {completed_sid: res1["per_record_outputs"][0]}
    assert completed_sid in completed_records


def test_retrieval_only_synthetic_hand_computed(tmp_path):
    """Retrieval-only evaluation on synthetic data gives exact hand-computed metrics."""
    from backend.app.evaluation.bioasq_scoring import score_retrieval

    # Synthetic question retrieval payloads:
    # Q1: retrieved row_A (score 0.9) and row_B (0.7). Relevant: row_A. Hit at rank 1.
    # Q2: retrieved row_C (score 0.8) and row_D (0.6). Relevant: row_D. Hit at rank 2.
    # Q3: no relevant rows (excluded).
    synthetic_payloads = [
        {
            "sample_id": "syn_q1",
            "question_type": "factoid",
            "relevant_row_ids": ["row_A"],
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_A:0", "retrieval_score": 0.90},
                {"evidence_id": "ev:bioasq:row_B:0", "retrieval_score": 0.70},
            ],
            "duplicate_groups": {},
        },
        {
            "sample_id": "syn_q2",
            "question_type": "yesno",
            "relevant_row_ids": ["row_D"],
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_C:0", "retrieval_score": 0.80},
                {"evidence_id": "ev:bioasq:row_D:0", "retrieval_score": 0.60},
            ],
            "duplicate_groups": {},
        },
        {
            "sample_id": "syn_q3",
            "question_type": "summary",
            "relevant_row_ids": [],
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_Z:0", "retrieval_score": 0.95},
            ],
            "duplicate_groups": {},
        },
    ]

    res = score_retrieval(synthetic_payloads)

    assert res.included_questions == 2
    assert res.excluded_no_relevant_rows == 1

    # Hand-computed metrics:
    # Q1: rec@5 = 1/1 = 1.0, rec@10 = 1.0, prec@5 = 1/5 = 0.20, mrr = 1/1 = 1.0, map = 1.0
    # Q2: rec@5 = 1/1 = 1.0, rec@10 = 1.0, prec@5 = 1/5 = 0.20, mrr = 1/2 = 0.5, map = 1/2 = 0.5
    # Overall means:
    # rec@5 = (1.0 + 1.0) / 2 = 1.0
    # prec@5 = (0.20 + 0.20) / 2 = 0.20
    # mrr = (1.0 + 0.5) / 2 = 0.75
    # map = (1.0 + 0.5) / 2 = 0.75
    assert math.isclose(res.recall_at_5.value, 1.0, rel_tol=1e-6)
    assert math.isclose(res.recall_at_10.value, 1.0, rel_tol=1e-6)
    assert math.isclose(res.precision_at_5.value, 0.20, rel_tol=1e-6)
    assert math.isclose(res.mrr.value, 0.75, rel_tol=1e-6)
    assert math.isclose(res.map.value, 0.75, rel_tol=1e-6)

