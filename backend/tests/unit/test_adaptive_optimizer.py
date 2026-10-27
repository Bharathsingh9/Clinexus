"""Unit tests for Adaptive Optimizer (Agent 5 - AGT-5, App A3).

Specification Section 4.5, Appendix A3:
- Without LLM provider, returns FAILED with LLM_UNAVAILABLE.
- Hard iteration cap (MAX_OPTIMIZER_ITERATIONS=3) produces STOP.
- Stagnation guard (NO_PROGRESS rule) stops on repeated stagnant operations.
- Rule-based routing to REFINE_RETRIEVAL, FETCH_EXTERNAL, REVISE_REPORT, REQUEST_CLARIFICATION.
- Stops with CONDITION_MET when confidence threshold (0.65) met.
- LLM used only for instruction text; never chooses route.
"""

from datetime import datetime, timezone
import pytest
from backend.app.agents.adaptive_optimizer import AdaptiveOptimizer
from backend.app.schemas.executions import (
    AgentStatusEnum,
    OptimizationIteration,
)
from backend.app.schemas.reports import CandidateClinicalReport
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider


def make_test_report(confidence_score: float = 0.50) -> CandidateClinicalReport:
    return CandidateClinicalReport(
        version_number=1,
        case_id="case-101",
        summary="Clinical differential report summary.",
        sections={},
        claims=[],
        confidence_score=confidence_score,
    )


def test_optimizer_returns_llm_unavailable_without_provider():
    """Without an LLM provider, Agent 5 must return FAILED with LLM_UNAVAILABLE."""
    optimizer = AdaptiveOptimizer(llm_provider=None)
    report = make_test_report(0.50)
    res = optimizer.run(report)

    assert res.status == AgentStatusEnum.FAILED
    assert res.error_code == "LLM_UNAVAILABLE"


def test_optimizer_stops_when_confidence_threshold_met():
    """When confidence is >= 0.65 and no blocking conditions exist, route is STOP."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider(default_text="Target met."))
    report = make_test_report(confidence_score=0.72)
    iter_record = optimizer.run(report)

    assert iter_record.node_selected == "STOP"
    assert "CONDITION_MET" in iter_record.action_taken


def test_optimizer_hard_cap_produces_stop():
    """When iteration count reaches MAX_OPTIMIZER_ITERATIONS (3), route is STOP."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider())
    report = make_test_report(0.40)
    # Already 2 previous iterations
    history = [
        OptimizationIteration(
            execution_id="e-1",
            iteration_number=1,
            node_selected="REFINE_RETRIEVAL",
            confidence_before=0.35,
            action_taken="Initial refinement",
        ),
        OptimizationIteration(
            execution_id="e-1",
            iteration_number=2,
            node_selected="FETCH_EXTERNAL",
            confidence_before=0.40,
            action_taken="External scan",
        ),
    ]
    # Iteration 3 reaches the hard cap
    iter_record = optimizer.run(report, iteration_history=history)

    assert iter_record.node_selected == "STOP"
    assert "BUDGET_EXHAUSTED" in iter_record.action_taken
    assert iter_record.iteration_number == 3


def test_optimizer_stagnation_guard_produces_stop():
    """When consecutive iterations repeat the same operation with no progress, stagnation guard stops."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider())
    report = make_test_report(0.45)
    history = [
        OptimizationIteration(
            execution_id="e-1",
            iteration_number=1,
            node_selected="REFINE_RETRIEVAL",
            confidence_before=0.45,
            action_taken="Expanded queries",
        ),
        OptimizationIteration(
            execution_id="e-1",
            iteration_number=2,
            node_selected="REFINE_RETRIEVAL",
            confidence_before=0.45,  # Exactly same confidence
            action_taken="Expanded queries again",
        ),
    ]
    iter_record = optimizer.run(report, iteration_history=history)

    assert iter_record.node_selected == "STOP"
    assert "NO_PROGRESS" in iter_record.action_taken


def test_optimizer_routes_to_clarification_on_blocking_finding():
    """Blocking patient findings take precedence and route to REQUEST_CLARIFICATION."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider(default_text="Clarify age and duration."))
    report = make_test_report(0.40)
    iter_record = optimizer.run(report, has_blocking_clarification=True)

    assert iter_record.node_selected == "REQUEST_CLARIFICATION"
    assert "Unresolved blocking patient finding" in iter_record.action_taken


def test_optimizer_routes_to_fetch_external_on_scanner_degraded():
    """Degraded scanner routes to FETCH_EXTERNAL."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider(default_text="Retry PubMed query."))
    report = make_test_report(0.45)
    iter_record = optimizer.run(report, scanner_degraded=True)

    assert iter_record.node_selected == "FETCH_EXTERNAL"


def test_optimizer_routes_to_revise_report_on_unsupported_claims():
    """Unsupported claims with existing evidence route to REVISE_REPORT."""
    optimizer = AdaptiveOptimizer(llm_provider=FakeLLMProvider(default_text="Revise claim 2."))
    report = make_test_report(0.50)
    iter_record = optimizer.run(report, unsupported_claims_count=2)

    assert iter_record.node_selected == "REVISE_REPORT"
