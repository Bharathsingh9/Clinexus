"""Adaptive Optimizer (Agent 5 - AGT-5, App A3).

Specification Section 4.5, Appendix A3:
- Input: CandidateClinicalReport, iteration_history, confidence_score, findings, gate signals.
- Output: OptimizationIteration (or AgentExecutionRecord if FAILED with LLM_UNAVAILABLE).
- Rule-based deterministic routing:
    * REFINE_RETRIEVAL: on empty/low-similarity retrieval or NO_RELEVANT_EVIDENCE.
    * FETCH_EXTERNAL: on degraded scanner or missing external support.
    * REVISE_REPORT: on unsupported material claims with existing evidence.
    * REQUEST_CLARIFICATION: on unresolved blocking patient findings.
    * STOP: when confidence threshold (0.65) met, budget exhausted (3 iterations), or NO_PROGRESS.
- Stagnation guard (NO_PROGRESS rule): stops when consecutive iterations leave non-SUPPORTED claims or confidence stagnant.
- Hard cap in code: MAX_OPTIMIZER_ITERATIONS = 3.
- The LLM writes ONLY instruction text and query hints; it never routes or relaxes a criterion.
- Invariant: with no provider configured, returns FAILED with error_code LLM_UNAVAILABLE.
"""

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.schemas.executions import (
    AgentExecutionRecord,
    AgentStatusEnum,
    OptimizationIteration,
)
from backend.app.schemas.reports import CandidateClinicalReport

logger = logging.getLogger(__name__)


class AdaptiveOptimizer:
    """Agent 5: Adaptive feedback router and iteration manager."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.model_name = model_name or settings.LLM_MODEL_OPTIMIZER

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Call provider handling both sync and async signatures."""
        if self.llm_provider is None:
            raise ValueError("LLM provider is not configured.")
        method = getattr(self.llm_provider, "generate_text", None) or getattr(
            self.llm_provider, "generate", None
        )
        if not callable(method):
            raise AttributeError("LLM provider missing generate_text or generate method")

        try:
            res = method(
                prompt=prompt,
                system_instruction=system,
                model=self.model_name,
                temperature=0.0,
            )
        except TypeError:
            res = method(
                prompt=prompt,
                system=system,
                model=self.model_name,
                temperature=0.0,
            )

        if inspect.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, res).result()
            else:
                return asyncio.run(res)
        return str(res)

    def _determine_route_by_rules(
        self,
        confidence_score: float,
        iteration_number: int,
        iteration_history: List[OptimizationIteration],
        has_blocking_clarification: bool,
        scanner_degraded: bool,
        unsupported_claims_count: int,
        abstention_reasons: Optional[List[str]],
    ) -> Tuple[str, str]:
        """Deterministic rule-based routing of findings to operations (SPEC 4.5)."""
        # Rule 1: Stagnation guard (NO_PROGRESS rule per SPEC 4.5)
        if len(iteration_history) >= 2:
            prev1 = iteration_history[-1]
            prev2 = iteration_history[-2]
            # Stagnant confidence or repeated failed operation
            if prev1.node_selected == prev2.node_selected and abs(prev1.confidence_before - prev2.confidence_before) < 1e-4:
                return "STOP", "NO_PROGRESS: Consecutive iterations produced identical operation with no confidence improvement."

        # Rule 2: Hard iteration cap (MAX_OPTIMIZER_ITERATIONS = 3)
        if iteration_number >= settings.MAX_OPTIMIZER_ITERATIONS:
            return "STOP", "BUDGET_EXHAUSTED: Maximum optimizer iterations (3) reached."

        # Rule 3: Blocking patient information missing
        if has_blocking_clarification:
            return "REQUEST_CLARIFICATION", "Unresolved blocking patient finding detected."

        # Rule 4: No relevant evidence or empty retrieval
        reasons = abstention_reasons or []
        if "NO_RELEVANT_EVIDENCE" in reasons or "LOW_SIMILARITY" in reasons:
            return "REFINE_RETRIEVAL", "Insufficient local retrieval signal; refining query synthesis."

        # Rule 5: External scanner failure or degradation
        if scanner_degraded:
            return "FETCH_EXTERNAL", "External literature scanner degraded; retrying literature fetch."

        # Rule 6: Unsupported material claims with existing evidence
        if unsupported_claims_count > 0:
            return "REVISE_REPORT", f"Found {unsupported_claims_count} unverified material claims; revising report synthesis."

        # Rule 7: Confidence threshold satisfied
        if confidence_score >= settings.CONFIDENCE_THRESHOLD:
            return "STOP", f"CONDITION_MET: Confidence score {confidence_score:.3f} meets threshold {settings.CONFIDENCE_THRESHOLD}."

        # Fallback default: Refine retrieval if confidence remains below threshold
        return "REFINE_RETRIEVAL", f"Confidence score {confidence_score:.3f} below threshold; expanding retrieval."

    def run(
        self,
        report: CandidateClinicalReport,
        iteration_history: Optional[List[OptimizationIteration]] = None,
        confidence_score: Optional[float] = None,
        has_blocking_clarification: bool = False,
        scanner_degraded: bool = False,
        unsupported_claims_count: int = 0,
        abstention_reasons: Optional[List[str]] = None,
        execution_id: Optional[str] = None,
    ) -> Union[OptimizationIteration, AgentExecutionRecord]:
        """Execute Agent 5 optimizer routing and return OptimizationIteration record."""
        # Invariant: No silent fallbacks. Unconfigured LLM provider returns FAILED(LLM_UNAVAILABLE).
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="adaptive_optimizer",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured (LLM_PROVIDER unset or 'none'). Real LLM calls are blocked.",
                input_summary=f"Report version {report.report_version_id} (confidence={report.confidence_score})",
                output_summary=None,
            )

        history = iteration_history or []
        current_conf = confidence_score if confidence_score is not None else report.confidence_score
        iteration_num = min(settings.MAX_OPTIMIZER_ITERATIONS, len(history) + 1)
        exec_id = execution_id or str(uuid.uuid4())

        # Step 1: Deterministic rule-based routing
        selected_op, rule_reason = self._determine_route_by_rules(
            confidence_score=current_conf,
            iteration_number=iteration_num,
            iteration_history=history,
            has_blocking_clarification=has_blocking_clarification,
            scanner_degraded=scanner_degraded,
            unsupported_claims_count=unsupported_claims_count,
            abstention_reasons=abstention_reasons,
        )

        # Step 2: LLM writes ONLY instruction text and query hints (SPEC 4.5)
        system_prompt = (
            "You are the Adaptive Optimizer agent in Clinexus. Given the selected operation and findings, "
            "write brief, targeted instruction text and query hints for the next agent iteration. "
            "You do NOT choose the operation route."
        )
        prompt = (
            f"SELECTED OPERATION: {selected_op}\n"
            f"ROUTING REASON: {rule_reason}\n"
            f"CURRENT CONFIDENCE: {current_conf:.3f}\n"
            f"REPORT SUMMARY: {report.summary}\n\n"
            f"Provide actionable instruction text in one concise sentence:"
        )

        try:
            instruction_text = self._invoke_llm(prompt=prompt, system=system_prompt).strip()
            action_desc = f"{rule_reason} Instruction: {instruction_text}"
        except Exception as exc:
            logger.warning("LLM instruction writing failed (%s). Using rule reason.", exc)
            action_desc = rule_reason

        return OptimizationIteration(
            iteration_id=str(uuid.uuid4()),
            execution_id=exec_id,
            iteration_number=iteration_num,
            node_selected=selected_op,
            confidence_before=round(current_conf, 4),
            confidence_after=None,
            action_taken=action_desc[:500],
            created_at=datetime.now(timezone.utc),
        )
