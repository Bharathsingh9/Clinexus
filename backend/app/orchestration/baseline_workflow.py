"""Baseline Multi-Agent Workflow Orchestration (LangGraph).

Specification Section 2, 3, Section 4, Appendix A1, A2, A3 (Req ID: WKF-1, DB-3).
Paper-faithful baseline pipeline without extensions A-D.

Nodes (strictly sequential, deterministic routing):
1. validate_case: validates raw CaseSubmission; invalid input ends the run FAILED.
2. clinical_text_clarifier: Agent 1 extracts clinical findings, demographics, and questions.
3. clarification_check: halts with AWAITING_CLARIFICATION if any blocking clarification question exists.
4. symptom_rag_analyzer: Agent 2 retrieves evidence from Qdrant and generates candidate hypotheses.
5. evidence_based_scanner: Agent 3 fetches external literature from PubMed (DEGRADED if unavailable).
6. clinical_data_fusion: Agent 4 synthesizes evidence into CandidateClinicalReport.
7. baseline decision: evaluates confidence via pure compute_confidence(is_baseline=True).
   If >= CONFIDENCE_THRESHOLD -> persist_and_finish.
   Else if iterations remain -> adaptive_optimizer.
   Else (cap reached) -> persist_and_finish with "BASELINE: budget reached, not verified".
8. adaptive_optimizer: Agent 5 deterministic routing on structural findings.
9. persist_and_finish: persists WorkflowResult and updates execution status.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from langgraph.graph import END, START, StateGraph

from backend.app.agents.adaptive_optimizer import AdaptiveOptimizer
from backend.app.agents.clinical_data_fusion import ClinicalDataFusion
from backend.app.agents.clinical_text_clarifier import ClinicalTextClarifier
from backend.app.agents.evidence_based_scanner import EvidenceBasedScanner
from backend.app.agents.symptom_rag_analyzer import SymptomRAGAnalyzer
from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.evidence_sources.ncbi_client import NCBIClientProtocol
from backend.app.persistence.execution_repository import ExecutionRepository
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper
from backend.app.safety.confidence import compute_confidence
from backend.app.schemas.clinical_case import (
    CaseSubmission,
    ClarificationRequest,
    ClinicalCase,
)
from backend.app.schemas.evidence import (
    ExternalEvidenceSet,
    QueryStatusEnum,
    RetrievedEvidenceSet,
)
from backend.app.schemas.executions import (
    AgentExecutionRecord,
    AgentStatusEnum,
    OptimizationIteration,
    WorkflowResult,
    WorkflowState,
)
from backend.app.schemas.reports import (
    CandidateClinicalReport,
    ReportStatusEnum,
)

logger = logging.getLogger(__name__)


def compute_payload_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of structured or text data."""
    if data is None:
        return hashlib.sha256(b"null").hexdigest()
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    if isinstance(data, str):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    if hasattr(data, "model_dump_json"):
        return hashlib.sha256(data.model_dump_json().encode("utf-8")).hexdigest()
    if isinstance(data, dict):
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


class BaselineWorkflowOrchestrator:
    """Orchestrates the 5-agent baseline clinical workflow with LangGraph."""

    def __init__(
        self,
        clarifier: Optional[ClinicalTextClarifier] = None,
        rag_analyzer: Optional[SymptomRAGAnalyzer] = None,
        scanner: Optional[EvidenceBasedScanner] = None,
        fusion: Optional[ClinicalDataFusion] = None,
        optimizer: Optional[AdaptiveOptimizer] = None,
        repository: Optional[ExecutionRepository] = None,
        llm_provider: Optional[LLMProvider] = None,
        ncbi_client: Optional[NCBIClientProtocol] = None,
        qdrant_wrapper: Optional[QdrantWrapper] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.repository = repository or ExecutionRepository()
        self.embedding_service = embedding_service or EmbeddingService()

        self.clarifier = clarifier or ClinicalTextClarifier(
            llm_provider=llm_provider,
        )
        self.rag_analyzer = rag_analyzer or SymptomRAGAnalyzer(
            qdrant_wrapper=qdrant_wrapper,
            embedding_service=self.embedding_service,
            llm_provider=llm_provider,
        )
        self.scanner = scanner or EvidenceBasedScanner(
            ncbi_client=ncbi_client,
            llm_provider=llm_provider,
            embedding_service=self.embedding_service,
        )
        self.fusion = fusion or ClinicalDataFusion(
            llm_provider=llm_provider,
        )
        self.optimizer = optimizer or AdaptiveOptimizer(
            llm_provider=llm_provider,
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph StateGraph with deterministic conditional routing."""
        builder = StateGraph(WorkflowState)

        # Register nodes
        builder.add_node("validate_case", self.node_validate_case)
        builder.add_node("clinical_text_clarifier", self.node_clinical_text_clarifier)
        builder.add_node("clarification_check", self.node_clarification_check)
        builder.add_node("symptom_rag_analyzer", self.node_symptom_rag_analyzer)
        builder.add_node("evidence_based_scanner", self.node_evidence_based_scanner)
        builder.add_node("clinical_data_fusion", self.node_clinical_data_fusion)
        builder.add_node("adaptive_optimizer", self.node_adaptive_optimizer)
        builder.add_node("persist_and_finish", self.node_persist_and_finish)

        # Wire edges with deterministic conditional routing
        builder.add_edge(START, "validate_case")

        builder.add_conditional_edges(
            "validate_case",
            self.route_validate_case,
            {
                "continue": "clinical_text_clarifier",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "clinical_text_clarifier",
            self.route_after_clarifier,
            {
                "continue": "clarification_check",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "clarification_check",
            self.route_after_clarification_check,
            {
                "continue": "symptom_rag_analyzer",
                "awaiting_clarification": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "symptom_rag_analyzer",
            self.route_after_rag,
            {
                "continue": "evidence_based_scanner",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "evidence_based_scanner",
            self.route_after_scanner,
            {
                "continue": "clinical_data_fusion",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "clinical_data_fusion",
            self.route_baseline_decision,
            {
                "adaptive_optimizer": "adaptive_optimizer",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_conditional_edges(
            "adaptive_optimizer",
            self.route_after_optimizer,
            {
                "symptom_rag_analyzer": "symptom_rag_analyzer",
                "evidence_based_scanner": "evidence_based_scanner",
                "clinical_data_fusion": "clinical_data_fusion",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_edge("persist_and_finish", END)

        return builder.compile()

    # -------------------------------------------------------------------------
    # Routing Functions (100% deterministic code, never LLM)
    # -------------------------------------------------------------------------

    @staticmethod
    def route_validate_case(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"
        return "continue"

    @staticmethod
    def route_after_clarifier(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"
        return "continue"

    @staticmethod
    def route_after_clarification_check(state: WorkflowState) -> str:
        if state.workflow_status == "AWAITING_CLARIFICATION":
            return "awaiting_clarification"
        return "continue"

    @staticmethod
    def route_after_rag(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"
        return "continue"

    @staticmethod
    def route_after_scanner(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"
        return "continue"

    @staticmethod
    def route_baseline_decision(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"

        if state.report is None:
            return "persist_and_finish"

        # Baseline decision: check confidence threshold (Appendix A1, A3)
        if state.report.confidence_score >= settings.CONFIDENCE_THRESHOLD:
            return "persist_and_finish"

        # If iterations remain, loop to optimizer; else stop
        if state.current_iteration < settings.MAX_OPTIMIZER_ITERATIONS:
            return "adaptive_optimizer"

        # Budget exhausted cap reached
        return "persist_and_finish"

    @staticmethod
    def route_after_optimizer(state: WorkflowState) -> str:
        if state.workflow_status in ("FAILED", "AWAITING_CLARIFICATION"):
            return "persist_and_finish"

        if state.iteration_cap_reached or not state.iterations:
            return "persist_and_finish"

        last_op = state.iterations[-1].node_selected
        if last_op == "REFINE_RETRIEVAL":
            return "symptom_rag_analyzer"
        elif last_op == "FETCH_EXTERNAL":
            return "evidence_based_scanner"
        elif last_op == "REVISE_REPORT":
            return "clinical_data_fusion"
        elif last_op == "REQUEST_CLARIFICATION":
            return "persist_and_finish"
        else:
            return "persist_and_finish"

    # -------------------------------------------------------------------------
    # Graph Node Implementations
    # -------------------------------------------------------------------------

    def node_validate_case(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 1: Validate CaseSubmission. Invalid input ends run FAILED immediately."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        raw_input = state.submission or state.submission_payload
        in_hash = compute_payload_hash(raw_input)

        # Record start before starting
        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="validate_case",
            input_summary="Validating submitted clinical narrative text",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            if state.submission is None:
                if not isinstance(state.submission_payload, dict):
                    raise ValueError("Submission payload is not a valid dictionary.")
                submission = CaseSubmission.model_validate(state.submission_payload)
            else:
                submission = state.submission

            if not submission.raw_text or not submission.raw_text.strip():
                raise ValueError("Case narrative raw_text is empty.")

            out_hash = compute_payload_hash(submission)
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=f"Case submission validated (len={len(submission.raw_text)})",
                output_hash=out_hash,
            )

            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="validate_case",
                status=AgentStatusEnum.COMPLETED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=out_hash,
                input_summary="Validating submitted clinical narrative text",
                output_summary=f"Case submission validated (len={len(submission.raw_text)})",
            )
            return {
                "submission": submission,
                "current_node": "validate_case",
                "agent_records": state.agent_records + [rec],
            }

        except Exception as exc:
            err_msg = f"Validation failed: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="INVALID_SUBMISSION",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="validate_case",
                status=AgentStatusEnum.FAILED,
                error_code="INVALID_SUBMISSION",
                error_message=err_msg,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                input_summary="Validating submitted clinical narrative text",
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "INVALID_SUBMISSION",
                "error_message": err_msg,
                "current_node": "validate_case",
                "agent_records": state.agent_records + [rec],
            }

    def node_clinical_text_clarifier(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 2: Clinical Text Clarifier (Agent 1). Extracts findings and questions."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.submission)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="clinical_text_clarifier",
            input_summary=f"Extracting clinical findings for submission {state.submission.submission_id}",
            input_hash=in_hash,
            record_id=record_id,
        )

        result = self.clarifier.run(state.submission)

        if isinstance(result, AgentExecutionRecord):
            # Clarifier is a REQUIRED node; failure halts the workflow
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code=result.error_code or "LLM_UNAVAILABLE",
                error_message=result.error_message,
            )
            result.record_id = record_id
            result.input_hash = in_hash
            return {
                "workflow_status": "FAILED",
                "error_code": result.error_code or "LLM_UNAVAILABLE",
                "error_message": result.error_message,
                "current_node": "clinical_text_clarifier",
                "agent_records": state.agent_records + [result],
            }

        case: ClinicalCase = result
        case.case_id = state.case_id
        out_hash = compute_payload_hash(case)

        # Persist clinical case and findings in database
        self.repository.create_case(
            case_id=state.case_id,
            raw_text_hash=case.raw_text_hash,
            chief_complaint=case.chief_complaint,
            age=case.age,
            age_missing_reason=case.age_missing_reason,
            sex=case.sex,
            symptom_duration=case.symptom_duration,
            duration_missing_reason=case.duration_missing_reason,
            status="ANALYZING",
        )
        self.repository.save_case_findings(state.case_id, case.structured_symptoms)

        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=AgentStatusEnum.COMPLETED.value,
            output_summary=f"Extracted {len(case.structured_symptoms)} symptoms, {len(case.clarification_questions)} questions",
            output_hash=out_hash,
        )

        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="clinical_text_clarifier",
            status=AgentStatusEnum.COMPLETED,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary=f"Extracting clinical findings for submission {state.submission.submission_id}",
            output_summary=f"Extracted {len(case.structured_symptoms)} symptoms, {len(case.clarification_questions)} questions",
        )

        blocking = [q for q in case.clarification_questions if q.is_blocking]
        non_blocking = [q for q in case.clarification_questions if not q.is_blocking]

        return {
            "case_id": case.case_id,
            "case": case,
            "blocking_clarifications": blocking,
            "non_blocking_clarifications": non_blocking,
            "clarification_needed": len(case.clarification_questions) > 0,
            "current_node": "clinical_text_clarifier",
            "agent_records": state.agent_records + [rec],
        }

    def node_clarification_check(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 3: Deterministic check for blocking clarification questions.

        Halts before retrieval with status AWAITING_CLARIFICATION if blocking questions exist.
        Non-blocking questions continue to retrieval and report.
        """
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.case)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="clarification_check",
            input_summary="Evaluating blocking vs non-blocking clarification needs",
            input_hash=in_hash,
            record_id=record_id,
        )

        has_blocking = len(state.blocking_clarifications) > 0 and not state.is_benchmark

        if has_blocking:
            summary = f"Halting before retrieval: {len(state.blocking_clarifications)} blocking clarification question(s)."
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=summary,
                output_hash=compute_payload_hash({"status": "AWAITING_CLARIFICATION"}),
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="clarification_check",
                status=AgentStatusEnum.COMPLETED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash({"status": "AWAITING_CLARIFICATION"}),
                input_summary="Evaluating blocking vs non-blocking clarification needs",
                output_summary=summary,
            )
            return {
                "workflow_status": "AWAITING_CLARIFICATION",
                "current_node": "clarification_check",
                "agent_records": state.agent_records + [rec],
            }

        summary = f"Passed clarification check: {len(state.non_blocking_clarifications)} non-blocking question(s) noted."
        out_hash = compute_payload_hash({"status": "PROCEED", "non_blocking": len(state.non_blocking_clarifications)})
        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=AgentStatusEnum.COMPLETED.value,
            output_summary=summary,
            output_hash=out_hash,
        )
        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="clarification_check",
            status=AgentStatusEnum.COMPLETED,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary="Evaluating blocking vs non-blocking clarification needs",
            output_summary=summary,
        )
        return {
            "current_node": "clarification_check",
            "agent_records": state.agent_records + [rec],
        }

    def node_symptom_rag_analyzer(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 4: Symptom RAG Analyzer (Agent 2). Vector retrieval and candidate hypotheses."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.case)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="symptom_rag_analyzer",
            input_summary=f"Retrieving evidence for case {state.case_id} (iteration {state.current_iteration})",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            result = self.rag_analyzer.run(
                case=state.case,
                alternative_hints=state.gap_hints,
                benchmark_mode=state.is_benchmark,
            )
        except Exception as exc:
            err_msg = f"RAG Analyzer failure: {exc}"
            err_code = "QDRANT_UNAVAILABLE" if "qdrant" in str(exc).lower() else "RETRIEVAL_ERROR"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code=err_code,
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="symptom_rag_analyzer",
                status=AgentStatusEnum.FAILED,
                error_code=err_code,
                error_message=err_msg,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                input_summary=f"Retrieving evidence for case {state.case_id}",
            )
            return {
                "workflow_status": "FAILED",
                "error_code": err_code,
                "error_message": err_msg,
                "current_node": "symptom_rag_analyzer",
                "agent_records": state.agent_records + [rec],
            }

        if isinstance(result, AgentExecutionRecord):
            # RAG Analyzer is a REQUIRED node; failure halts the workflow
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code=result.error_code or "LLM_UNAVAILABLE",
                error_message=result.error_message,
            )
            result.record_id = record_id
            result.input_hash = in_hash
            return {
                "workflow_status": "FAILED",
                "error_code": result.error_code or "LLM_UNAVAILABLE",
                "error_message": result.error_message,
                "current_node": "symptom_rag_analyzer",
                "agent_records": state.agent_records + [result],
            }

        retrieved: RetrievedEvidenceSet = result
        out_hash = compute_payload_hash(retrieved)

        # Check if vector DB is unreachable (infrastructure failure -> run ends FAILED)
        if retrieved.insufficient and retrieved.reason_code == "VECTOR_DB_UNAVAILABLE":
            err_msg = "Qdrant vector database is unreachable."
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="QDRANT_UNAVAILABLE",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="symptom_rag_analyzer",
                status=AgentStatusEnum.FAILED,
                error_code="QDRANT_UNAVAILABLE",
                error_message=err_msg,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                input_summary=f"Retrieving evidence for case {state.case_id}",
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "QDRANT_UNAVAILABLE",
                "error_message": err_msg,
                "current_node": "symptom_rag_analyzer",
                "agent_records": state.agent_records + [rec],
            }

        # Save retrieval runs to PostgreSQL
        corpus_ver = retrieved.results[0].corpus_version if retrieved.results else "v1"
        embed_mdl = retrieved.results[0].embedding_model if retrieved.results else self.embedding_service.model_name
        self.repository.save_retrieval_run(
            execution_id=state.execution_id,
            query_id=retrieved.query_id,
            query_text=state.case.chief_complaint if state.case else "Symptom RAG retrieval",
            query_type="SYMPTOM_DENSE",
            top_k=settings.RETRIEVAL_TOP_K,
            min_score=settings.RETRIEVAL_MIN_SCORE,
            corpus_version=corpus_ver,
            embedding_model=embed_mdl,
            insufficient=retrieved.insufficient,
            reason_code=retrieved.reason_code,
        )

        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=AgentStatusEnum.COMPLETED.value,
            output_summary=f"Retrieved {len(retrieved.results)} passages, generated {len(retrieved.candidate_hypotheses)} hypotheses",
            output_hash=out_hash,
        )

        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="symptom_rag_analyzer",
            status=AgentStatusEnum.COMPLETED,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary=f"Retrieving evidence for case {state.case_id}",
            output_summary=f"Retrieved {len(retrieved.results)} passages, generated {len(retrieved.candidate_hypotheses)} hypotheses",
        )

        return {
            "retrieved_evidence": retrieved,
            "candidate_hypotheses": retrieved.candidate_hypotheses,
            "current_node": "symptom_rag_analyzer",
            "agent_records": state.agent_records + [rec],
        }

    def node_evidence_based_scanner(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 5: Evidence-Based Scanner (Agent 3). PubMed external scanning.

        SPEC 3 Failure Semantics: A failed or unavailable Scanner is DEGRADED, NOT FAILED.
        The workflow continues with local evidence only; source_availability reflects it.
        External evidence is stored in PostgreSQL only (NOT Qdrant).
        """
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.retrieved_evidence)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="evidence_based_scanner",
            input_summary=f"External PubMed scanning for {len(state.candidate_hypotheses)} hypotheses",
            input_hash=in_hash,
            record_id=record_id,
        )

        scanner_degraded = False
        try:
            result = self.scanner.run(
                case=state.case,
                retrieved_evidence=state.retrieved_evidence,
                gap_hints=state.gap_hints,
                benchmark_mode=state.is_benchmark,
            )
        except Exception as exc:
            logger.warning("Scanner execution encountered error: %s. Continuing in DEGRADED mode.", exc)
            result = AgentExecutionRecord(
                agent_name="evidence_based_scanner",
                status=AgentStatusEnum.DEGRADED,
                error_code="SCANNER_OUTAGE",
                error_message=str(exc),
                input_summary="External literature scanning",
            )

        if isinstance(result, AgentExecutionRecord):
            # Scanner outage / failure -> DEGRADED status, run continues!
            scanner_degraded = True
            empty_ext = ExternalEvidenceSet(
                query_ids=["q:external:degraded"],
                documents=[],
                passages=[],
                query_statuses={"q:external:degraded": QueryStatusEnum.UNAVAILABLE},
                summary="External scanner was degraded or unavailable. Proceeding with local evidence only.",
            )
            out_hash = compute_payload_hash(empty_ext)

            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.DEGRADED.value,
                output_summary="External scanner DEGRADED. Continuing with local evidence.",
                output_hash=out_hash,
                error_code="SCANNER_DEGRADED",
                error_message=result.error_message,
            )

            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="evidence_based_scanner",
                status=AgentStatusEnum.DEGRADED,
                error_code="SCANNER_DEGRADED",
                error_message=result.error_message,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=out_hash,
                input_summary=f"External PubMed scanning for {len(state.candidate_hypotheses)} hypotheses",
                output_summary="External scanner DEGRADED. Continuing with local evidence.",
            )

            return {
                "external_evidence": empty_ext,
                "scanner_degraded": True,
                "current_node": "evidence_based_scanner",
                "agent_records": state.agent_records + [rec],
            }

        external_evidence: ExternalEvidenceSet = result
        out_hash = compute_payload_hash(external_evidence)

        # Check if queries had rate-limit or outage leading to degraded status
        all_unavail = all(
            st in (QueryStatusEnum.UNAVAILABLE, QueryStatusEnum.RATE_LIMITED, QueryStatusEnum.TIMEOUT)
            for st in external_evidence.query_statuses.values()
        ) if external_evidence.query_statuses else False

        if all_unavail:
            scanner_degraded = True
            agent_status = AgentStatusEnum.DEGRADED
        else:
            agent_status = AgentStatusEnum.COMPLETED

        # Persist external evidence into PostgreSQL ONLY (not Qdrant)
        if external_evidence.documents or external_evidence.passages:
            self.repository.save_external_evidence(
                documents=external_evidence.documents,
                passages=external_evidence.passages,
            )

        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=agent_status.value,
            output_summary=f"Retrieved {len(external_evidence.documents)} external docs, {len(external_evidence.passages)} passages",
            output_hash=out_hash,
        )

        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="evidence_based_scanner",
            status=agent_status,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary=f"External PubMed scanning for {len(state.candidate_hypotheses)} hypotheses",
            output_summary=f"Retrieved {len(external_evidence.documents)} external docs, {len(external_evidence.passages)} passages",
        )

        return {
            "external_evidence": external_evidence,
            "scanner_degraded": scanner_degraded,
            "current_node": "evidence_based_scanner",
            "agent_records": state.agent_records + [rec],
        }

    def node_clinical_data_fusion(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 6: Clinical Data Fusion (Agent 4). Synthesizes report and candidate claims."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        version_num = len(state.report_history) + 1

        in_hash = compute_payload_hash({
            "retrieved": state.retrieved_evidence.model_dump() if state.retrieved_evidence else None,
            "external": state.external_evidence.model_dump() if state.external_evidence else None,
        })

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="clinical_data_fusion",
            input_summary=f"Fusing evidence into report v{version_num}",
            input_hash=in_hash,
            record_id=record_id,
        )

        result = self.fusion.run(
            case=state.case,
            local_evidence=state.retrieved_evidence,
            external_evidence=state.external_evidence,
            version_number=version_num,
            case_id=state.case_id,
        )

        if isinstance(result, AgentExecutionRecord):
            # Fusion is a REQUIRED node; failure halts the workflow
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code=result.error_code or "LLM_UNAVAILABLE",
                error_message=result.error_message,
            )
            result.record_id = record_id
            result.input_hash = in_hash
            return {
                "workflow_status": "FAILED",
                "error_code": result.error_code or "LLM_UNAVAILABLE",
                "error_message": result.error_message,
                "current_node": "clinical_data_fusion",
                "agent_records": state.agent_records + [result],
            }

        report: CandidateClinicalReport = result
        out_hash = compute_payload_hash(report)

        # Check if iteration cap is reached: label with "BASELINE: budget reached, not verified"
        if state.current_iteration >= settings.MAX_OPTIMIZER_ITERATIONS and report.confidence_score < settings.CONFIDENCE_THRESHOLD:
            budget_label = " [BASELINE: budget reached, not verified]"
            if budget_label not in report.summary:
                report.summary = report.summary + budget_label

        # Persist immutable report version and claims
        self.repository.save_report_version(
            case_id=state.case_id,
            execution_id=state.execution_id,
            version_number=report.version_number,
            summary=report.summary,
            confidence_score=report.confidence_score,
            status=report.status.value,
            claims=report.claims,
        )

        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=AgentStatusEnum.COMPLETED.value,
            output_summary=f"Generated report v{report.version_number} with {len(report.claims)} claims (conf={report.confidence_score:.3f})",
            output_hash=out_hash,
        )

        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="clinical_data_fusion",
            status=AgentStatusEnum.COMPLETED,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary=f"Fusing evidence into report v{version_num}",
            output_summary=f"Generated report v{report.version_number} with {len(report.claims)} claims (conf={report.confidence_score:.3f})",
        )

        return {
            "report": report,
            "report_history": state.report_history + [report],
            "current_node": "clinical_data_fusion",
            "agent_records": state.agent_records + [rec],
        }

    def node_adaptive_optimizer(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 7: Adaptive Optimizer (Agent 5). Manages iteration feedback loops."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        iter_num = state.current_iteration + 1
        in_hash = compute_payload_hash(state.report)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="adaptive_optimizer",
            input_summary=f"Optimizing report iteration {iter_num} (conf={state.report.confidence_score:.3f})",
            input_hash=in_hash,
            record_id=record_id,
        )

        # Compute structural findings (baseline optimizer operates on structural findings only)
        unsupported = sum(1 for c in state.report.claims if not c.cited_evidence_ids)
        reasons: List[str] = []
        if state.retrieved_evidence is None or len(state.retrieved_evidence.results) < settings.MIN_RELEVANT_PASSAGES:
            reasons.append("NO_RELEVANT_EVIDENCE")

        result = self.optimizer.run(
            report=state.report,
            iteration_history=state.iterations,
            confidence_score=state.report.confidence_score,
            has_blocking_clarification=len(state.blocking_clarifications) > 0,
            scanner_degraded=state.scanner_degraded,
            unsupported_claims_count=unsupported,
            abstention_reasons=reasons,
            execution_id=state.execution_id,
        )

        if isinstance(result, AgentExecutionRecord):
            # Optimizer failure halts the workflow
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code=result.error_code or "LLM_UNAVAILABLE",
                error_message=result.error_message,
            )
            result.record_id = record_id
            result.input_hash = in_hash
            return {
                "workflow_status": "FAILED",
                "error_code": result.error_code or "LLM_UNAVAILABLE",
                "error_message": result.error_message,
                "current_node": "adaptive_optimizer",
                "agent_records": state.agent_records + [result],
            }

        opt_iter: OptimizationIteration = result
        out_hash = compute_payload_hash(opt_iter)

        # Check if budget exhausted
        cap_reached = (
            iter_num >= settings.MAX_OPTIMIZER_ITERATIONS
            or "BUDGET_EXHAUSTED" in opt_iter.action_taken
            or opt_iter.node_selected == "STOP"
        )

        # Save optimization iteration row
        self.repository.save_optimization_iteration(
            execution_id=state.execution_id,
            iteration_number=iter_num,
            node_selected=opt_iter.node_selected,
            confidence_before=opt_iter.confidence_before,
            confidence_after=opt_iter.confidence_after,
            action_taken=opt_iter.action_taken,
            iteration_id=opt_iter.iteration_id,
        )

        self.repository.record_agent_execution_finish(
            record_id=record_id,
            status=AgentStatusEnum.COMPLETED.value,
            output_summary=f"Iteration {iter_num}: selected {opt_iter.node_selected} ({opt_iter.action_taken[:60]}...)",
            output_hash=out_hash,
        )

        rec = AgentExecutionRecord(
            record_id=record_id,
            execution_id=state.execution_id,
            agent_name="adaptive_optimizer",
            status=AgentStatusEnum.COMPLETED,
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_hash=in_hash,
            output_hash=out_hash,
            input_summary=f"Optimizing report iteration {iter_num}",
            output_summary=f"Selected {opt_iter.node_selected}",
        )

        # Formulate gap hints if refining retrieval
        hints = state.gap_hints
        if opt_iter.node_selected == "REFINE_RETRIEVAL":
            hints = hints + [f"Refined query from iteration {iter_num}"]

        return {
            "current_iteration": iter_num,
            "iterations": state.iterations + [opt_iter],
            "iteration_cap_reached": cap_reached,
            "gap_hints": hints,
            "current_node": "adaptive_optimizer",
            "agent_records": state.agent_records + [rec],
        }

    def node_persist_and_finish(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 8: Persist and Finish. Finalizes WorkflowResult and execution status."""
        now = datetime.now(timezone.utc)

        if state.workflow_status == "AWAITING_CLARIFICATION":
            final_status = ReportStatusEnum.AWAITING_CLARIFICATION
            fail_reason = None
        elif state.workflow_status == "FAILED":
            final_status = ReportStatusEnum.FAILED
            fail_reason = state.error_message or state.error_code
        else:
            final_status = ReportStatusEnum.READY_FOR_REVIEW
            fail_reason = None

        # Check if budget was reached with low confidence: append required baseline label (Appendix A3)
        if state.report and state.report.confidence_score < settings.CONFIDENCE_THRESHOLD:
            if (
                state.current_iteration >= settings.MAX_OPTIMIZER_ITERATIONS
                or state.iteration_cap_reached
                or any("BUDGET_EXHAUSTED" in it.action_taken for it in state.iterations)
            ):
                budget_label = " [BASELINE: budget reached, not verified]"
                if budget_label not in state.report.summary:
                    state.report.summary = state.report.summary + budget_label
                    if state.report.report_version_id:
                        self.repository.update_report_version_summary(
                            report_version_id=state.report.report_version_id,
                            summary=state.report.summary,
                        )

        # Update workflow_executions in database
        self.repository.update_workflow_execution(
            execution_id=state.execution_id,
            status=final_status.value,
            failure_reason=fail_reason,
            completed=True,
        )

        result = WorkflowResult(
            execution_id=state.execution_id,
            case_id=state.case_id,
            final_state=final_status,
            report=state.report,
            abstention=None,
            iterations_count=len(state.iterations),
            completed_at=now,
        )

        return {
            "final_result": result,
            "workflow_status": final_status.value,
            "current_node": "persist_and_finish",
        }

    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute the baseline workflow graph end-to-end synchronously."""
        # Ensure parent case row exists in PostgreSQL before workflow_execution is created (FK constraint)
        raw_text = ""
        if initial_state.submission:
            raw_text = initial_state.submission.raw_text
        elif initial_state.submission_payload:
            raw_text = str(initial_state.submission_payload.get("raw_text", ""))

        self.repository.create_case(
            case_id=initial_state.case_id,
            raw_text_hash=compute_payload_hash(raw_text),
            chief_complaint=raw_text[:200] if raw_text else "Pending narrative evaluation",
            status="DRAFT",
        )

        # Create workflow execution row in DB
        self.repository.create_workflow_execution(
            execution_id=initial_state.execution_id,
            case_id=initial_state.case_id,
            is_benchmark=initial_state.is_benchmark,
            status="RUNNING",
        )

        final_dict = self.graph.invoke(initial_state)
        if isinstance(final_dict, dict):
            return WorkflowState.model_validate(final_dict)
        return final_dict
