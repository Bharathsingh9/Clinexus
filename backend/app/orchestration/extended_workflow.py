"""Extended Multi-Agent Clinical Workflow Orchestration (LangGraph).

Specification Section 2, Section 3, Section 5.1, 5.2, Appendix A1, A2, A3 (Req ID: WKF-2, EXT-A, EXT-B).
Full sequential node sequence:
validate_case -> clinical_text_clarifier -> clarification_check ->
symptom_rag_analyzer -> evidence_based_scanner -> clinical_data_fusion ->
claim_verification -> evidence_safety_gate ->
one of (adaptive_optimizer | finalize_ready | finalize_abstain) ->
persist_and_finish.

Safety & Topological Invariants:
1. READY_FOR_REVIEW is reachable ONLY from finalize_ready (after passing all 5 gate rules).
2. The optimizer routes back to the earlier node it selects; every revision passes through
   claim_verification and evidence_safety_gate again.
3. Routing is 100% deterministic Python code, never an LLM.
4. Confidence function uses FULL weights (claim_support 0.50, retrieval_strength 0.20,
   source_availability 0.15, info_completeness 0.15).
5. The iteration counter increments once per optimizer call; at the maximum the optimizer
   is not called again and the gate must finalize (ready or abstain).
6. When abstained, finalize_abstain produces a report containing ONLY the decision,
   SUPPORTED observations (marked), patient-fact summary, missing information, and next steps;
   no unverified considerations, speculative diagnoses, or filler medical content.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from langgraph.graph import END, START, StateGraph

from backend.app.agents.adaptive_optimizer import AdaptiveOptimizer
from backend.app.agents.claim_verification_agent import ClaimVerificationAgent
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
from backend.app.safety.abstention import (
    build_abstention_decision,
    create_abstained_report,
)
from backend.app.safety.confidence import compute_confidence
from backend.app.safety.gate import (
    GateDecision,
    GateOutcome,
    evaluate_gate,
    is_material,
    is_safety_critical,
)
from backend.app.schemas.clinical_case import (
    CaseSubmission,
    ClarificationRequest,
    ClinicalCase,
)
from backend.app.schemas.evidence import (
    EvidenceChunk,
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
    AbstentionDecision,
    AbstentionReasonCode,
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
)

logger = logging.getLogger(__name__)

from sqlalchemy import func, select
from backend.app.orchestration.states import (
    CaseReviewState,
    transition_case_state,
)
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import (
    ClinicalCase as ClinicalCaseModel,
    ReportVersion as ReportVersionModel,
)
from backend.app.provenance import (
    assert_provenance_complete,
    build_provenance_edges_from_workflow,
    save_provenance_edges,
)
from backend.app.safety.urgent import evaluate_urgent_rules


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


class ExtendedWorkflowOrchestrator:
    """Orchestrates the 5-agent pipeline plus Claim Verification and Evidence Safety Gate."""

    def __init__(
        self,
        clarifier: Optional[ClinicalTextClarifier] = None,
        rag_analyzer: Optional[SymptomRAGAnalyzer] = None,
        scanner: Optional[EvidenceBasedScanner] = None,
        fusion: Optional[ClinicalDataFusion] = None,
        verifier: Optional[ClaimVerificationAgent] = None,
        optimizer: Optional[AdaptiveOptimizer] = None,
        repository: Optional[ExecutionRepository] = None,
        llm_provider: Optional[LLMProvider] = None,
        ncbi_client: Optional[NCBIClientProtocol] = None,
        qdrant_wrapper: Optional[QdrantWrapper] = None,
        embedding_service: Optional[EmbeddingService] = None,
        collection_version: int = 1,
        audit_repository: Optional[AuditRepository] = None,
    ):
        self.repository = repository or ExecutionRepository()
        self.embedding_service = embedding_service or EmbeddingService()
        self.audit_repository = audit_repository or AuditRepository(SessionLocal())

        self.clarifier = clarifier or ClinicalTextClarifier(
            llm_provider=llm_provider,
        )
        self.rag_analyzer = rag_analyzer or SymptomRAGAnalyzer(
            qdrant_wrapper=qdrant_wrapper,
            embedding_service=self.embedding_service,
            llm_provider=llm_provider,
            collection_version=collection_version,
        )
        self.scanner = scanner or EvidenceBasedScanner(
            ncbi_client=ncbi_client,
            llm_provider=llm_provider,
            embedding_service=self.embedding_service,
        )
        self.fusion = fusion or ClinicalDataFusion(
            llm_provider=llm_provider,
        )
        self.verifier = verifier or ClaimVerificationAgent(
            llm_provider=llm_provider,
            repository=self.repository,
        )
        self.optimizer = optimizer or AdaptiveOptimizer(
            llm_provider=llm_provider,
        )

        # In-memory tracking of gate decisions per execution to route dynamically
        self._latest_gate_decisions: Dict[str, GateDecision] = {}
        # In-memory tracking of non-supported material claims history for NO_PROGRESS check
        self._unsupported_claims_history: Dict[str, List[Set[str]]] = {}

        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph StateGraph with deterministic conditional routing."""
        builder = StateGraph(WorkflowState)

        # Register all 12 nodes
        builder.add_node("validate_case", self.node_validate_case)
        builder.add_node("clinical_text_clarifier", self.node_clinical_text_clarifier)
        builder.add_node("clarification_check", self.node_clarification_check)
        builder.add_node("symptom_rag_analyzer", self.node_symptom_rag_analyzer)
        builder.add_node("evidence_based_scanner", self.node_evidence_based_scanner)
        builder.add_node("clinical_data_fusion", self.node_clinical_data_fusion)
        builder.add_node("claim_verification", self.node_claim_verification)
        builder.add_node("evidence_safety_gate", self.node_evidence_safety_gate)
        builder.add_node("adaptive_optimizer", self.node_adaptive_optimizer)
        builder.add_node("finalize_ready", self.node_finalize_ready)
        builder.add_node("finalize_abstain", self.node_finalize_abstain)
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
            self.route_after_fusion,
            {
                "continue": "claim_verification",
                "persist_and_finish": "persist_and_finish",
            },
        )

        builder.add_edge("claim_verification", "evidence_safety_gate")

        builder.add_conditional_edges(
            "evidence_safety_gate",
            self.route_safety_gate,
            {
                "finalize_ready": "finalize_ready",
                "finalize_abstain": "finalize_abstain",
                "adaptive_optimizer": "adaptive_optimizer",
            },
        )

        builder.add_conditional_edges(
            "adaptive_optimizer",
            self.route_after_optimizer,
            {
                "symptom_rag_analyzer": "symptom_rag_analyzer",
                "evidence_based_scanner": "evidence_based_scanner",
                "clinical_data_fusion": "clinical_data_fusion",
                "clarification_check": "clarification_check",
                "finalize_abstain": "finalize_abstain",
            },
        )

        builder.add_edge("finalize_ready", "persist_and_finish")
        builder.add_edge("finalize_abstain", "persist_and_finish")
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
    def route_after_fusion(state: WorkflowState) -> str:
        if state.workflow_status == "FAILED":
            return "persist_and_finish"
        return "continue"

    def route_safety_gate(self, state: WorkflowState) -> str:
        """Route outcome from evidence safety gate strictly per GateDecision."""
        gate_dec = self._latest_gate_decisions.get(state.execution_id)
        if not gate_dec:
            # Fallback evaluate gate
            gate_dec = evaluate_gate(state, settings)

        if gate_dec.outcome == GateOutcome.FINALIZE_READY:
            return "finalize_ready"
        elif gate_dec.outcome == GateOutcome.OPTIMIZER:
            return "adaptive_optimizer"
        else:
            return "finalize_abstain"

    @staticmethod
    def route_after_optimizer(state: WorkflowState) -> str:
        """Route optimizer operation back to target node for re-analysis/revision."""
        if state.workflow_status in ("FAILED", "AWAITING_CLARIFICATION"):
            return "finalize_abstain"

        if not state.iterations:
            return "finalize_abstain"

        last_op = state.iterations[-1].node_selected
        if last_op == "REFINE_RETRIEVAL":
            return "symptom_rag_analyzer"
        elif last_op == "FETCH_EXTERNAL":
            return "evidence_based_scanner"
        elif last_op == "REVISE_REPORT":
            return "clinical_data_fusion"
        elif last_op == "REQUEST_CLARIFICATION":
            return "clarification_check"
        else:
            return "finalize_abstain"

    # -------------------------------------------------------------------------
    # Graph Node Implementations
    # -------------------------------------------------------------------------

    def node_validate_case(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 1: Validate CaseSubmission."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        raw_input = state.submission or state.submission_payload
        in_hash = compute_payload_hash(raw_input)

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
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Validating submitted clinical narrative text",
                error_code="INVALID_SUBMISSION",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "INVALID_SUBMISSION",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "validate_case",
            }

    def node_clinical_text_clarifier(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 2: Agent 1 (ClinicalTextClarifier)."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.submission)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="clinical_text_clarifier",
            input_summary="Extracting structured clinical findings",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.submission is not None
            result = self.clarifier.run(state.submission)

            if isinstance(result, AgentExecutionRecord):
                self.repository.record_agent_execution_finish(
                    record_id=record_id,
                    status=AgentStatusEnum.FAILED.value,
                    error_code=result.error_code,
                    error_message=result.error_message,
                )
                return {
                    "workflow_status": "FAILED",
                    "error_code": result.error_code or "CLARIFIER_FAILED",
                    "error_message": result.error_message,
                    "agent_records": state.agent_records + [result],
                    "current_node": "clinical_text_clarifier",
                }

            case: ClinicalCase = result
            case.case_id = state.case_id

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

            blocking = [q for q in case.clarification_questions if q.is_blocking]
            non_blocking = [q for q in case.clarification_questions if not q.is_blocking]

            out_hash = compute_payload_hash(case)
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
                input_summary="Extracting structured clinical findings",
                output_summary=f"Extracted {len(case.structured_symptoms)} symptoms",
            )
            return {
                "case": case,
                "clarification_needed": len(case.clarification_questions) > 0,
                "blocking_clarifications": blocking,
                "non_blocking_clarifications": non_blocking,
                "agent_records": state.agent_records + [rec],
                "current_node": "clinical_text_clarifier",
            }

        except Exception as exc:
            err_msg = f"Clarifier node execution error: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="CLARIFIER_ERROR",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="clinical_text_clarifier",
                status=AgentStatusEnum.FAILED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Extracting structured clinical findings",
                error_code="CLARIFIER_ERROR",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "CLARIFIER_ERROR",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "clinical_text_clarifier",
            }

    def node_clarification_check(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 3: Halts with AWAITING_CLARIFICATION if any blocking questions exist."""
        if state.is_benchmark:
            return {"current_node": "clarification_check"}

        if state.blocking_clarifications:
            logger.info("Blocking clarification questions detected (%d). Halting graph.", len(state.blocking_clarifications))
            with SessionLocal() as session:
                try:
                    transition_case_state(
                        session=session,
                        case_id=state.case_id,
                        target_state=CaseReviewState.EVIDENCE_REVIEW_REQUIRED,
                        actor_id="system_orchestrator",
                        actor_role="system",
                        audit_repo=self.audit_repository,
                        reason="Workflow halted: blocking clarification required",
                    )
                except Exception as t_err:
                    logger.warning("Case transition to EVIDENCE_REVIEW_REQUIRED failed: %s", t_err)

            return {
                "workflow_status": "AWAITING_CLARIFICATION",
                "current_node": "clarification_check",
            }

        return {"current_node": "clarification_check"}

    def node_symptom_rag_analyzer(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 4: Agent 2 (SymptomRAGAnalyzer). Required node."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.case)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="symptom_rag_analyzer",
            input_summary="Dense vector search across BioASQ corpus",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.case is not None
            retrieved_set = self.rag_analyzer.run(
                case=state.case,
                alternative_hints=state.gap_hints,
                benchmark_mode=state.is_benchmark,
            )

            if isinstance(retrieved_set, AgentExecutionRecord):
                self.repository.record_agent_execution_finish(
                    record_id=record_id,
                    status=AgentStatusEnum.FAILED.value,
                    error_code=retrieved_set.error_code,
                    error_message=retrieved_set.error_message,
                )
                return {
                    "workflow_status": "FAILED",
                    "error_code": retrieved_set.error_code or "RAG_FAILED",
                    "error_message": retrieved_set.error_message,
                    "agent_records": state.agent_records + [retrieved_set],
                    "current_node": "symptom_rag_analyzer",
                }

            corpus_ver = retrieved_set.results[0].corpus_version if retrieved_set.results else "v1"
            embed_mdl = retrieved_set.results[0].embedding_model if retrieved_set.results else self.embedding_service.model_name
            self.repository.save_retrieval_run(
                execution_id=state.execution_id,
                query_id=retrieved_set.query_id,
                query_text=state.case.chief_complaint if state.case else "Symptom RAG retrieval",
                query_type="SYMPTOM_DENSE",
                top_k=settings.RETRIEVAL_TOP_K,
                min_score=settings.RETRIEVAL_MIN_SCORE,
                corpus_version=corpus_ver,
                embedding_model=embed_mdl,
                insufficient=retrieved_set.insufficient,
                reason_code=retrieved_set.reason_code,
            )

            hypotheses = retrieved_set.candidate_hypotheses or []
            out_hash = compute_payload_hash(retrieved_set)

            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=f"Retrieved {len(retrieved_set.results)} passages, {len(hypotheses)} hypotheses",
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
                input_summary="Dense vector search across BioASQ corpus",
                output_summary=f"Retrieved {len(retrieved_set.results)} passages",
            )
            return {
                "retrieved_evidence": retrieved_set,
                "candidate_hypotheses": hypotheses,
                "agent_records": state.agent_records + [rec],
                "current_node": "symptom_rag_analyzer",
            }

        except Exception as exc:
            err_msg = f"RAG Analyzer node error: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="RAG_ERROR",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="symptom_rag_analyzer",
                status=AgentStatusEnum.FAILED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Dense vector search across BioASQ corpus",
                error_code="RAG_ERROR",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "RAG_ERROR",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "symptom_rag_analyzer",
            }

    def node_evidence_based_scanner(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 5: Agent 3 (EvidenceBasedScanner). Non-fatal: DEGRADED on outage."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.case)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="evidence_based_scanner",
            input_summary="PubMed external literature targeted scan",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.case is not None
            assert state.retrieved_evidence is not None

            external_set = self.scanner.run(
                case=state.case,
                retrieved_evidence=state.retrieved_evidence,
                gap_hints=state.gap_hints,
                benchmark_mode=state.is_benchmark,
            )

            is_degraded = False
            if external_set.query_statuses:
                if any(st == QueryStatusEnum.UNAVAILABLE for st in external_set.query_statuses.values()):
                    is_degraded = True

            out_status = AgentStatusEnum.DEGRADED if is_degraded else AgentStatusEnum.COMPLETED
            out_hash = compute_payload_hash(external_set)

            if external_set.documents or external_set.passages:
                self.repository.save_external_evidence(
                    documents=external_set.documents,
                    passages=external_set.passages,
                )

            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=out_status.value,
                output_summary=f"Scanned PubMed: {len(external_set.passages)} external passages (degraded={is_degraded})",
                output_hash=out_hash,
            )

            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="evidence_based_scanner",
                status=out_status,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=out_hash,
                input_summary="PubMed external literature scan",
                output_summary=f"External passages: {len(external_set.passages)}",
            )
            return {
                "external_evidence": external_set,
                "scanner_degraded": is_degraded,
                "agent_records": state.agent_records + [rec],
                "current_node": "evidence_based_scanner",
            }

        except Exception as exc:
            err_msg = f"Scanner failed; graceful degradation to local evidence: {exc}"
            logger.warning(err_msg)
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.DEGRADED.value,
                error_code="SCANNER_DEGRADED",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="evidence_based_scanner",
                status=AgentStatusEnum.DEGRADED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="PubMed external literature scan",
                error_code="SCANNER_DEGRADED",
                error_message=err_msg,
            )
            empty_external = ExternalEvidenceSet(execution_id=state.execution_id)
            return {
                "external_evidence": empty_external,
                "scanner_degraded": True,
                "agent_records": state.agent_records + [rec],
                "current_node": "evidence_based_scanner",
            }

    def node_clinical_data_fusion(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 6: Agent 4 (ClinicalDataFusion). Required node."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.case)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="clinical_data_fusion",
            input_summary="Synthesizing multi-source clinical report",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.case is not None
            assert state.retrieved_evidence is not None

            # Query max version in DB for this case so revisions produce new version numbers
            with SessionLocal() as db_session:
                stmt_max = select(func.max(ReportVersionModel.version_number)).where(
                    ReportVersionModel.case_id == state.case_id
                )
                db_max = db_session.execute(stmt_max).scalar() or 0

            version_num = max(len(state.report_history) + 1, db_max + 1)
            report = self.fusion.run(
                case=state.case,
                local_evidence=state.retrieved_evidence,
                external_evidence=state.external_evidence,
                version_number=version_num,
                case_id=state.case_id,
            )

            if isinstance(report, AgentExecutionRecord):
                self.repository.record_agent_execution_finish(
                    record_id=record_id,
                    status=AgentStatusEnum.FAILED.value,
                    error_code=report.error_code,
                    error_message=report.error_message,
                )
                return {
                    "workflow_status": "FAILED",
                    "error_code": report.error_code or "FUSION_FAILED",
                    "error_message": report.error_message,
                    "agent_records": state.agent_records + [report],
                    "current_node": "clinical_data_fusion",
                }

            self.repository.save_report_version(
                case_id=state.case_id,
                execution_id=state.execution_id,
                version_number=report.version_number,
                summary=report.summary,
                confidence_score=report.confidence_score,
                status=report.status.value,
                claims=report.claims,
                report_version_id=report.report_version_id,
            )

            out_hash = compute_payload_hash(report)
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=f"Synthesized CandidateClinicalReport v{report.version_number} ({len(report.claims)} claims)",
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
                input_summary="Synthesizing multi-source clinical report",
                output_summary=f"Report version {report.version_number} generated",
            )
            return {
                "report": report,
                "report_history": state.report_history + [report],
                "agent_records": state.agent_records + [rec],
                "current_node": "clinical_data_fusion",
            }

        except Exception as exc:
            err_msg = f"Fusion node execution error: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="FUSION_ERROR",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="clinical_data_fusion",
                status=AgentStatusEnum.FAILED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Synthesizing multi-source clinical report",
                error_code="FUSION_ERROR",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "FUSION_ERROR",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "clinical_data_fusion",
            }

    def node_claim_verification(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 7: Extension A (ClaimVerificationAgent)."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        in_hash = compute_payload_hash(state.report)

        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="claim_verification",
            input_summary=f"Verifying claims for report v{state.report.version_number if state.report else 'none'}",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.report is not None

            # Gather union of evidence sets
            all_chunks: List[Any] = []
            if state.retrieved_evidence and state.retrieved_evidence.results:
                all_chunks.extend(state.retrieved_evidence.results)
            if state.external_evidence and state.external_evidence.passages:
                all_chunks.extend(state.external_evidence.passages)

            verif_res = self.verifier.run(
                report=state.report,
                evidence_passages=all_chunks,
                case=state.case,
            )

            if isinstance(verif_res, AgentExecutionRecord):
                self.repository.record_agent_execution_finish(
                    record_id=record_id,
                    status=AgentStatusEnum.FAILED.value,
                    error_code=verif_res.error_code,
                    error_message=verif_res.error_message,
                )
                return {
                    "workflow_status": "FAILED",
                    "error_code": verif_res.error_code or "VERIFIER_FAILED",
                    "error_message": verif_res.error_message,
                    "agent_records": state.agent_records + [verif_res],
                    "current_node": "claim_verification",
                }

            verifications: List[ClaimVerification] = verif_res

            # Re-compute full confidence score with claim support (SPEC Section 2, App A1)
            verif_map = {v.claim_id: v for v in verifications}
            material_claims = [c for c in state.report.claims if is_material(c, verif_map.get(c.claim_id))]

            if material_claims:
                supported_val = sum(
                    1.0 if verif_map[c.claim_id].status == ClaimVerificationStatusEnum.SUPPORTED
                    else (0.5 if verif_map[c.claim_id].status == ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED else 0.0)
                    for c in material_claims if c.claim_id in verif_map
                )
                claim_support = supported_val / float(len(material_claims))
            else:
                claim_support = 1.0

            retrieval_scores = [
                p.retrieval_score for p in (state.retrieved_evidence.results if state.retrieved_evidence else [])
            ]
            scanner_unavail = 1 if state.scanner_degraded else 0
            miss_fields = 0
            if state.case and not state.is_benchmark:
                if state.case.age is None:
                    miss_fields += 1
                if state.case.symptom_duration is None:
                    miss_fields += 1

            conf_res = compute_confidence(
                retrieval_scores=retrieval_scores,
                claim_support_score=claim_support,
                unavailable_sources=scanner_unavail,
                configured_sources=2,
                missing_case_fields=miss_fields,
                total_case_fields=4,
                is_baseline=False,
            )

            # Update report confidence score with full calculation
            updated_report = state.report.model_copy(
                update={"confidence_score": conf_res.confidence_score}
            )

            out_hash = compute_payload_hash(verifications)
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=f"Verified {len(verifications)} claims (support={claim_support:.2f}, conf={conf_res.confidence_score:.3f})",
                output_hash=out_hash,
            )

            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="claim_verification",
                status=AgentStatusEnum.COMPLETED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=out_hash,
                input_summary=f"Verifying {len(state.report.claims)} claims",
                output_summary=f"Verified {len(verifications)} claims",
            )
            return {
                "verifications": verifications,
                "report": updated_report,
                "agent_records": state.agent_records + [rec],
                "current_node": "claim_verification",
            }

        except Exception as exc:
            err_msg = f"Verifier node error: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="VERIFIER_ERROR",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="claim_verification",
                status=AgentStatusEnum.FAILED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Verifying claims",
                error_code="VERIFIER_ERROR",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "VERIFIER_ERROR",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "claim_verification",
            }

    def node_evidence_safety_gate(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 8: Extension B (evaluate_gate). Rules 1 to 6 in exact order."""
        gate_decision = evaluate_gate(state=state, settings=settings)
        self._latest_gate_decisions[state.execution_id] = gate_decision

        logger.info(
            "Evidence Safety Gate evaluated: Rule %d triggered -> %s (%s)",
            gate_decision.rule_triggered,
            gate_decision.outcome.value,
            gate_decision.explanation,
        )

        updates: Dict[str, Any] = {
            "current_node": "evidence_safety_gate",
        }
        if gate_decision.abstention_decision:
            updates["abstention"] = gate_decision.abstention_decision

        return updates

    def node_finalize_ready(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 9a: Marks synthesis report READY_FOR_REVIEW upon passing all gate safety checks."""
        assert state.report is not None
        ready_report = state.report.model_copy(
            update={
                "status": ReportStatusEnum.READY_FOR_REVIEW,
                "urgent_banner": state.urgent_banner,
            }
        )

        try:
            self.repository.update_report_version_summary(
                report_version_id=ready_report.report_version_id,
                summary=ready_report.summary,
            )
        except Exception as err:
            logger.warning("Updating report version status failed: %s", err)

        # Build provenance edges from real agent outputs
        edges = build_provenance_edges_from_workflow(
            report=ready_report,
            case=state.case,
            verifications=state.verifications,
            retrieved_evidence=state.retrieved_evidence,
            external_evidence=state.external_evidence,
            iterations=state.iterations,
            abstention=state.abstention,
        )

        # Assert provenance completeness: fails if any material claim has no edge (SPEC 5.3)
        try:
            assert_provenance_complete(ready_report, edges=edges)
        except Exception as prov_err:
            logger.error("Provenance completeness assertion failed at finalize: %s", prov_err)
            # Divert to abstained report with SYSTEM_ERROR (SPEC 5.3)
            system_abst_dec = build_abstention_decision(
                primary_reason=AbstentionReasonCode.SYSTEM_ERROR,
                explanation=f"System error: provenance completeness assertion failed ({prov_err}). Material claims must be grounded.",
                at_iteration_budget=False,
            )
            abstained_rep = create_abstained_report(
                original_report=ready_report,
                abstention_decision=system_abst_dec,
                case=state.case,
                verifications=state.verifications,
            ).model_copy(update={"urgent_banner": state.urgent_banner})

            with SessionLocal() as session:
                save_provenance_edges(session, edges)
                try:
                    transition_case_state(
                        session=session,
                        case_id=state.case_id,
                        target_state=CaseReviewState.ABSTAINED,
                        actor_id="system_orchestrator",
                        actor_role="system",
                        report_version_id=abstained_rep.report_version_id,
                        audit_repo=self.audit_repository,
                        reason="Abstained due to provenance completeness failure",
                    )
                except Exception as t_err:
                    logger.warning("Case transition to ABSTAINED failed: %s", t_err)

            return {
                "report": abstained_rep,
                "abstention": system_abst_dec,
                "workflow_status": ReportStatusEnum.ABSTAINED.value,
                "current_node": "finalize_ready",
            }

        # If provenance complete, save edges and transition case state to READY_FOR_REVIEW
        with SessionLocal() as session:
            save_provenance_edges(session, edges)
            try:
                transition_case_state(
                    session=session,
                    case_id=state.case_id,
                    target_state=CaseReviewState.READY_FOR_REVIEW,
                    actor_id="system_orchestrator",
                    actor_role="system",
                    report_version_id=ready_report.report_version_id,
                    audit_repo=self.audit_repository,
                    reason="Passed safety gate checks and provenance completeness verification",
                )
            except Exception as t_err:
                logger.warning("Case transition to READY_FOR_REVIEW failed: %s", t_err)

        return {
            "report": ready_report,
            "workflow_status": ReportStatusEnum.READY_FOR_REVIEW.value,
            "current_node": "finalize_ready",
        }

    def node_finalize_abstain(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 9b: Creates sanitized abstained report suppressing all unverified considerations."""
        gate_dec = self._latest_gate_decisions.get(state.execution_id)
        abst_dec = state.abstention or (gate_dec.abstention_decision if gate_dec else None)

        if not abst_dec:
            primary_reason = (
                gate_dec.primary_reason if gate_dec and gate_dec.primary_reason
                else AbstentionReasonCode.UNSUPPORTED_MATERIAL_CLAIMS
            )
            expl = gate_dec.explanation if gate_dec else "Clinical decision support abstained for safety."
            abst_dec = build_abstention_decision(
                primary_reason=primary_reason,
                explanation=expl,
                at_iteration_budget=(state.current_iteration >= settings.MAX_OPTIMIZER_ITERATIONS),
            )

        abstained_report = create_abstained_report(
            original_report=state.report,
            abstention_decision=abst_dec,
            case=state.case,
            verifications=state.verifications,
        ).model_copy(update={"urgent_banner": state.urgent_banner})

        try:
            self.repository.save_report_version(
                case_id=state.case_id,
                execution_id=state.execution_id,
                version_number=abstained_report.version_number,
                summary=abstained_report.summary,
                confidence_score=abstained_report.confidence_score,
                status=abstained_report.status.value,
                claims=abstained_report.claims,
                report_version_id=abstained_report.report_version_id,
            )
            self.repository.save_abstention_decision(
                execution_id=state.execution_id,
                case_id=state.case_id,
                decision=abst_dec,
                report_version_id=abstained_report.report_version_id,
            )
        except Exception as db_err:
            logger.warning("Persisting abstention decision failed: %s", db_err)

        # Build and save provenance edges for abstained report
        edges = build_provenance_edges_from_workflow(
            report=abstained_report,
            case=state.case,
            verifications=state.verifications,
            retrieved_evidence=state.retrieved_evidence,
            external_evidence=state.external_evidence,
            iterations=state.iterations,
            abstention=abst_dec,
        )
        with SessionLocal() as session:
            save_provenance_edges(session, edges)
            try:
                transition_case_state(
                    session=session,
                    case_id=state.case_id,
                    target_state=CaseReviewState.ABSTAINED,
                    actor_id="system_orchestrator",
                    actor_role="system",
                    report_version_id=abstained_report.report_version_id,
                    audit_repo=self.audit_repository,
                    reason=abst_dec.explanation,
                )
            except Exception as t_err:
                logger.warning("Case transition to ABSTAINED failed: %s", t_err)

        return {
            "report": abstained_report,
            "abstention": abst_dec,
            "workflow_status": ReportStatusEnum.ABSTAINED.value,
            "current_node": "finalize_abstain",
        }

    def node_adaptive_optimizer(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 9c: Agent 5 (AdaptiveOptimizer). Iteration counter increments once per call."""
        now = datetime.now(timezone.utc)
        record_id = str(uuid.uuid4())
        new_iteration = state.current_iteration + 1
        is_cap_reached = new_iteration >= settings.MAX_OPTIMIZER_ITERATIONS

        # Calculate non-SUPPORTED material claims set for NO_PROGRESS rule (SPEC 4.5)
        verif_map = {v.claim_id: v for v in state.verifications}
        report_claims = state.report.claims if state.report else []
        current_non_supported: Set[str] = {
            c.claim_id for c in report_claims
            if is_material(c, verif_map.get(c.claim_id))
            and (c.claim_id not in verif_map or verif_map[c.claim_id].status != ClaimVerificationStatusEnum.SUPPORTED)
        }

        # Track history per execution
        exec_history = self._unsupported_claims_history.setdefault(state.execution_id, [])
        exec_history.append(current_non_supported)

        in_hash = compute_payload_hash(state.report)
        self.repository.record_agent_execution_start(
            execution_id=state.execution_id,
            agent_name="adaptive_optimizer",
            input_summary=f"Optimizer iteration {new_iteration} evaluation",
            input_hash=in_hash,
            record_id=record_id,
        )

        try:
            assert state.report is not None

            # Pass verification and gate findings to optimizer
            gate_dec = self._latest_gate_decisions.get(state.execution_id)
            abst_reasons = [rc.value for rc in gate_dec.reason_codes] if gate_dec else []

            iter_record = self.optimizer.run(
                report=state.report,
                iteration_history=state.iterations,
                confidence_score=state.report.confidence_score,
                has_blocking_clarification=len(state.blocking_clarifications) > 0,
                scanner_degraded=state.scanner_degraded,
                unsupported_claims_count=len(current_non_supported),
                abstention_reasons=abst_reasons,
                execution_id=state.execution_id,
                non_supported_material_claims=current_non_supported,
                non_supported_claims_history=exec_history,
                gate_findings={"rule_triggered": gate_dec.rule_triggered if gate_dec else None},
            )

            if isinstance(iter_record, AgentExecutionRecord):
                self.repository.record_agent_execution_finish(
                    record_id=record_id,
                    status=AgentStatusEnum.FAILED.value,
                    error_code=iter_record.error_code,
                    error_message=iter_record.error_message,
                )
                return {
                    "workflow_status": "FAILED",
                    "error_code": iter_record.error_code or "OPTIMIZER_FAILED",
                    "error_message": iter_record.error_message,
                    "agent_records": state.agent_records + [iter_record],
                    "current_node": "adaptive_optimizer",
                    "current_iteration": new_iteration,
                    "iteration_cap_reached": is_cap_reached,
                }

            self.repository.save_optimization_iteration(
                execution_id=state.execution_id,
                iteration_number=iter_record.iteration_number,
                node_selected=iter_record.node_selected,
                confidence_before=iter_record.confidence_before,
                confidence_after=iter_record.confidence_after,
                action_taken=iter_record.action_taken,
                iteration_id=iter_record.iteration_id,
            )

            out_hash = compute_payload_hash(iter_record)
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.COMPLETED.value,
                output_summary=f"Iteration {iter_record.iteration_number}: {iter_record.node_selected}",
                output_hash=out_hash,
            )

            agent_rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="adaptive_optimizer",
                status=AgentStatusEnum.COMPLETED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=out_hash,
                input_summary=f"Optimizer routing for iteration {iter_record.iteration_number}",
                output_summary=f"Selected {iter_record.node_selected}",
            )

            # Generate query hints if refining retrieval
            new_gap_hints = list(state.gap_hints)
            if iter_record.node_selected == "REFINE_RETRIEVAL":
                new_gap_hints.append(f"Refinement query iteration {new_iteration}")

            return {
                "current_iteration": new_iteration,
                "iteration_cap_reached": is_cap_reached,
                "iterations": state.iterations + [iter_record],
                "gap_hints": new_gap_hints,
                "agent_records": state.agent_records + [agent_rec],
                "current_node": "adaptive_optimizer",
            }

        except Exception as exc:
            err_msg = f"Optimizer node error: {exc}"
            self.repository.record_agent_execution_finish(
                record_id=record_id,
                status=AgentStatusEnum.FAILED.value,
                error_code="OPTIMIZER_ERROR",
                error_message=err_msg,
            )
            rec = AgentExecutionRecord(
                record_id=record_id,
                execution_id=state.execution_id,
                agent_name="adaptive_optimizer",
                status=AgentStatusEnum.FAILED,
                started_at=now,
                completed_at=datetime.now(timezone.utc),
                input_hash=in_hash,
                output_hash=compute_payload_hash(None),
                input_summary="Optimizer routing",
                error_code="OPTIMIZER_ERROR",
                error_message=err_msg,
            )
            return {
                "workflow_status": "FAILED",
                "error_code": "OPTIMIZER_ERROR",
                "error_message": err_msg,
                "agent_records": state.agent_records + [rec],
                "current_node": "adaptive_optimizer",
                "current_iteration": new_iteration,
                "iteration_cap_reached": is_cap_reached,
            }

    def node_persist_and_finish(self, state: WorkflowState) -> Dict[str, Any]:
        """Node 10: Persists final WorkflowResult and updates WorkflowExecution status."""
        final_state_enum = ReportStatusEnum.FAILED
        if state.workflow_status == ReportStatusEnum.READY_FOR_REVIEW.value:
            final_state_enum = ReportStatusEnum.READY_FOR_REVIEW
        elif state.workflow_status == ReportStatusEnum.ABSTAINED.value:
            final_state_enum = ReportStatusEnum.ABSTAINED
        elif state.workflow_status == "AWAITING_CLARIFICATION":
            final_state_enum = ReportStatusEnum.AWAITING_CLARIFICATION

        workflow_res = WorkflowResult(
            execution_id=state.execution_id,
            case_id=state.case_id,
            final_state=final_state_enum,
            report=state.report,
            abstention=state.abstention,
            iterations_count=len(state.iterations),
            completed_at=datetime.now(timezone.utc),
        )

        try:
            self.repository.update_workflow_execution(
                execution_id=state.execution_id,
                status=state.workflow_status,
                completed=True,
                failure_reason=state.error_message or state.error_code,
            )
        except Exception as exc:
            logger.warning("Updating workflow execution failed: %s", exc)

        if state.workflow_status == "FAILED":
            with SessionLocal() as session:
                try:
                    transition_case_state(
                        session=session,
                        case_id=state.case_id,
                        target_state=CaseReviewState.FAILED,
                        actor_id="system_orchestrator",
                        actor_role="system",
                        audit_repo=self.audit_repository,
                        reason=state.error_message or "Execution failure",
                    )
                except Exception as t_err:
                    logger.warning("Could not transition case to FAILED: %s", t_err)

        return {
            "final_result": workflow_res,
            "current_node": "persist_and_finish",
        }

    # -------------------------------------------------------------------------
    # Public Execution Entry Point
    # -------------------------------------------------------------------------

    def run(
        self,
        submission: Union[WorkflowState, CaseSubmission, Dict[str, Any]],
        case_id: Optional[str] = None,
        is_benchmark: bool = False,
    ) -> WorkflowState:
        """Execute the extended LangGraph workflow synchronously."""
        if isinstance(submission, WorkflowState):
            initial_state = submission
        else:
            cid = case_id or str(uuid.uuid4())
            exec_id = str(uuid.uuid4())
            sub_obj = submission if isinstance(submission, CaseSubmission) else None
            sub_dict = submission if isinstance(submission, dict) else None

            initial_state = WorkflowState(
                execution_id=exec_id,
                case_id=cid,
                submission=sub_obj,
                submission_payload=sub_dict,
                is_benchmark=is_benchmark,
                workflow_status="RUNNING",
            )

        # Ensure parent case row exists in PostgreSQL before workflow_execution is created (FK constraint)
        raw_text = ""
        if initial_state.submission:
            raw_text = initial_state.submission.raw_text
        elif initial_state.submission_payload:
            raw_text = str(initial_state.submission_payload.get("raw_text", ""))

        # Urgent symptom rules evaluation BEFORE analysis (SPEC 8, SAF-3)
        urgent_eval = evaluate_urgent_rules(
            raw_text=raw_text,
            findings=None,
            case_id=initial_state.case_id,
            audit_repo=self.audit_repository,
        )
        if urgent_eval.is_urgent:
            initial_state.urgent_banner = urgent_eval.banner_text

        # Ensure parent case exists and transition to ANALYZING
        with SessionLocal() as session:
            existing_case = session.get(ClinicalCaseModel, initial_state.case_id)
            if not existing_case:
                self.repository.create_case(
                    case_id=initial_state.case_id,
                    raw_text_hash=compute_payload_hash(raw_text),
                    chief_complaint=raw_text[:200] if raw_text else "Pending narrative evaluation",
                    status="DRAFT",
                )
            try:
                transition_case_state(
                    session=session,
                    case_id=initial_state.case_id,
                    target_state=CaseReviewState.ANALYZING,
                    actor_id="system_orchestrator",
                    actor_role="system",
                    audit_repo=self.audit_repository,
                    reason="Starting multi-agent clinical workflow execution",
                )
            except Exception as t_err:
                logger.warning("Case transition to ANALYZING failed: %s", t_err)

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
