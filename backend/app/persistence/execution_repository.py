"""Execution, Evidence, Case, and Report Persistence Repository.

Specification Section 2, 3, 6 (Req ID: DB-3).
Handles persistence of:
- clinical_cases and case_findings
- workflow_executions
- agent_executions (before with RUNNING and after with COMPLETED/DEGRADED/FAILED)
- retrieval_runs
- report_versions and candidate_claims (immutable versioning)
- optimization_iterations
- external evidence_documents and evidence_passages (PostgreSQL only, not Qdrant)
- reconcile_interrupted_executions() (at startup, marks any RUNNING execution as FAILED(reason=INTERRUPTED))
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.persistence.database import SessionLocal, sync_engine
from backend.app.persistence.models import (
    AbstentionDecision as AbstentionDecisionModel,
    AgentExecution as AgentExecutionModel,
    CandidateClaim as CandidateClaimModel,
    CaseFinding as CaseFindingModel,
    ClaimVerification as ClaimVerificationModel,
    ClinicalCase as ClinicalCaseModel,
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
    OptimizationIteration as OptimizationIterationModel,
    ReportVersion as ReportVersionModel,
    RetrievalRun as RetrievalRunModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.schemas.clinical_case import (
    CaseSubmission,
    ClinicalCase as ClinicalCaseSchema,
)
from backend.app.schemas.evidence import (
    EvidenceChunk as EvidenceChunkSchema,
    EvidenceDocument as EvidenceDocumentSchema,
)
from backend.app.schemas.executions import (
    AgentExecutionRecord as AgentExecutionRecordSchema,
    AgentStatusEnum,
    OptimizationIteration as OptimizationIterationSchema,
)
from backend.app.schemas.reports import (
    AbstentionDecision as AbstentionDecisionSchema,
    CandidateClaim as CandidateClaimSchema,
    CandidateClinicalReport as CandidateClinicalReportSchema,
    ClaimVerification as ClaimVerificationSchema,
)

logger = logging.getLogger(__name__)


def reconcile_interrupted_executions(session: Optional[Session] = None) -> int:
    """Mark any workflow execution left in RUNNING status as FAILED(reason='INTERRUPTED').

    Specification Section 2, 3 (SPEC 2, DB-3).
    Hooked into application startup in Phase 6.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        now = datetime.now(timezone.utc)
        stmt = (
            select(WorkflowExecutionModel)
            .where(WorkflowExecutionModel.status == "RUNNING")
        )
        running_execs = session.scalars(stmt).all()
        count = len(running_execs)

        for exec_row in running_execs:
            exec_row.status = "FAILED"
            exec_row.failure_reason = "INTERRUPTED"
            exec_row.completed_at = now

            # Also reconcile any orphaned RUNNING agent execution records for this workflow
            agent_stmt = (
                select(AgentExecutionModel)
                .where(
                    AgentExecutionModel.execution_id == exec_row.id,
                    AgentExecutionModel.status == "RUNNING",
                )
            )
            running_agents = session.scalars(agent_stmt).all()
            for agent_row in running_agents:
                agent_row.status = "FAILED"
                agent_row.error_message = "Workflow execution was interrupted."
                agent_row.error_code = "INTERRUPTED"
                agent_row.completed_at = now

        session.commit()
        if count > 0:
            logger.info("Reconciled %d interrupted workflow executions to FAILED(INTERRUPTED)", count)
        return count
    except Exception as exc:
        session.rollback()
        logger.error("Failed to reconcile interrupted executions: %s", exc)
        raise
    finally:
        if close_session:
            session.close()


class ExecutionRepository:
    """Synchronous repository for workflow execution persistence in PostgreSQL."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    @contextmanager
    def _session_scope(self):
        if self._session is not None:
            yield self._session
        else:
            session = SessionLocal()
            try:
                yield session
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def create_case(
        self,
        case_id: str,
        raw_text_hash: str,
        chief_complaint: str,
        age: Optional[int] = None,
        age_missing_reason: Optional[str] = None,
        sex: Optional[str] = None,
        symptom_duration: Optional[str] = None,
        duration_missing_reason: Optional[str] = None,
        status: str = "DRAFT",
    ) -> ClinicalCaseModel:
        """Persist or update clinical_cases record."""
        with self._session_scope() as session:
            existing = session.get(ClinicalCaseModel, case_id)
            if existing:
                existing.raw_text_hash = raw_text_hash
                existing.chief_complaint = chief_complaint
                existing.age = age
                existing.age_missing_reason = age_missing_reason
                existing.sex = sex
                existing.symptom_duration = symptom_duration
                existing.duration_missing_reason = duration_missing_reason
                existing.status = status
                session.commit()
                session.refresh(existing)
                return existing

            case_row = ClinicalCaseModel(
                id=case_id,
                raw_text_hash=raw_text_hash,
                age=age,
                age_missing_reason=age_missing_reason,
                sex=sex,
                chief_complaint=chief_complaint,
                symptom_duration=symptom_duration,
                duration_missing_reason=duration_missing_reason,
                status=status,
                created_at=datetime.now(timezone.utc),
            )
            session.add(case_row)
            session.commit()
            session.refresh(case_row)
            return case_row

    def save_case_findings(
        self,
        case_id: str,
        findings: List[Any],
    ) -> List[CaseFindingModel]:
        """Persist structured findings for a clinical case."""
        with self._session_scope() as session:
            created: List[CaseFindingModel] = []
            for f in findings:
                finding_id = getattr(f, "finding_id", str(uuid.uuid4()))
                existing = session.get(CaseFindingModel, finding_id)
                if existing:
                    continue

                span = getattr(f, "source_span", None)
                span_start = span.start if span else 0
                span_end = span.end if span else 0

                term = getattr(f, "term", str(f))
                norm_term = getattr(f, "normalized_term", None)
                vocab = getattr(f, "vocabulary", "LOCAL")
                pol = getattr(f, "polarity", "PRESENT")
                pol_str = pol.value if hasattr(pol, "value") else str(pol)

                orig = getattr(f, "origin", "STATED")
                orig_str = orig.value if hasattr(orig, "value") else str(orig)

                row = CaseFindingModel(
                    id=finding_id,
                    case_id=case_id,
                    term=term,
                    normalized_term=norm_term,
                    vocabulary=vocab,
                    polarity=pol_str,
                    onset=getattr(f, "onset", None),
                    duration=getattr(f, "duration", None),
                    severity=getattr(f, "severity", None),
                    progression=getattr(f, "progression", None),
                    source_span_start=span_start,
                    source_span_end=span_end,
                    origin=orig_str,
                    uncertainty_note=getattr(f, "uncertainty_note", None),
                )
                session.add(row)
                created.append(row)
            session.commit()
            return created

    def create_workflow_execution(
        self,
        execution_id: str,
        case_id: str,
        is_benchmark: bool = False,
        status: str = "RUNNING",
    ) -> WorkflowExecutionModel:
        """Create new workflow execution row."""
        with self._session_scope() as session:
            existing = session.get(WorkflowExecutionModel, execution_id)
            if existing:
                return existing

            exec_row = WorkflowExecutionModel(
                id=execution_id,
                case_id=case_id,
                status=status,
                is_benchmark=is_benchmark,
                started_at=datetime.now(timezone.utc),
            )
            session.add(exec_row)
            session.commit()
            session.refresh(exec_row)
            return exec_row

    def update_workflow_execution(
        self,
        execution_id: str,
        status: str,
        failure_reason: Optional[str] = None,
        completed: bool = False,
    ) -> Optional[WorkflowExecutionModel]:
        """Update workflow execution status."""
        with self._session_scope() as session:
            exec_row = session.get(WorkflowExecutionModel, execution_id)
            if not exec_row:
                return None

            exec_row.status = status
            if failure_reason:
                exec_row.failure_reason = failure_reason
            if completed:
                exec_row.completed_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(exec_row)
            return exec_row

    def record_agent_execution_start(
        self,
        execution_id: str,
        agent_name: str,
        input_summary: str,
        input_hash: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> AgentExecutionModel:
        """Persist AgentExecution with RUNNING status before node starts."""
        with self._session_scope() as session:
            rid = record_id or str(uuid.uuid4())
            agent_row = AgentExecutionModel(
                id=rid,
                execution_id=execution_id,
                agent_name=agent_name,
                status=AgentStatusEnum.RUNNING.value,
                started_at=datetime.now(timezone.utc),
                input_summary=input_summary,
                input_hash=input_hash,
            )
            session.add(agent_row)
            session.commit()
            session.refresh(agent_row)
            return agent_row

    def record_agent_execution_finish(
        self,
        record_id: str,
        status: str,
        output_summary: Optional[str] = None,
        output_hash: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[AgentExecutionModel]:
        """Update AgentExecution after node finishes with outcome and timing."""
        with self._session_scope() as session:
            agent_row = session.get(AgentExecutionModel, record_id)
            if not agent_row:
                return None

            now = datetime.now(timezone.utc)
            agent_row.status = status
            agent_row.completed_at = now
            if agent_row.started_at:
                st = agent_row.started_at
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                duration = int((now - st).total_seconds() * 1000)
                agent_row.duration_ms = max(0, duration)

            if output_summary is not None:
                agent_row.output_summary = output_summary
            if output_hash is not None:
                agent_row.output_hash = output_hash
            if error_code is not None:
                agent_row.error_code = error_code
            if error_message is not None:
                agent_row.error_message = error_message

            session.commit()
            session.refresh(agent_row)
            return agent_row

    def save_retrieval_run(
        self,
        execution_id: str,
        query_id: str,
        query_text: str,
        query_type: str,
        top_k: int,
        min_score: float,
        corpus_version: str,
        embedding_model: str,
        insufficient: bool = False,
        reason_code: Optional[str] = None,
    ) -> RetrievalRunModel:
        """Persist retrieval run record."""
        with self._session_scope() as session:
            run_row = RetrievalRunModel(
                id=str(uuid.uuid4()),
                execution_id=execution_id,
                query_id=query_id,
                query_text=query_text,
                query_type=query_type,
                top_k=top_k,
                min_score=min_score,
                corpus_version=corpus_version,
                embedding_model=embedding_model,
                insufficient=insufficient,
                reason_code=reason_code,
                created_at=datetime.now(timezone.utc),
            )
            session.add(run_row)
            session.commit()
            session.refresh(run_row)
            return run_row

    def save_external_evidence(
        self,
        documents: List[EvidenceDocumentSchema],
        passages: List[EvidenceChunkSchema],
    ) -> Tuple[int, int]:
        """Persist external evidence to PostgreSQL ONLY (not Qdrant).

        Returns count of inserted (docs, passages).
        """
        with self._session_scope() as session:
            docs_inserted = 0
            passages_inserted = 0

            # 1. Evidence Documents
            for doc in documents:
                src = doc.source.value if hasattr(doc.source, "value") else str(doc.source)
                src_id = str(doc.source_id)

                stmt = select(EvidenceDocumentModel).where(
                    EvidenceDocumentModel.source == src,
                    EvidenceDocumentModel.source_id == src_id,
                )
                existing_doc = session.scalars(stmt).first()
                if existing_doc:
                    continue

                doc_row = EvidenceDocumentModel(
                    id=doc.document_id,
                    source=src,
                    source_id=src_id,
                    title=doc.title,
                    authors=", ".join(doc.authors) if isinstance(doc.authors, list) else doc.authors,
                    journal=doc.journal,
                    publication_date=doc.publication_date,
                    pmid=doc.pmid,
                    doi=doc.doi,
                    url=doc.url,
                    abstract=doc.abstract,
                    abstract_missing=doc.abstract_missing,
                    evidence_level=doc.evidence_level or "UNKNOWN",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(doc_row)
                docs_inserted += 1

            session.flush()

            # 2. Evidence Passages
            for chunk in passages:
                stmt = select(EvidencePassageModel).where(
                    EvidencePassageModel.evidence_id == chunk.evidence_id
                )
                existing_chunk = session.scalars(stmt).first()
                if existing_chunk:
                    continue

                passage_row = EvidencePassageModel(
                    id=str(uuid.uuid4()),
                    document_id=chunk.document_id,
                    evidence_id=chunk.evidence_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    text_hash=chunk.text_hash,
                    short_passage=chunk.short_passage,
                    token_count=chunk.token_count,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(passage_row)
                passages_inserted += 1

            session.commit()
            return docs_inserted, passages_inserted

    def save_report_version(
        self,
        case_id: str,
        execution_id: str,
        version_number: int,
        summary: str,
        confidence_score: float,
        status: str,
        claims: List[CandidateClaimSchema],
        report_version_id: Optional[str] = None,
    ) -> ReportVersionModel:
        """Persist immutable report version and its candidate claims."""
        with self._session_scope() as session:
            report_row = ReportVersionModel(
                id=report_version_id or str(uuid.uuid4()),
                case_id=case_id,
                execution_id=execution_id,
                version_number=version_number,
                summary=summary,
                confidence_score=confidence_score,
                status=status,
                created_at=datetime.now(timezone.utc),
            )
            session.add(report_row)
            session.flush()

            for c in claims:
                claim_row = CandidateClaimModel(
                    id=str(uuid.uuid4()),
                    report_version_id=report_row.id,
                    claim_text=c.claim_text,
                    section=c.section,
                    is_material=c.is_material,
                    is_safety_critical=c.is_safety_critical,
                )
                session.add(claim_row)

            session.commit()
            session.refresh(report_row)
            return report_row

    def update_report_version_summary(
        self,
        report_version_id: str,
        summary: str,
    ) -> Optional[ReportVersionModel]:
        """Update summary of a persisted report version (e.g. to attach baseline budget exhaustion label)."""
        with self._session_scope() as session:
            row = session.get(ReportVersionModel, report_version_id)
            if row:
                row.summary = summary
                session.commit()
                session.refresh(row)
            return row

    def save_optimization_iteration(
        self,
        execution_id: str,
        iteration_number: int,
        node_selected: str,
        confidence_before: float,
        confidence_after: Optional[float],
        action_taken: str,
        iteration_id: Optional[str] = None,
    ) -> OptimizationIterationModel:
        """Persist an optimization iteration row."""
        with self._session_scope() as session:
            row = OptimizationIterationModel(
                id=iteration_id or str(uuid.uuid4()),
                execution_id=execution_id,
                iteration_number=iteration_number,
                node_selected=node_selected,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                action_taken=action_taken,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def save_claim_verifications(
        self,
        verifications: List[ClaimVerificationSchema],
    ) -> List[ClaimVerificationModel]:
        """Persist a list of ClaimVerification records to claim_verifications table."""
        if not verifications:
            return []
        with self._session_scope() as session:
            rows: List[ClaimVerificationModel] = []
            for v in verifications:
                row = ClaimVerificationModel(
                    id=str(uuid.uuid4()),
                    claim_id=v.claim_id,
                    category=v.category,
                    status=v.status.value if hasattr(v.status, "value") else str(v.status),
                    justification=v.justification,
                    method=v.method.value if hasattr(v.method, "value") else str(v.method),
                    verifier_version=v.verifier_version,
                    verified_at=v.verified_at,
                    limitations=v.limitations,
                )
                session.add(row)
                rows.append(row)
            session.commit()
            for r in rows:
                session.refresh(r)
            return rows

    def save_abstention_decision(
        self,
        execution_id: str,
        case_id: str,
        decision: AbstentionDecisionSchema,
        report_version_id: Optional[str] = None,
    ) -> AbstentionDecisionModel:
        """Persist an AbstentionDecision record to abstention_decisions table."""
        with self._session_scope() as session:
            reason_codes_json = json.dumps([
                rc.value if hasattr(rc, "value") else str(rc) for rc in decision.reason_codes
            ])
            requires_json = json.dumps([
                req.value if hasattr(req, "value") else str(req) for req in decision.requires
            ])
            row = AbstentionDecisionModel(
                id=decision.decision_id or str(uuid.uuid4()),
                execution_id=execution_id,
                case_id=case_id,
                report_version_id=report_version_id,
                abstained=decision.abstained,
                reason_codes=reason_codes_json,
                explanation=decision.explanation,
                affected_claim_ids=json.dumps(decision.affected_claim_ids),
                affected_sections=json.dumps(decision.affected_sections),
                missing_evidence=json.dumps(decision.missing_evidence),
                missing_information=json.dumps(decision.missing_information),
                suggested_next_steps=json.dumps(decision.suggested_next_steps),
                requires=requires_json,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row
