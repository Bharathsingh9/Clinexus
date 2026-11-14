"""Workflow Execution and Analysis Endpoints.

Specification Section 7, Section 3, Section 5.1, 5.2, 5.4 (API-4, WKF-2, EXT-D):
- POST /cases/{id}/analyze: Triggers extended multi-agent workflow asynchronously in BackgroundTasks.
  Returns 202 Accepted with execution_id.
  Idempotent via Idempotency-Key header.
  State machine checked: second request while ANALYZING returns 409 INVALID_STATE.
  Never uses benchmark mode via the API.
- GET /executions/{id}: Execution status, timestamps, error_code, last node.
- GET /executions/{id}/agents: Per-agent records (agent, status, times, durations, error_code, warnings).
  STRICT INVARIANT: NEVER returns prompts or outputs.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Header, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.api.deps import (
    get_current_user,
    get_db,
    get_llm_provider,
    require_clinician_or_admin,
)
from backend.app.api.errors import (
    InvalidStateAPIError,
    NotFoundAPIError,
)
from backend.app.api.rate_limiter import rate_limit_analyze
from backend.app.core.llm_provider import LLMProvider
from backend.app.orchestration.extended_workflow import ExtendedWorkflowOrchestrator
from backend.app.orchestration.states import (
    CaseReviewState,
    transition_case_state,
)
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.execution_repository import ExecutionRepository
from backend.app.persistence.models import (
    AgentExecution as AgentExecutionModel,
    ClinicalCase as ClinicalCaseModel,
    User as UserModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.schemas.clinical_case import CaseSubmission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["executions"])


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_id: str
    status: str


class ExecutionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    case_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_code: Optional[str] = None
    last_node: Optional[str] = None


class AgentRecordResponse(BaseModel):
    """Sanitized agent execution telemetry strictly omitting prompts or generated outputs."""
    model_config = ConfigDict(extra="forbid")
    agent: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    warnings: Optional[str] = None


def _run_workflow_background_task(
    case_id: str,
    execution_id: str,
    raw_text: str,
    idempotency_key: Optional[str],
    llm_provider: Optional[LLMProvider],
) -> None:
    """Execute extended LangGraph workflow in asynchronous background task."""
    repo = ExecutionRepository()
    audit_repo = AuditRepository(SessionLocal())
    orchestrator = ExtendedWorkflowOrchestrator(
        llm_provider=llm_provider,
        repository=repo,
        audit_repository=audit_repo,
    )

    try:
        submission = CaseSubmission(raw_text=raw_text)
        orchestrator.run(
            submission=submission,
            case_id=case_id,
            is_benchmark=False,
            execution_id=execution_id,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        logger.error("Background workflow execution failed: %s", exc, exc_info=True)
        repo.update_workflow_execution(
            execution_id=execution_id,
            status="FAILED",
            failure_reason=str(exc),
            completed=True,
        )
        with SessionLocal() as session:
            try:
                transition_case_state(
                    session=session,
                    case_id=case_id,
                    target_state=CaseReviewState.FAILED,
                    actor_id="system_orchestrator",
                    actor_role="system",
                    audit_repo=audit_repo,
                    reason=f"Background execution exception: {exc}",
                )
            except Exception as t_err:
                logger.warning("Could not transition case to FAILED after background failure: %s", t_err)


@router.post(
    "/cases/{id}/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger clinical case multi-agent analysis workflow",
    dependencies=[Depends(rate_limit_analyze)],
)
def trigger_analysis(
    id: str,
    background_tasks: BackgroundTasks,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    current_user: UserModel = Depends(require_clinician_or_admin),
    db: Session = Depends(get_db),
    llm_provider: Optional[LLMProvider] = Depends(get_llm_provider),
) -> AnalyzeResponse:
    """Trigger the multi-agent clinical decision support workflow.

    Enforces Idempotency-Key uniqueness, state machine transitions, and background execution.
    A second request while ANALYZING returns 409 INVALID_STATE.
    """
    case_row = db.get(ClinicalCaseModel, id)
    if not case_row:
        raise NotFoundAPIError(f"ClinicalCase '{id}' not found.")

    repo = ExecutionRepository(db)

    # 1. Idempotency Check
    if idempotency_key:
        clean_key = idempotency_key.strip()
        existing_exec = repo.get_workflow_execution_by_idempotency_key(case_id=id, idempotency_key=clean_key)
        if existing_exec:
            logger.info("Idempotent repeat detected for case '%s' with key '%s'", id, clean_key)
            return AnalyzeResponse(
                execution_id=existing_exec.id,
                status=existing_exec.status,
            )

    # 2. State Machine Checks (SPEC 5.4, EXT-D)
    if case_row.status == CaseReviewState.ANALYZING.value:
        raise InvalidStateAPIError(
            f"Case '{id}' is currently ANALYZING. Concurrent analysis is not permitted."
        )

    # Allowed transitions to ANALYZING: DRAFT, EVIDENCE_REVIEW_REQUIRED, ABSTAINED, REVISION_REQUESTED, FAILED
    allowed_from_states = {
        CaseReviewState.DRAFT.value,
        CaseReviewState.EVIDENCE_REVIEW_REQUIRED.value,
        CaseReviewState.ABSTAINED.value,
        CaseReviewState.REVISION_REQUESTED.value,
        CaseReviewState.FAILED.value,
    }
    if case_row.status not in allowed_from_states:
        raise InvalidStateAPIError(
            f"Cannot trigger analysis for case in state '{case_row.status}'. "
            f"Allowed states: {sorted(allowed_from_states)}"
        )

    execution_id = str(uuid.uuid4())
    audit_repo = AuditRepository(db)

    # 3. Synchronously Transition State to ANALYZING and Persist Execution Row
    transition_case_state(
        session=db,
        case_id=id,
        target_state=CaseReviewState.ANALYZING,
        actor_id=current_user.id,
        actor_role=current_user.role,
        audit_repo=audit_repo,
        reason="Clinician triggered multi-agent analysis workflow",
    )

    repo.create_workflow_execution(
        execution_id=execution_id,
        case_id=id,
        is_benchmark=False,
        status="RUNNING",
        idempotency_key=idempotency_key.strip() if idempotency_key else None,
    )

    # 4. Dispatch Asynchronous Background Execution
    background_tasks.add_task(
        _run_workflow_background_task,
        case_id=id,
        execution_id=execution_id,
        raw_text=case_row.chief_complaint,
        idempotency_key=idempotency_key.strip() if idempotency_key else None,
        llm_provider=llm_provider,
    )

    return AnalyzeResponse(
        execution_id=execution_id,
        status="RUNNING",
    )


@router.get(
    "/executions/{id}",
    response_model=ExecutionStatusResponse,
    summary="Retrieve workflow execution status",
    dependencies=[Depends(get_current_user)],
)
def get_execution_status(id: str, db: Session = Depends(get_db)) -> ExecutionStatusResponse:
    """Retrieve top-level workflow execution telemetry and progress."""
    exec_row = db.get(WorkflowExecutionModel, id)
    if not exec_row:
        raise NotFoundAPIError(f"WorkflowExecution '{id}' not found.")

    # Determine last node executed and error code
    agent_stmt = (
        select(AgentExecutionModel)
        .where(AgentExecutionModel.execution_id == id)
        .order_by(desc(AgentExecutionModel.started_at))
    )
    last_agent = db.scalars(agent_stmt).first()
    last_node = last_agent.agent_name if last_agent else None
    error_code = (
        last_agent.error_code
        if (last_agent and last_agent.error_code)
        else exec_row.failure_reason
    )

    return ExecutionStatusResponse(
        id=exec_row.id,
        case_id=exec_row.case_id,
        status=exec_row.status,
        started_at=exec_row.started_at,
        completed_at=exec_row.completed_at,
        error_code=error_code,
        last_node=last_node,
    )


@router.get(
    "/executions/{id}/agents",
    response_model=List[AgentRecordResponse],
    summary="Retrieve per-agent execution telemetry",
    dependencies=[Depends(get_current_user)],
)
def get_execution_agents(id: str, db: Session = Depends(get_db)) -> List[AgentRecordResponse]:
    """Retrieve agent telemetry records for an execution. Never returns prompts or outputs."""
    exec_row = db.get(WorkflowExecutionModel, id)
    if not exec_row:
        raise NotFoundAPIError(f"WorkflowExecution '{id}' not found.")

    stmt = (
        select(AgentExecutionModel)
        .where(AgentExecutionModel.execution_id == id)
        .order_by(AgentExecutionModel.started_at.asc())
    )
    agent_rows = list(db.scalars(stmt).all())

    return [
        AgentRecordResponse(
            agent=a.agent_name,
            status=a.status,
            started_at=a.started_at,
            completed_at=a.completed_at,
            duration_ms=a.duration_ms,
            error_code=a.error_code,
            warnings=a.error_message,
        )
        for a in agent_rows
    ]
