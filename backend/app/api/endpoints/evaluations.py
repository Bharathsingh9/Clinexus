"""Evaluation Endpoints.

Specification Section 7, Section 9, Section 10, App A11-A13 (API-10, EVL-1..6):
- POST /evaluations: Triggers evaluation run as background task with safety gates.
  Admin only. Refusal returns 409 with harness reason code.
- GET /evaluations: Lists completed and running evaluation benchmarks.
- GET /evaluations/{id}: Returns aggregated evaluation metrics with counts and labels
  (UNCONFIRMED, PARTIAL, CLOSED_WORLD_SNIPPET_CORPUS, NON_OFFICIAL_SPLIT,
   BENCHMARK_NOT_CLINICAL_USE, ONLINE).
  STRICT INVARIANT: NEVER returns question text, passage text, or gold text.
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, get_db, require_admin
from backend.app.api.errors import (
    AppAPIException,
    ConflictAPIError,
    NotFoundAPIError,
    ValidationAPIError,
)
from backend.app.core.config import settings
from backend.app.evaluation.harness import EvaluationHarness
from backend.app.persistence.models import (
    EvaluationMetric as EvaluationMetricModel,
    EvaluationRun as EvaluationRunModel,
    User as UserModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluations"])


class TriggerEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    set: str = Field(..., description="Evaluation set: smoke30 | retrieval_dev | retrieval_eval | e2e_sample")
    config: str = Field(..., description="Configuration: retrieval_only | baseline | full | full_online")
    limit: Optional[int] = Field(default=None, ge=1)


class TriggerEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: str
    set_name: str
    config_name: str


class EvaluationRunListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    dataset_name: str
    split_name: str
    set_name: str
    config_name: str
    sample_size: int
    status: str
    labels: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None


class MetricEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float
    question_type: str
    is_unconfirmed: bool


class EvaluationRunDetailResponse(BaseModel):
    """Evaluation metrics summary strictly omitting question text, passage text, or gold answers."""
    model_config = ConfigDict(extra="forbid")
    id: str
    dataset_name: str
    split_name: str
    set_name: str
    config_name: str
    sample_size: int
    status: str
    labels: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    metrics: List[MetricEntry]


def _run_harness_background(set_name: str, config_name: str, limit: Optional[int]) -> None:
    """Execute evaluation harness run in background."""
    harness = EvaluationHarness(settings_obj=settings)
    try:
        if config_name == "retrieval_only":
            harness.run_retrieval_only(set_name=set_name, limit=limit)
        else:
            harness.run_e2e(set_name=set_name, config_name=config_name, limit=limit)
    except Exception as exc:
        logger.error("Background evaluation harness error: %s", exc, exc_info=True)


@router.post(
    "/evaluations",
    response_model=TriggerEvaluationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger BioASQ benchmark evaluation run (Admin only)",
)
def trigger_evaluation(
    payload: TriggerEvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db),
) -> TriggerEvaluationResponse:
    """Trigger an evaluation run with full safety gate verification. Admin-only."""
    valid_sets = {"smoke30", "retrieval_dev", "retrieval_eval", "e2e_sample"}
    valid_configs = {"retrieval_only", "baseline", "full", "full_online"}

    if payload.set not in valid_sets:
        raise ValidationAPIError(f"Unknown evaluation set '{payload.set}'. Allowed: {sorted(valid_sets)}")
    if payload.config not in valid_configs:
        raise ValidationAPIError(f"Unknown evaluation config '{payload.config}'. Allowed: {sorted(valid_configs)}")

    harness = EvaluationHarness(settings_obj=settings)

    # Validate safety gates: Gate A (LLM allowed in manifest), Gate B (human confirmation for eval split), Gate C (call cap)
    try:
        split_name = "dev" if "dev" in payload.set or payload.set == "smoke30" else "eval"
        harness.check_safety_gates(
            set_name=payload.set,
            config_name=payload.config,
            eval_max_llm_calls=settings.EVAL_MAX_LLM_CALLS,
            allow_rerun=None,
            config_hash=None,
        )
    except RuntimeError as r_err:
        err_msg = str(r_err)
        logger.warning("Evaluation safety gate refused execution: %s", err_msg)
        raise AppAPIException(
            status_code=409,
            code="SAFETY_GATE_REFUSAL",
            message=err_msg,
        )

    run_id = f"run_{payload.config}_{payload.set}_{int(time.time())}"

    # Dispatch to background tasks
    background_tasks.add_task(
        _run_harness_background,
        set_name=payload.set,
        config_name=payload.config,
        limit=payload.limit,
    )

    return TriggerEvaluationResponse(
        run_id=run_id,
        status="RUNNING",
        set_name=payload.set,
        config_name=payload.config,
    )


@router.get(
    "/evaluations",
    response_model=List[EvaluationRunListItem],
    summary="List all benchmark evaluation runs",
    dependencies=[Depends(get_current_user)],
)
def list_evaluations(db: Session = Depends(get_db)) -> List[EvaluationRunListItem]:
    """Retrieve all evaluation runs."""
    stmt = select(EvaluationRunModel).order_by(desc(EvaluationRunModel.started_at))
    rows = list(db.scalars(stmt).all())

    results: List[EvaluationRunListItem] = []
    for r in rows:
        meta: Dict[str, Any] = {}
        if r.run_metadata:
            try:
                meta = json.loads(r.run_metadata)
            except Exception:
                meta = {}

        results.append(
            EvaluationRunListItem(
                id=r.id,
                dataset_name=r.dataset_name,
                split_name=r.split_name,
                set_name=meta.get("set_name", "UNKNOWN"),
                config_name=meta.get("config_name", "UNKNOWN"),
                sample_size=r.sample_size,
                status=r.status,
                labels=meta.get("labels", []),
                started_at=r.started_at,
                completed_at=r.completed_at,
            )
        )
    return results


@router.get(
    "/evaluations/{id}",
    response_model=EvaluationRunDetailResponse,
    summary="Retrieve evaluation benchmark metrics",
    dependencies=[Depends(get_current_user)],
)
def get_evaluation_detail(id: str, db: Session = Depends(get_db)) -> EvaluationRunDetailResponse:
    """Retrieve aggregated metrics and labels for a specific run.

    STRICT INVARIANT: Never returns question, passage, or gold answer text.
    """
    run_row = db.get(EvaluationRunModel, id)
    if not run_row:
        raise NotFoundAPIError(f"EvaluationRun '{id}' not found.")

    meta: Dict[str, Any] = {}
    if run_row.run_metadata:
        try:
            meta = json.loads(run_row.run_metadata)
        except Exception:
            meta = {}

    m_stmt = (
        select(EvaluationMetricModel)
        .where(EvaluationMetricModel.evaluation_run_id == id)
        .order_by(EvaluationMetricModel.metric_name.asc())
    )
    metric_rows = list(db.scalars(m_stmt).all())

    metrics: List[MetricEntry] = [
        MetricEntry(
            name=m.metric_name,
            value=m.metric_value,
            question_type=m.question_type,
            is_unconfirmed=m.is_unconfirmed,
        )
        for m in metric_rows
    ]

    return EvaluationRunDetailResponse(
        id=run_row.id,
        dataset_name=run_row.dataset_name,
        split_name=run_row.split_name,
        set_name=meta.get("set_name", "UNKNOWN"),
        config_name=meta.get("config_name", "UNKNOWN"),
        sample_size=run_row.sample_size,
        status=run_row.status,
        labels=meta.get("labels", []),
        started_at=run_row.started_at,
        completed_at=run_row.completed_at,
        metrics=metrics,
    )
