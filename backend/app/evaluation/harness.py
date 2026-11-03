"""Evaluation Harness for BioASQ Benchmark (Phase 5 Stage D).

Specification Section 9.6, 9.7, 9.8, 9.9, 10, Appendix A11, A12, A13 (Req ID: EVL-1..6, SAF-1):
- Evaluation sets:
    * smoke30: 30 candidate questions in file order (dev).
    * retrieval_dev: all 1,133 questions with corpus rows in dev split.
    * retrieval_eval: all 4,532 questions with corpus rows in eval split.
    * e2e_sample: 200 questions drawn strictly from eval, stratified by question type (seed 42).
- Configurations:
    * retrieval_only: Question text embedded as single query, search main collection with RETRIEVAL_TOP_K,
      collapse chunks to corpus rows, score with score_retrieval against derived labels.
      Labeled RETRIEVAL_ONLY_QUESTION_AS_QUERY.
    * baseline: Runs baseline workflow in benchmark mode. Reports retrieval, workflow, confidence,
      iteration, latency, and failure metrics ONLY (no answer metrics).
    * full: Extended workflow in benchmark mode, EXTERNAL_MODE=offline. Calls AnswerExtractor only when
      workflow did not abstain. Gold data read ONLY after workflow and extractor finished.
    * full_online: Extended workflow in benchmark mode, EXTERNAL_MODE=online. Labeled ONLINE.
- Safety Gates:
    * LLM runs require third_party_llm_allowed=true for corpus and questions in MANIFEST.yaml.
    * Eval split runs require confirmed_by_human=true in manifest and evaluation_labels.
    * Eval split runs refuse duplicate run of same configuration hash unless --allow-rerun REASON is provided.
    * E2E runs require EVAL_MAX_LLM_CALLS to be set and positive.
    * Unconfirmed manifests allow dev runs but label outputs UNCONFIRMED.
- Operational Safeguards:
    * --dry-run prints planned question counts and estimated max LLM calls without calling any service.
    * Per-question JSONL checkpointing in git-ignored folder; resume skips completed questions.
    * Circuit breaker: stops on 3 consecutive LLM_AUTH_ERROR or 5 consecutive FAILED questions.
    * Call cap: stops when EVAL_MAX_LLM_CALLS is reached, marking result PARTIAL.
    * Output JSON contains NO question text, NO snippet text, NO gold text.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import yaml

from sqlalchemy import create_engine, select, text

from backend.app.agents.answer_extractor import (
    AnswerExtractor,
    AnswerExtractorInput,
    ExtractedReportClaim,
)
from backend.app.core.config import Settings, settings
from backend.app.core.llm_exceptions import LLMAuthError, LLMInvalidResponse
from backend.app.core.llm_provider import LLMProvider, create_llm_provider
from backend.app.evidence_sources.ncbi_client import NCBIClient, NCBIClientProtocol
from backend.app.orchestration.baseline_workflow import BaselineWorkflowOrchestrator
from backend.app.orchestration.extended_workflow import ExtendedWorkflowOrchestrator
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.execution_repository import ExecutionRepository
from backend.app.persistence.models import (
    EvaluationLabel,
    EvaluationMetric,
    EvaluationRun,
)
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name
from backend.app.schemas.clinical_case import CaseSubmission
from backend.app.schemas.executions import AgentStatusEnum, WorkflowState
from backend.app.schemas.reports import (
    ClaimVerificationStatusEnum,
    ReportStatusEnum,
)
from backend.app.evaluation.bioasq_scoring import (
    SCORING_BASIS,
    score_factoid,
    score_list,
    score_retrieval,
    score_summary,
    score_yesno,
)
from backend.app.evaluation.splits import (
    build_e2e_sample,
    build_smoke30_sample,
    compute_file_sha256,
    load_smoke30_candidate_ids,
)

logger = logging.getLogger(__name__)

WHITELISTED_SETTINGS = [
    "PROJECT_NAME",
    "ENVIRONMENT",
    "QDRANT_COLLECTION_PREFIX",
    "QDRANT_COLLECTION_VERSION",
    "EMBEDDING_MODEL",
    "EMBED_BATCH_SIZE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_MODEL_CLARIFIER",
    "LLM_MODEL_RAG",
    "LLM_MODEL_SCANNER",
    "LLM_MODEL_FUSION",
    "LLM_MODEL_OPTIMIZER",
    "VERIFIER_MODEL",
    "LLM_MODEL_EXTRACTOR",
    "MAX_OPTIMIZER_ITERATIONS",
    "CONFIDENCE_THRESHOLD",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_MIN_SCORE",
    "RETRIEVAL_SCORE_FLOOR",
    "RETRIEVAL_SCORE_CEILING",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "MIN_RELEVANT_PASSAGES",
    "EXTERNAL_MODE",
    "CONF_WEIGHT_CLAIM_SUPPORT",
    "CONF_WEIGHT_RETRIEVAL_STRENGTH",
    "CONF_WEIGHT_SOURCE_AVAILABILITY",
    "CONF_WEIGHT_INFO_COMPLETENESS",
    "EVAL_MAX_LLM_CALLS",
    "EVAL_SAMPLE_SIZE",
    "LLM_TEMPERATURE",
    "LLM_THINKING_LEVEL",
    "LLM_MAX_RETRY_WAIT_S",
    "LLM_MAX_CALL_SECONDS",
    "temperature_sent",
    "thinking_level",
    "model_name",
]

DEFAULT_SOURCE_FILES = {
    "corpus": "datasets/processed/bioasq/bioasq_evidence_corpus.csv",
    "questions": "datasets/processed/bioasq/bioasq_normalized.csv",
    "candidates": "datasets/processed/bioasq/retrieval_eval_candidates.csv",
    "splits": "datasets/processed/splits_v1.json",
    "manifest": "datasets/MANIFEST.yaml",
}


def get_git_info() -> Tuple[str, bool]:
    """Retrieve current git commit hash and dirty status safely."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        commit = "UNKNOWN_COMMIT"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        dirty = len(status_out) > 0
    except Exception:
        dirty = True

    return commit, dirty


def compute_settings_snapshot(settings_obj: Settings) -> Dict[str, Any]:
    """Return whitelisted non-secret settings snapshot."""
    snapshot: Dict[str, Any] = {}
    for key in WHITELISTED_SETTINGS:
        val = getattr(settings_obj, key, None)
        if isinstance(val, (int, float, str, bool)) or val is None:
            snapshot[key] = val
        else:
            snapshot[key] = str(val)
    return snapshot


def compute_configuration_hash(
    dataset_name: str,
    split_name: str,
    set_name: str,
    config_name: str,
    models_dict: Dict[str, str],
    settings_snapshot: Dict[str, Any],
    corpus_version: str,
    source_checksums: Dict[str, str],
) -> str:
    """Compute a reproducible SHA-256 hash identifying the evaluation configuration."""
    payload = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "set_name": set_name,
        "config_name": config_name,
        "models": models_dict,
        "settings": settings_snapshot,
        "corpus_version": corpus_version,
        "source_checksums": source_checksums,
    }
    dumped = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class EvaluationHarness:
    """BioASQ Evaluation Harness implementing SPEC Section 9, 10, Appendix A11-A13."""

    def __init__(
        self,
        settings_obj: Optional[Settings] = None,
        source_files: Optional[Dict[str, str]] = None,
        llm_provider: Optional[LLMProvider] = None,
        ncbi_client: Optional[NCBIClientProtocol] = None,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_wrapper: Optional[QdrantWrapper] = None,
        output_dir: str = "evaluation_results",
    ):
        self.settings = settings_obj or settings
        self.source_files = source_files or DEFAULT_SOURCE_FILES
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._llm_provider = llm_provider
        self._ncbi_client = ncbi_client
        self._embedding_service = embedding_service
        self._qdrant_wrapper = qdrant_wrapper

    def get_source_checksums(self) -> Dict[str, str]:
        """Compute SHA-256 for all key source files."""
        checksums = {}
        for key, filepath in self.source_files.items():
            if os.path.exists(filepath):
                checksums[key] = compute_file_sha256(filepath)
            else:
                checksums[key] = "FILE_NOT_FOUND"
        return checksums

    def inspect_manifest(self) -> Dict[str, Any]:
        """Inspect datasets/MANIFEST.yaml for safety gate conditions."""
        manifest_path = self.source_files.get("manifest", "datasets/MANIFEST.yaml")
        if not os.path.exists(manifest_path):
            return {
                "exists": False,
                "all_confirmed": False,
                "corpus_llm_allowed": False,
                "questions_llm_allowed": False,
                "corpus_confirmed": False,
                "questions_confirmed": False,
            }

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = yaml.safe_load(f)

        files = manifest_data.get("files", [])
        corpus_entry = next((e for e in files if e.get("name") == "bioasq_evidence_corpus.csv"), None)
        questions_entry = next((e for e in files if e.get("name") == "bioasq_normalized.csv"), None)

        corpus_llm = bool(corpus_entry.get("third_party_llm_allowed", False)) if corpus_entry else False
        questions_llm = bool(questions_entry.get("third_party_llm_allowed", False)) if questions_entry else False

        corpus_conf = bool(corpus_entry.get("confirmed_by_human", False)) if corpus_entry else False
        questions_conf = bool(questions_entry.get("confirmed_by_human", False)) if questions_entry else False

        all_conf = all(bool(e.get("confirmed_by_human", False)) for e in files)

        return {
            "exists": True,
            "all_confirmed": all_conf,
            "corpus_llm_allowed": corpus_llm,
            "questions_llm_allowed": questions_llm,
            "corpus_confirmed": corpus_conf,
            "questions_confirmed": questions_conf,
        }

    def check_safety_gates(
        self,
        set_name: str,
        config_name: str,
        eval_max_llm_calls: Optional[int],
        allow_rerun: Optional[str] = None,
        config_hash: Optional[str] = None,
    ) -> List[str]:
        """Validate safety gates (Gate A, B, C, D) per SPEC Appendix A13.

        Returns:
            list of active labels (e.g. ['UNCONFIRMED', 'ONLINE', ...])
        Raises:
            RuntimeError on fatal gate refusal.
        """
        labels: List[str] = [
            "CLOSED_WORLD_SNIPPET_CORPUS",
            "NON_OFFICIAL_SPLIT",
            "BENCHMARK_NOT_CLINICAL_USE",
        ]
        manifest_status = self.inspect_manifest()

        # Gate (a): LLM calls require third_party_llm_allowed true
        is_llm_run = config_name in ("baseline", "full", "full_online")
        if is_llm_run:
            if not (manifest_status["corpus_llm_allowed"] and manifest_status["questions_llm_allowed"]):
                raise RuntimeError(
                    "Safety gate refusal: Configuration requires calling an LLM, but 'third_party_llm_allowed' "
                    "is not true for both corpus and questions entries in datasets/MANIFEST.yaml. "
                    "Refusing execution."
                )

        # Gate (b): Eval split runs require confirmed_by_human: true and unique config_hash
        is_eval_split = set_name in ("retrieval_eval", "e2e_sample")
        if is_eval_split:
            if not (manifest_status["corpus_confirmed"] and manifest_status["questions_confirmed"]):
                raise RuntimeError(
                    "Safety gate refusal: Eval-split evaluation requires confirmed_by_human: true for corpus "
                    "and questions manifest entries. Refusing execution."
                )

            # Check evaluation_labels table confirmation
            with SessionLocal() as session:
                unconf_cnt = session.execute(
                    select(EvaluationLabel).where(EvaluationLabel.confirmed_by_human == False)  # noqa: E712
                ).first()
                if unconf_cnt is not None:
                    raise RuntimeError(
                        "Safety gate refusal: Eval-split evaluation requires confirmed_by_human: true in "
                        "evaluation_labels table. Refusing execution."
                    )

            # Check rerun duplicate
            if config_hash:
                with SessionLocal() as session:
                    existing = session.execute(
                        select(EvaluationRun).where(
                            EvaluationRun.status == "COMPLETED",
                            EvaluationRun.run_metadata.like(f"%{config_hash}%"),
                        )
                    ).first()
                    if existing and not allow_rerun:
                        raise RuntimeError(
                            f"Safety gate refusal: Configuration hash '{config_hash}' has already completed a final "
                            "run on the eval split. A second run is refused unless --allow-rerun REASON is specified."
                        )

        # Gate (c): e2e runs require EVAL_MAX_LLM_CALLS
        if is_llm_run:
            effective_cap = eval_max_llm_calls or self.settings.EVAL_MAX_LLM_CALLS
            if effective_cap is None or effective_cap <= 0:
                raise RuntimeError(
                    "Safety gate refusal: End-to-end evaluation runs require setting 'EVAL_MAX_LLM_CALLS' "
                    "to a positive integer before execution."
                )

        # Gate (d): UNCONFIRMED label if manifest not confirmed
        if not manifest_status["all_confirmed"]:
            labels.append("UNCONFIRMED")

        if config_name == "full_online":
            labels.append("ONLINE")

        return labels

    def load_questions_data(self) -> Dict[str, Dict[str, Any]]:
        """Load bioasq_normalized.csv into memory keyed by sample_id."""
        import csv
        questions_file = self.source_files["questions"]
        by_id: Dict[str, Dict[str, Any]] = {}
        for enc in ("utf-8", "utf-8-sig"):
            try:
                with open(questions_file, mode="r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        sid = r["sample_id"].strip()
                        by_id[sid] = {
                            "sample_id": sid,
                            "question_type": r.get("question_type", "").strip().lower(),
                            "question": r.get("question", "").strip(),
                        }
                break
            except UnicodeDecodeError:
                continue
        return by_id

    def load_evaluation_set(self, set_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Resolve the list of question objects for the chosen set."""
        questions_by_id = self.load_questions_data()

        if set_name == "smoke30":
            candidates_path = self.source_files["candidates"]
            smoke_ids = load_smoke30_candidate_ids(candidates_path)
            q_list = [questions_by_id[sid] for sid in smoke_ids if sid in questions_by_id]
        elif set_name in ("retrieval_dev", "retrieval_eval"):
            splits_path = self.source_files["splits"]
            with open(splits_path, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            key = "dev_ids" if set_name == "retrieval_dev" else "eval_ids"
            target_ids = splits_data.get(key, [])
            q_list = [questions_by_id[sid] for sid in target_ids if sid in questions_by_id]
        elif set_name == "e2e_sample":
            splits_path = self.source_files["splits"]
            with open(splits_path, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            eval_ids = splits_data.get("eval_ids", [])
            sample_ids = build_e2e_sample(
                eval_ids=eval_ids,
                questions_by_id=questions_by_id,
                sample_size=self.settings.EVAL_SAMPLE_SIZE,
                seed=42,
            )
            q_list = [questions_by_id[sid] for sid in sample_ids if sid in questions_by_id]
        else:
            raise ValueError(f"Unknown evaluation set name: {set_name}")

        if limit is not None and limit > 0:
            q_list = q_list[:limit]
        return q_list

    def dry_run(
        self,
        set_name: str,
        config_name: str,
        limit: Optional[int] = None,
        eval_max_llm_calls: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform dry run calculation without calling vector DB, NCBI, or LLM."""
        questions = self.load_evaluation_set(set_name, limit=limit)
        q_count = len(questions)
        type_counts = Counter(q["question_type"] for q in questions)

        if config_name == "retrieval_only":
            max_llm_calls = 0
            calls_per_question = 0
        elif config_name == "baseline":
            # Up to (MAX_OPTIMIZER_ITERATIONS + 1) loops * ~3 agents (Clarifier, Fusion, Optimizer)
            calls_per_question = (self.settings.MAX_OPTIMIZER_ITERATIONS + 1) * 3
            max_llm_calls = q_count * calls_per_question
        else:
            # Full / full_online: (MAX_ITER + 1) * 4 agents (Clarifier, Fusion, Verifier, Optimizer) + 1 Extractor
            calls_per_question = ((self.settings.MAX_OPTIMIZER_ITERATIONS + 1) * 4) + 1
            max_llm_calls = q_count * calls_per_question

        effective_cap = eval_max_llm_calls or self.settings.EVAL_MAX_LLM_CALLS
        if effective_cap is not None and max_llm_calls > effective_cap:
            est_call_ceiling = effective_cap
        else:
            est_call_ceiling = max_llm_calls

        result = {
            "set_name": set_name,
            "config_name": config_name,
            "planned_question_count": q_count,
            "question_counts_by_type": dict(type_counts),
            "estimated_calls_per_question_worst_case": calls_per_question,
            "estimated_max_llm_calls_uncapped": max_llm_calls,
            "effective_eval_max_llm_calls_cap": effective_cap,
            "estimated_call_ceiling": est_call_ceiling,
            "is_llm_required": config_name in ("baseline", "full", "full_online"),
        }
        return result

    def run_retrieval_only(
        self,
        set_name: str,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Execute retrieval_only dev or eval evaluation."""
        run_id = f"run_retrieval_only_{set_name}_{int(time.time())}"
        config_name = "retrieval_only"
        split_name = "dev" if "dev" in set_name or set_name == "smoke30" else "eval"

        source_checksums = self.get_source_checksums()
        git_commit, git_dirty = get_git_info()
        settings_snap = compute_settings_snapshot(self.settings)

        corpus_version = f"{source_checksums.get('corpus', 'UNKNOWN')}_{self.settings.EMBEDDING_MODEL}_v{self.settings.QDRANT_COLLECTION_VERSION}"
        models_dict = {
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "llm_provider": "none",
        }
        config_hash = compute_configuration_hash(
            dataset_name="bioasq",
            split_name=split_name,
            set_name=set_name,
            config_name=config_name,
            models_dict=models_dict,
            settings_snapshot=settings_snap,
            corpus_version=corpus_version,
            source_checksums=source_checksums,
        )

        labels = self.check_safety_gates(
            set_name=set_name,
            config_name=config_name,
            eval_max_llm_calls=None,
            config_hash=config_hash,
        )
        labels.append("RETRIEVAL_ONLY_QUESTION_AS_QUERY")

        questions = self.load_evaluation_set(set_name, limit=limit)
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{run_id}.jsonl")

        completed_records: Dict[str, Dict[str, Any]] = {}
        if resume and os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed_records[r["sample_id"]] = r

        embedder = self._embedding_service or EmbeddingService(model_name=self.settings.EMBEDDING_MODEL)
        qdrant = self._qdrant_wrapper or QdrantWrapper(
            url=self.settings.QDRANT_URL,
            api_key=self.settings.QDRANT_API_KEY,
        )
        collection_name = format_collection_name(
            prefix=self.settings.QDRANT_COLLECTION_PREFIX,
            model_name=self.settings.EMBEDDING_MODEL,
            version=self.settings.QDRANT_COLLECTION_VERSION,
        )

        # Pre-fetch relevant passages from evaluation_labels in one query
        sample_ids = [q["sample_id"] for q in questions]
        eval_labels_map: Dict[str, List[str]] = {}
        with SessionLocal() as session:
            stmt = select(EvaluationLabel.sample_id, EvaluationLabel.relevant_passages).where(
                EvaluationLabel.sample_id.in_(sample_ids)
            )
            for row in session.execute(stmt):
                eval_labels_map[row[0]] = row[1] or []

        records_for_scoring: List[Dict[str, Any]] = []
        per_record_outputs: List[Dict[str, Any]] = []

        # Recalibration statistics collectors
        score_stats = {
            "relevant_in_top_k": {"top1": [], "mean_top3": []},
            "no_relevant_in_top_k": {"top1": [], "mean_top3": []},
        }

        start_time = datetime.now(timezone.utc)

        with open(checkpoint_file, "a", encoding="utf-8") as cp_f:
            for q in questions:
                sid = q["sample_id"]
                q_type = q["question_type"]

                if sid in completed_records:
                    rec = completed_records[sid]
                    per_record_outputs.append(rec)
                    records_for_scoring.append(rec["_scoring_payload"])
                    continue

                t0 = time.monotonic()
                query_vec = embedder.embed_text(q["question"])
                hits = qdrant.search(
                    collection_name=collection_name,
                    query_vector=query_vec,
                    limit=self.settings.RETRIEVAL_TOP_K,
                )
                duration_ms = int((time.monotonic() - t0) * 1000)

                raw_relevant_passages = eval_labels_map.get(sid, [])
                relevant_row_ids = set()
                for p in raw_relevant_passages:
                    parts = p.split(":")
                    if len(parts) >= 3:
                        relevant_row_ids.add(parts[2])
                    else:
                        relevant_row_ids.add(p)

                retrieved_chunks = []
                best_row_scores: Dict[str, float] = {}
                for h in hits:
                    ev_id = h.payload.get("evidence_id", "")
                    sc = float(h.score)
                    retrieved_chunks.append({"evidence_id": ev_id, "retrieval_score": sc})
                    parts = ev_id.split(":")
                    row_id = parts[2] if len(parts) >= 3 else ev_id
                    if row_id not in best_row_scores or sc > best_row_scores[row_id]:
                        best_row_scores[row_id] = sc

                # Score metrics for this question
                sorted_scores = sorted(best_row_scores.values(), reverse=True)
                top1 = sorted_scores[0] if sorted_scores else 0.0
                mean_top3 = (sum(sorted_scores[:3]) / len(sorted_scores[:3])) if sorted_scores else 0.0

                has_rel_in_top_k = bool(relevant_row_ids.intersection(set(best_row_scores.keys())))
                if has_rel_in_top_k:
                    score_stats["relevant_in_top_k"]["top1"].append(top1)
                    score_stats["relevant_in_top_k"]["mean_top3"].append(mean_top3)
                else:
                    score_stats["no_relevant_in_top_k"]["top1"].append(top1)
                    score_stats["no_relevant_in_top_k"]["mean_top3"].append(mean_top3)

                scoring_payload = {
                    "sample_id": sid,
                    "question_type": q_type,
                    "relevant_row_ids": list(relevant_row_ids),
                    "retrieved_chunks": retrieved_chunks,
                    "duplicate_groups": {},
                }
                records_for_scoring.append(scoring_payload)

                rec_out = {
                    "sample_id": sid,
                    "question_type": q_type,
                    "workflow_status": "READY_FOR_REVIEW",
                    "gate_rule": 6,
                    "reason_codes": [],
                    "claim_status_counts": {},
                    "iteration_count": 0,
                    "extracted_answer": None,
                    "correctness_flags": {},
                    "error_code": None,
                    "durations": {"retrieval_ms": duration_ms},
                    "llm_calls": 0,
                    "llm_tokens": {"input": 0, "output": 0},
                    "top_1_score": round(top1, 4),
                    "mean_top_3_score": round(mean_top3, 4),
                    "has_relevant_in_top_k": has_rel_in_top_k,
                }
                per_record_outputs.append(rec_out)

                # Write checkpoint JSON line (internal payload stripped from final output)
                cp_line = dict(rec_out)
                cp_line["_scoring_payload"] = scoring_payload
                cp_f.write(json.dumps(cp_line) + "\n")
                cp_f.flush()

        completed_time = datetime.now(timezone.utc)

        # Compute retrieval metrics overall and per type
        overall_res = score_retrieval(records_for_scoring)

        type_metrics: Dict[str, Any] = {}
        for qt in ("factoid", "yesno", "summary", "list"):
            qt_records = [r for r in records_for_scoring if r["question_type"] == qt]
            if qt_records:
                qt_res = score_retrieval(qt_records)
                type_metrics[qt] = {
                    "recall_at_5": {"value": qt_res.recall_at_5.value, "count": qt_res.recall_at_5.num_questions},
                    "recall_at_10": {"value": qt_res.recall_at_10.value, "count": qt_res.recall_at_10.num_questions},
                    "precision_at_5": {"value": qt_res.precision_at_5.value, "count": qt_res.precision_at_5.num_questions},
                    "mrr": {"value": qt_res.mrr.value, "count": qt_res.mrr.num_questions},
                    "map": {"value": qt_res.map.value, "count": qt_res.map.num_questions},
                    "included_questions": qt_res.included_questions,
                    "excluded_no_relevant_rows": qt_res.excluded_no_relevant_rows,
                }

        aggregated_metrics = {
            "overall": {
                "recall_at_5": {"value": overall_res.recall_at_5.value, "count": overall_res.recall_at_5.num_questions},
                "recall_at_10": {"value": overall_res.recall_at_10.value, "count": overall_res.recall_at_10.num_questions},
                "precision_at_5": {"value": overall_res.precision_at_5.value, "count": overall_res.precision_at_5.num_questions},
                "mrr": {"value": overall_res.mrr.value, "count": overall_res.mrr.num_questions},
                "map": {"value": overall_res.map.value, "count": overall_res.map.num_questions},
                "included_questions": overall_res.included_questions,
                "excluded_no_relevant_rows": overall_res.excluded_no_relevant_rows,
            },
            "by_question_type": type_metrics,
            "scoring_basis": SCORING_BASIS,
        }

        # Compute percentile distribution for recalibration
        def calc_percentiles(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "count": 0}
            s = sorted(vals)
            n = len(s)
            def p_at(pct: float) -> float:
                k = (n - 1) * (pct / 100.0)
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return float(s[int(k)])
                return float(s[f] * (c - k) + s[c] * (k - f))

            return {
                "count": n,
                "p5": round(p_at(5), 4),
                "p25": round(p_at(25), 4),
                "p50": round(p_at(50), 4),
                "p75": round(p_at(75), 4),
                "p95": round(p_at(95), 4),
            }

        recalibration_percentiles = {
            "relevant_in_top_k": {
                "top1": calc_percentiles(score_stats["relevant_in_top_k"]["top1"]),
                "mean_top3": calc_percentiles(score_stats["relevant_in_top_k"]["mean_top3"]),
            },
            "no_relevant_in_top_k": {
                "top1": calc_percentiles(score_stats["no_relevant_in_top_k"]["top1"]),
                "mean_top3": calc_percentiles(score_stats["no_relevant_in_top_k"]["mean_top3"]),
            },
        }

        final_run_data = {
            "run_id": run_id,
            "dataset": "bioasq",
            "split": split_name,
            "set_name": set_name,
            "config_name": config_name,
            "status": "COMPLETED",
            "seed": 42,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "settings_snapshot": settings_snap,
            "models": models_dict,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "corpus_version": corpus_version,
            "source_checksums": source_checksums,
            "config_hash": config_hash,
            "labels": labels,
            "record_counts": {
                "total_planned": len(questions),
                "total_evaluated": len(per_record_outputs),
                "included_in_metrics": overall_res.included_questions,
                "excluded_no_relevant_rows": overall_res.excluded_no_relevant_rows,
            },
            "aggregated_metrics": aggregated_metrics,
            "recalibration_percentiles": recalibration_percentiles,
            "error_counts": {},
            "total_llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "per_record_outputs": per_record_outputs,
        }

        # Persist JSON file
        out_json_path = os.path.join(self.output_dir, f"{run_id}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_run_data, f, indent=2)

        # Persist into Postgres evaluation_runs and evaluation_metrics
        with SessionLocal() as session:
            db_run = EvaluationRun(
                id=str(uuid.uuid4()),
                dataset_name="bioasq",
                split_name=split_name,
                sample_size=len(questions),
                corpus_version=corpus_version,
                embedding_model=self.settings.EMBEDDING_MODEL,
                run_metadata=json.dumps({
                    "config_name": config_name,
                    "set_name": set_name,
                    "config_hash": config_hash,
                    "labels": labels,
                    "json_file": out_json_path,
                }),
                started_at=start_time,
                completed_at=completed_time,
                status="COMPLETED",
            )
            session.add(db_run)
            session.flush()

            is_unconf = "UNCONFIRMED" in labels
            # Save overall metrics
            for m_name in ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "map"):
                m_obj = getattr(overall_res, m_name)
                session.add(EvaluationMetric(
                    id=str(uuid.uuid4()),
                    evaluation_run_id=db_run.id,
                    question_type="overall",
                    metric_name=m_name,
                    metric_value=float(m_obj.value),
                    is_unconfirmed=is_unconf,
                ))

            # Save per-type metrics
            for qt, qm in type_metrics.items():
                for m_name in ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "map"):
                    session.add(EvaluationMetric(
                        id=str(uuid.uuid4()),
                        evaluation_run_id=db_run.id,
                        question_type=qt,
                        metric_name=m_name,
                        metric_value=float(qm[m_name]["value"]),
                        is_unconfirmed=is_unconf,
                    ))

            session.commit()

        return final_run_data

    def run_e2e(
        self,
        set_name: str,
        config_name: str,
        limit: Optional[int] = None,
        eval_max_llm_calls: Optional[int] = None,
        allow_rerun: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Execute end-to-end evaluation run (baseline, full, full_online)."""
        run_id = f"run_{config_name}_{set_name}_{int(time.time())}"
        split_name = "dev" if "dev" in set_name or set_name == "smoke30" else "eval"

        effective_call_cap = eval_max_llm_calls or self.settings.EVAL_MAX_LLM_CALLS

        source_checksums = self.get_source_checksums()
        git_commit, git_dirty = get_git_info()
        settings_snap = compute_settings_snapshot(self.settings)

        corpus_version = f"{source_checksums.get('corpus', 'UNKNOWN')}_{self.settings.EMBEDDING_MODEL}_v{self.settings.QDRANT_COLLECTION_VERSION}"
        models_dict = {
            "llm_provider": self.settings.LLM_PROVIDER,
            "llm_model": self.settings.LLM_MODEL,
            "clarifier": self.settings.LLM_MODEL_CLARIFIER or self.settings.LLM_MODEL,
            "rag": self.settings.LLM_MODEL_RAG or self.settings.LLM_MODEL,
            "scanner": self.settings.LLM_MODEL_SCANNER or self.settings.LLM_MODEL,
            "fusion": self.settings.LLM_MODEL_FUSION or self.settings.LLM_MODEL,
            "optimizer": self.settings.LLM_MODEL_OPTIMIZER or self.settings.LLM_MODEL,
            "verifier": self.settings.VERIFIER_MODEL or self.settings.LLM_MODEL,
            "extractor": self.settings.LLM_MODEL_EXTRACTOR or self.settings.LLM_MODEL,
        }
        config_hash = compute_configuration_hash(
            dataset_name="bioasq",
            split_name=split_name,
            set_name=set_name,
            config_name=config_name,
            models_dict=models_dict,
            settings_snapshot=settings_snap,
            corpus_version=corpus_version,
            source_checksums=source_checksums,
        )

        labels = self.check_safety_gates(
            set_name=set_name,
            config_name=config_name,
            eval_max_llm_calls=effective_call_cap,
            allow_rerun=allow_rerun,
            config_hash=config_hash,
        )

        questions = self.load_evaluation_set(set_name, limit=limit)
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{run_id}.jsonl")

        completed_records: Dict[str, Dict[str, Any]] = {}
        if resume and os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed_records[r["sample_id"]] = r

        provider = self._llm_provider or create_llm_provider(self.settings)
        ncbi = self._ncbi_client or NCBIClient(email=self.settings.NCBI_EMAIL, api_key=self.settings.NCBI_API_KEY)
        embedder = self._embedding_service or EmbeddingService(model_name=self.settings.EMBEDDING_MODEL)
        qdrant = self._qdrant_wrapper or QdrantWrapper(
            url=self.settings.QDRANT_URL,
            api_key=self.settings.QDRANT_API_KEY,
        )
        repo = ExecutionRepository()

        if config_name == "baseline":
            orchestrator = BaselineWorkflowOrchestrator(
                llm_provider=provider,
                ncbi_client=ncbi,
                repository=repo,
                embedding_service=embedder,
                qdrant_wrapper=qdrant,
            )
            extractor = None
        else:
            orchestrator = ExtendedWorkflowOrchestrator(
                llm_provider=provider,
                ncbi_client=ncbi,
                repository=repo,
                embedding_service=embedder,
                qdrant_wrapper=qdrant,
                collection_version=self.settings.QDRANT_COLLECTION_VERSION,
            )
            extractor = AnswerExtractor(
                llm_provider=provider,
                model_name=self.settings.LLM_MODEL_EXTRACTOR or self.settings.LLM_MODEL,
            )

        consecutive_auth_errors = 0
        consecutive_failures = 0
        total_calls_tracked = 0
        total_in_tokens = 0
        total_out_tokens = 0
        total_thinking_tokens = 0
        error_counts: Dict[str, int] = defaultdict(int)

        per_record_outputs: List[Dict[str, Any]] = []
        is_partial = False
        partial_reason = ""

        # Pre-load gold data from evaluation_labels, but KEEP IN STRICT ISOLATION from workflow
        sample_ids = [q["sample_id"] for q in questions]
        eval_labels_gold: Dict[str, EvaluationLabel] = {}
        with SessionLocal() as session:
            stmt = select(EvaluationLabel).where(EvaluationLabel.sample_id.in_(sample_ids))
            for el in session.scalars(stmt):
                eval_labels_gold[el.sample_id] = el

        start_time = datetime.now(timezone.utc)

        with open(checkpoint_file, "a", encoding="utf-8") as cp_f:
            for q in questions:
                sid = q["sample_id"]
                q_type = q["question_type"]

                if sid in completed_records:
                    rec = completed_records[sid]
                    per_record_outputs.append(rec)
                    total_calls_tracked += rec.get("llm_calls", 0)
                    total_in_tokens += rec.get("llm_tokens", {}).get("input", 0)
                    total_out_tokens += rec.get("llm_tokens", {}).get("output", 0)
                    total_thinking_tokens += rec.get("llm_tokens", {}).get("thinking", 0)
                    continue

                # Circuit breaker check: call cap
                if effective_call_cap and total_calls_tracked >= effective_call_cap:
                    is_partial = True
                    partial_reason = f"Call cap reached: EVAL_MAX_LLM_CALLS={effective_call_cap}"
                    break

                # 1. Build CaseSubmission strictly from question text (INVARIANT: gold data never reaches workflow)
                sub = CaseSubmission(raw_text=q["question"])
                t_wf_start = time.monotonic()

                # Snapshot usage before workflow run
                usage_before = provider.get_usage() if hasattr(provider, "get_usage") else {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}
                calls_before = getattr(provider, "_call_counter", 0)

                # 2. Run workflow in benchmark mode
                try:
                    if config_name == "baseline":
                        init_state = WorkflowState(
                            case_id=str(uuid.uuid4()),
                            submission=sub,
                            is_benchmark=True,
                            workflow_status="RUNNING",
                        )
                        wf_res: WorkflowState = orchestrator.run(initial_state=init_state)
                    else:
                        wf_res: WorkflowState = orchestrator.run(
                            submission=sub,
                            is_benchmark=True,
                        )
                    wf_duration_ms = int((time.monotonic() - t_wf_start) * 1000)
                except Exception as exc:
                    wf_duration_ms = int((time.monotonic() - t_wf_start) * 1000)
                    err_name = type(exc).__name__
                    error_counts[err_name] += 1
                    consecutive_failures += 1
                    if isinstance(exc, LLMAuthError):
                        consecutive_auth_errors += 1
                    else:
                        consecutive_auth_errors = 0

                    rec_fail = {
                        "sample_id": sid,
                        "question_type": q_type,
                        "workflow_status": "FAILED",
                        "gate_rule": 0,
                        "reason_codes": ["SYSTEM_ERROR"],
                        "claim_status_counts": {},
                        "iteration_count": 0,
                        "extracted_answer": None,
                        "correctness_flags": {},
                        "error_code": err_name,
                        "durations": {"workflow_ms": wf_duration_ms, "extractor_ms": 0, "total_ms": wf_duration_ms},
                        "llm_calls": 0,
                        "llm_tokens": {"input": 0, "output": 0, "thinking": 0},
                    }
                    per_record_outputs.append(rec_fail)
                    cp_f.write(json.dumps(rec_fail) + "\n")
                    cp_f.flush()

                    if (
                        type(exc).__name__ in ("LLMQuotaExhausted", "LLM_QUOTA_EXHAUSTED")
                        or getattr(exc, "error_code", "") == "LLM_QUOTA_EXHAUSTED"
                    ):
                        is_partial = True
                        partial_reason = "Daily quota exhausted (LLM_QUOTA_EXHAUSTED). Stopping evaluation run."
                        break
                    if consecutive_auth_errors >= 3:
                        is_partial = True
                        partial_reason = "Circuit breaker triggered: 3 consecutive LLM_AUTH_ERROR exceptions."
                        break
                    if consecutive_failures >= 5:
                        is_partial = True
                        partial_reason = "Circuit breaker triggered: 5 consecutive FAILED questions."
                        break
                    continue

                usage_after_wf = provider.get_usage() if hasattr(provider, "get_usage") else {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}
                calls_after_wf = getattr(provider, "_call_counter", 0)

                wf_status = wf_res.workflow_status
                if wf_status == "FAILED":
                    consecutive_failures += 1
                    err_code = wf_res.error_code or "FAILED"
                    error_counts[err_code] += 1
                    if "AUTH" in err_code:
                        consecutive_auth_errors += 1
                    else:
                        consecutive_auth_errors = 0
                    if err_code == "LLM_QUOTA_EXHAUSTED":
                        is_partial = True
                        partial_reason = "Daily quota exhausted (LLM_QUOTA_EXHAUSTED). Stopping evaluation run."
                else:
                    consecutive_failures = 0
                    consecutive_auth_errors = 0

                # Extract gate rule and claim status counts
                gate_rule = 6
                if wf_res.abstention:
                    gate_rule = getattr(wf_res.abstention, "rule_triggered", 0) or 0

                claim_counts: Dict[str, int] = defaultdict(int)
                verif_map = {v.claim_id: v.status for v in wf_res.verifications}
                if wf_res.report and wf_res.report.claims:
                    for c in wf_res.report.claims:
                        st = verif_map.get(c.claim_id)
                        st_name = st.value if hasattr(st, "value") else (str(st) if st else "UNVERIFIED")
                        claim_counts[st_name] += 1

                # 3. Answer Extraction (full / full_online only, when workflow did NOT abstain)
                extracted_ans = None
                extractor_ms = 0
                extractor_err = None

                if extractor and wf_status == ReportStatusEnum.READY_FOR_REVIEW.value:
                    # Filter supported / partially supported claims only
                    supported_claims: List[ExtractedReportClaim] = []
                    if wf_res.report and wf_res.report.claims:
                        for c in wf_res.report.claims:
                            v_st = verif_map.get(c.claim_id)
                            if v_st in (ClaimVerificationStatusEnum.SUPPORTED, ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED):
                                supported_claims.append(
                                    ExtractedReportClaim(claim_text=c.text, status=v_st)
                                )

                    if supported_claims:
                        t_ext_start = time.monotonic()
                        try:
                            ext_inp = AnswerExtractorInput(
                                question_type=q_type,
                                question_text=q["question"],
                                verified_claims=supported_claims,
                            )
                            # AnswerExtractor.extract is async
                            import asyncio
                            ext_res = asyncio.run(extractor.extract(ext_inp))
                            extractor_ms = int((time.monotonic() - t_ext_start) * 1000)
                            if ext_res.status == AgentStatusEnum.COMPLETED:
                                extracted_ans = ext_res.extracted_answer
                            else:
                                extractor_err = ext_res.error_code
                                error_counts[ext_res.error_code or "EXTRACTOR_ERROR"] += 1
                        except Exception as e_err:
                            extractor_ms = int((time.monotonic() - t_ext_start) * 1000)
                            extractor_err = type(e_err).__name__
                            error_counts[extractor_err] += 1

                usage_final = provider.get_usage() if hasattr(provider, "get_usage") else {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}
                calls_final = getattr(provider, "_call_counter", 0)

                q_calls = max(0, calls_final - calls_before)
                q_in_tokens = max(0, usage_final.get("total_input_tokens", 0) - usage_before.get("total_input_tokens", 0))
                q_out_tokens = max(0, usage_final.get("total_output_tokens", 0) - usage_before.get("total_output_tokens", 0))
                q_thinking_tokens = max(0, usage_final.get("total_thinking_tokens", 0) - usage_before.get("total_thinking_tokens", 0))

                total_calls_tracked += q_calls
                total_in_tokens += q_in_tokens
                total_out_tokens += q_out_tokens
                total_thinking_tokens += q_thinking_tokens

                # 4. Score answers using gold from evaluation_labels read ONLY AFTER execution
                correctness_flags: Dict[str, Any] = {}
                gold_row = eval_labels_gold.get(sid)
                if gold_row and extracted_ans is not None and extracted_ans != "ABSTAIN":
                    try:
                        if q_type == "yesno":
                            yn_eval = score_yesno([{"sample_id": sid, "extracted_answer": extracted_ans, "gold_exact_answer": gold_row.exact_answer}])
                            correctness_flags["correct"] = bool(yn_eval.strict_accuracy.value == 1.0)
                        elif q_type == "factoid":
                            f_eval = score_factoid([{"sample_id": sid, "extracted_answer": extracted_ans, "gold_exact_answer": gold_row.exact_answer}])
                            correctness_flags["strict"] = bool(f_eval.strict_accuracy.value == 1.0)
                            correctness_flags["lenient"] = bool(f_eval.lenient_accuracy.value == 1.0)
                            correctness_flags["mrr"] = float(f_eval.mrr.value)
                        elif q_type == "list":
                            l_eval = score_list([{"sample_id": sid, "extracted_answer": extracted_ans, "gold_exact_answer": gold_row.exact_answer}])
                            correctness_flags["precision"] = float(l_eval.mean_precision.value)
                            correctness_flags["recall"] = float(l_eval.mean_recall.value)
                            correctness_flags["f1"] = float(l_eval.mean_f1.value)
                    except Exception as sc_err:
                        correctness_flags["scoring_error"] = str(sc_err)

                rec = {
                    "sample_id": sid,
                    "question_type": q_type,
                    "workflow_status": wf_status,
                    "gate_rule": gate_rule,
                    "reason_codes": [r.value if hasattr(r, "value") else str(r) for r in (wf_res.abstention.reason_codes if wf_res.abstention else [])],
                    "claim_status_counts": dict(claim_counts),
                    "iteration_count": len(wf_res.iterations),
                    "extracted_answer": extracted_ans,
                    "correctness_flags": correctness_flags,
                    "error_code": extractor_err or wf_res.error_code,
                    "durations": {
                        "workflow_ms": wf_duration_ms,
                        "extractor_ms": extractor_ms,
                        "total_ms": wf_duration_ms + extractor_ms,
                    },
                    "llm_calls": q_calls,
                    "llm_tokens": {"input": q_in_tokens, "output": q_out_tokens, "thinking": q_thinking_tokens},
                }
                per_record_outputs.append(rec)
                cp_f.write(json.dumps(rec) + "\n")
                cp_f.flush()

                # Quota exhaustion or circuit breaker immediate halt
                if is_partial:
                    break

                if extractor_err == "LLM_QUOTA_EXHAUSTED":
                    is_partial = True
                    partial_reason = "Daily quota exhausted (LLM_QUOTA_EXHAUSTED). Stopping evaluation run."
                    break

                if effective_call_cap and total_calls_tracked >= effective_call_cap:
                    is_partial = True
                    partial_reason = f"Call cap reached: EVAL_MAX_LLM_CALLS={effective_call_cap}"
                    break

                if consecutive_auth_errors >= 3:
                    is_partial = True
                    partial_reason = "Circuit breaker triggered: 3 consecutive LLM_AUTH_ERROR exceptions."
                    break
                if consecutive_failures >= 5:
                    is_partial = True
                    partial_reason = "Circuit breaker triggered: 5 consecutive FAILED questions."
                    break

        completed_time = datetime.now(timezone.utc)

        if is_partial:
            labels.append("PARTIAL")
            status_val = "PARTIAL"
        else:
            status_val = "COMPLETED"

        temp_sent = getattr(self.settings, "LLM_TEMPERATURE", None) is not None
        th_level = getattr(self.settings, "LLM_THINKING_LEVEL", None)
        mdl_name = getattr(self.settings, "LLM_MODEL", "unknown")

        per_run_usage = {
            "total_llm_calls": total_calls_tracked,
            "total_input_tokens": total_in_tokens,
            "total_output_tokens": total_out_tokens,
            "total_thinking_tokens": total_thinking_tokens,
            "temperature_sent": temp_sent,
            "thinking_level": th_level,
            "model_name": mdl_name,
        }

        final_run_data = {
            "run_id": run_id,
            "dataset": "bioasq",
            "split": split_name,
            "set_name": set_name,
            "config_name": config_name,
            "status": status_val,
            "partial_reason": partial_reason if is_partial else None,
            "seed": 42,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "settings_snapshot": settings_snap,
            "models": models_dict,
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "corpus_version": corpus_version,
            "source_checksums": source_checksums,
            "config_hash": config_hash,
            "labels": labels,
            "record_counts": {
                "total_planned": len(questions),
                "total_completed": len(per_record_outputs),
                "ready": sum(1 for r in per_record_outputs if r["workflow_status"] == ReportStatusEnum.READY_FOR_REVIEW.value),
                "abstained": sum(1 for r in per_record_outputs if r["workflow_status"] == ReportStatusEnum.ABSTAINED.value),
                "failed": sum(1 for r in per_record_outputs if r["workflow_status"] == "FAILED"),
            },
            "error_counts": dict(error_counts),
            "total_llm_calls": total_calls_tracked,
            "total_input_tokens": total_in_tokens,
            "total_output_tokens": total_out_tokens,
            "total_thinking_tokens": total_thinking_tokens,
            "temperature_sent": temp_sent,
            "thinking_level": th_level,
            "model_name": mdl_name,
            "usage": per_run_usage,
            "per_run_usage": per_run_usage,
            "per_record_outputs": per_record_outputs,
        }

        # Write run JSON file
        out_json_path = os.path.join(self.output_dir, f"{run_id}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_run_data, f, indent=2)

        # Write rows to PostgreSQL evaluation_runs
        with SessionLocal() as session:
            db_run = EvaluationRun(
                id=str(uuid.uuid4()),
                dataset_name="bioasq",
                split_name=split_name,
                sample_size=len(questions),
                corpus_version=corpus_version,
                embedding_model=self.settings.EMBEDDING_MODEL,
                run_metadata=json.dumps({
                    "config_name": config_name,
                    "set_name": set_name,
                    "config_hash": config_hash,
                    "labels": labels,
                    "status": status_val,
                    "json_file": out_json_path,
                }),
                started_at=start_time,
                completed_at=completed_time,
                status=status_val,
            )
            session.add(db_run)
            session.commit()

        return final_run_data
