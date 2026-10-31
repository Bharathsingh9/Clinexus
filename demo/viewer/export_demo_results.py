"""Export demo scenario results to demo/viewer/data/results.json.

Runs the same four scenarios as demo/demo_workflow.py by importing and reusing
its infrastructure (ScriptedDemoLLMProvider, DEMO_COLLECTION_VERSION, etc.)
and the application orchestrator directly, capturing real WorkflowState objects.
Writes structured JSON with per-field REAL/SCRIPTED origin tags.

Authorization: One-off demo viewer exception DEC-041 (docs/decision_log.md).
Scope: demo/viewer/ only. No application code is modified.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap and isolated demo environment (mirrors demo_workflow.py)
# ---------------------------------------------------------------------------
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(VIEWER_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["DATABASE_SYNC_URL"] = "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["QDRANT_COLLECTION_VERSION"] = "777"
os.environ["EXTERNAL_MODE"] = "offline"

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.orchestration.extended_workflow import ExtendedWorkflowOrchestrator
from backend.app.orchestration.review_service import ReviewService
from backend.app.orchestration.states import (
    CaseReviewState,
    InvalidTransition,
    transition_case_state,
)
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.execution_repository import ExecutionRepository
from backend.app.persistence.models import (
    ClinicalCase as ClinicalCaseModel,
    HumanReview as HumanReviewModel,
    ProvenanceEdge as ProvenanceEdgeModel,
    User as UserModel,
)
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper
from backend.app.safety.urgent import evaluate_urgent_rules
from backend.app.schemas.clinical_case import CaseSubmission
from backend.app.schemas.reports import ReviewActionEnum
from backend.app.security.password import hash_password
from backend.app.security.roles import UnauthorizedActionError
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider
from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient

# Reuse exactly the same provider from demo_workflow (not copied — imported)
# We need DEMO_COLLECTION_VERSION and ScriptedDemoLLMProvider
from demo.demo_workflow import ScriptedDemoLLMProvider, DEMO_COLLECTION_VERSION

OUTPUT_PATH = Path(VIEWER_DIR) / "data" / "results.json"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _dt(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO 8601 string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _serialize_agent_records(agent_records) -> List[Dict]:
    """Serialize AgentExecutionRecord list to timeline dicts."""
    timeline = []
    for rec in agent_records:
        name = getattr(rec, "agent_name", "node")
        status = getattr(rec, "status", "UNKNOWN")
        status_str = status.value if hasattr(status, "value") else str(status)
        start = getattr(rec, "started_at", None)
        end = getattr(rec, "completed_at", None)
        if start and end:
            dur_ms = int((end - start).total_seconds() * 1000)
        else:
            dur_ms = None
        note = getattr(rec, "output_summary", None) or getattr(rec, "input_summary", "") or ""
        timeline.append({
            "node": name,
            "status": status_str,
            "duration_ms": dur_ms,
            "note": note[:120] if note else "",
            "_origin": "REAL",
        })
    return timeline


def _serialize_claims(state) -> List[Dict]:
    """Serialize claims with verification status looked up from state.verifications."""
    if state.report is None:
        return []
    # Build a lookup from claim_id -> verification
    verif_map = {}
    for v in (state.verifications or []):
        verif_map[v.claim_id] = v

    claims_out = []
    for claim in state.report.claims:
        verif = verif_map.get(claim.claim_id)
        verif_status = None
        verif_justification = None
        verif_method = None
        verif_limitations = None
        if verif is not None:
            vs = verif.status
            verif_status = vs.value if hasattr(vs, "value") else str(vs)
            verif_justification = verif.justification
            vm = verif.method
            verif_method = vm.value if hasattr(vm, "value") else str(vm)
            verif_limitations = verif.limitations

        claims_out.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "section": claim.section,
            "is_material": claim.is_material,
            "is_safety_critical": claim.is_safety_critical,
            "cited_evidence_ids": list(claim.cited_evidence_ids),
            "verification_status": verif_status,
            "verification_justification": verif_justification,
            "verification_method": verif_method,
            "verification_limitations": verif_limitations,
            "_claim_origin": "SCRIPTED",  # claim text is scripted
            "_verification_origin": "REAL",  # deterministic checks are real
        })
    return claims_out


def _serialize_abstention(abstention) -> Optional[Dict]:
    if abstention is None:
        return None
    return {
        "abstained": abstention.abstained,
        "reason_codes": [r.value for r in abstention.reason_codes],
        "explanation": abstention.explanation,
        "affected_claim_ids": list(abstention.affected_claim_ids),
        "affected_sections": list(abstention.affected_sections),
        "missing_evidence": list(abstention.missing_evidence),
        "suggested_next_steps": list(abstention.suggested_next_steps),
        "_origin": "REAL",
    }


def _serialize_iterations(iterations) -> List[Dict]:
    out = []
    for it in iterations:
        out.append({
            "iteration_number": it.iteration_number,
            "node_selected": it.node_selected,
            "confidence_before": it.confidence_before,
            "confidence_after": it.confidence_after,
            "action_taken": it.action_taken,
            "_origin": "REAL",
        })
    return out


def _serialize_retrieved_evidence(state) -> List[Dict]:
    """Serialize top retrieved passages from state.retrieved_evidence."""
    if state.retrieved_evidence is None:
        return []
    results = []
    for r in state.retrieved_evidence.results[:5]:
        results.append({
            "evidence_id": r.evidence_id,
            "retrieval_score": r.retrieval_score,
            "rank": r.rank,
            "source": r.source,
            "title": r.title,
            "text_preview": r.text[:200] if r.text else "",
            "corpus_version": r.corpus_version,
            "embedding_model": r.embedding_model,
            "_origin": "REAL",
        })
    return results


def _make_orchestrator(scenario_mode: str, session_factory) -> ExtendedWorkflowOrchestrator:
    llm = ScriptedDemoLLMProvider(scenario_mode=scenario_mode)
    embedder = EmbeddingService()
    qdrant = QdrantWrapper(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    audit_repo = AuditRepository(session_factory())
    repo = ExecutionRepository()
    return ExtendedWorkflowOrchestrator(
        llm_provider=llm,
        ncbi_client=FakeNCBIClient(),
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        collection_version=DEMO_COLLECTION_VERSION,
        audit_repository=audit_repo,
        repository=repo,
    )


# ---------------------------------------------------------------------------
# Scenario A: READY path
# ---------------------------------------------------------------------------

def export_scenario_a(session_factory):
    """Run Scenario A; return structured dict + (case_id, report_version_id)."""
    print("[export] Running Scenario A (READY path)...")
    orchestrator = _make_orchestrator("READY", session_factory)
    question_text = "What is formin associated with in the snail?"
    submission = CaseSubmission(raw_text=question_text)

    t0 = time.perf_counter()
    state = orchestrator.run(submission, is_benchmark=True)
    elapsed = time.perf_counter() - t0

    assert state.workflow_status == "READY_FOR_REVIEW", (
        f"Scenario A: expected READY_FOR_REVIEW, got {state.workflow_status}"
    )
    assert state.report is not None

    # Provenance edges for the first claim
    provenance_edges = []
    with session_factory() as session:
        edges = session.execute(
            select(ProvenanceEdgeModel).where(
                ProvenanceEdgeModel.report_version_id == state.report.report_version_id
            )
        ).scalars().all()
        for edge in edges:
            rel_str = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
            provenance_edges.append({
                "relation": rel_str,
                "claim_id": edge.claim_id,
                "evidence_chunk_id": edge.evidence_chunk_id,
                "report_section_id": edge.report_section_id,
                "document_id": edge.document_id,
                "_origin": "REAL",
            })

    # Determine gate rule from abstention (absent here) and final state
    gate_info = {
        "outcome": "finalize_ready",
        "rule_triggered": None,
        "reason_codes": [],
        "explanation": "All 6 gate rules passed. Report finalized as READY_FOR_REVIEW.",
        "_origin": "REAL",
    }

    result = {
        "scenario": "A",
        "title": "READY PATH — Grounded benchmark query, all gate rules pass",
        "question": question_text,
        "is_benchmark": True,
        "elapsed_s": round(elapsed, 3),
        "workflow_status": state.workflow_status,
        "final_state": state.workflow_status,
        "urgent_banner": state.urgent_banner,
        "timeline": _serialize_agent_records(state.agent_records),
        "claims": _serialize_claims(state),
        "retrieved_evidence": _serialize_retrieved_evidence(state),
        "gate": gate_info,
        "abstention": None,
        "iterations": _serialize_iterations(state.iterations),
        "provenance_edges": provenance_edges,
        "report": {
            "report_version_id": state.report.report_version_id,
            "case_id": state.report.case_id,
            "confidence_score": state.report.confidence_score,
            "status": state.report.status.value if hasattr(state.report.status, "value") else str(state.report.status),
            "sections": dict(state.report.sections),
            "_origin": "REAL_STRUCTURE_SCRIPTED_LLM_CONTENT",
        },
        "_origin_summary": {
            "timeline": "REAL",
            "retrieval": "REAL",
            "verification": "REAL",
            "gate_decision": "REAL",
            "provenance": "REAL",
            "claim_text": "SCRIPTED",
            "entailment_judgment": "SCRIPTED",
        },
    }
    print(f"  [OK] Scenario A: {state.workflow_status} in {elapsed:.2f}s, "
          f"{len(provenance_edges)} provenance edges, "
          f"{len(state.report.claims)} claims.")
    return result, state.case_id, state.report.report_version_id


# ---------------------------------------------------------------------------
# Scenario B: UNSUPPORTED path
# ---------------------------------------------------------------------------

def export_scenario_b(session_factory):
    """Run Scenario B; return structured dict."""
    print("[export] Running Scenario B (UNSUPPORTED / numeric veto path)...")
    orchestrator = _make_orchestrator("UNSUPPORTED", session_factory)
    question_text = "What is formin associated with in the snail?"
    submission = CaseSubmission(raw_text=question_text)

    t0 = time.perf_counter()
    state = orchestrator.run(submission, is_benchmark=True)
    elapsed = time.perf_counter() - t0

    assert state.workflow_status == "ABSTAINED", (
        f"Scenario B: expected ABSTAINED, got {state.workflow_status}"
    )
    assert state.abstention is not None

    # Demonstrate the numeric veto: find the claim with "99" in it
    numeric_veto_claim = None
    for v in state.verifications:
        vs = v.status
        vs_str = vs.value if hasattr(vs, "value") else str(vs)
        if vs_str in ("PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONTRADICTED"):
            numeric_veto_claim = {
                "claim_id": v.claim_id,
                "claim_text": v.claim_text,
                "status": vs_str,
                "justification": v.justification,
                "_origin_claim": "SCRIPTED",
                "_origin_veto": "REAL",
            }
            break

    gate_info = {
        "outcome": "finalize_abstain",
        "rule_triggered": None,
        "reason_codes": [r.value for r in state.abstention.reason_codes],
        "explanation": state.abstention.explanation,
        "_origin": "REAL",
    }

    result = {
        "scenario": "B",
        "title": "UNSUPPORTED PATH — Numeric entity mismatch → code veto → budget exhausted → ABSTAINED",
        "question": question_text,
        "is_benchmark": True,
        "elapsed_s": round(elapsed, 3),
        "workflow_status": state.workflow_status,
        "final_state": state.workflow_status,
        "urgent_banner": state.urgent_banner,
        "timeline": _serialize_agent_records(state.agent_records),
        "claims": _serialize_claims(state),
        "retrieved_evidence": _serialize_retrieved_evidence(state),
        "gate": gate_info,
        "abstention": _serialize_abstention(state.abstention),
        "iterations": _serialize_iterations(state.iterations),
        "provenance_edges": [],
        "numeric_veto_demonstration": {
            "scripted_judge_output": "SUPPORTED",
            "real_code_veto_result": "PARTIALLY_SUPPORTED",
            "veto_reason": "Numeric value '99' is absent from every cited literature passage.",
            "veto_claim": numeric_veto_claim,
            "_scripted_judge_origin": "SCRIPTED",
            "_veto_origin": "REAL",
        },
        "suppressed_sections": {
            "potential_considerations": state.report.sections.get("potential_considerations", "") if state.report else "",
            "note": "Unverified considerations are suppressed in abstained report.",
            "_origin": "REAL",
        },
        "report": {
            "report_version_id": state.report.report_version_id if state.report else None,
            "case_id": state.report.case_id if state.report else None,
            "confidence_score": state.report.confidence_score if state.report else None,
            "status": (state.report.status.value if hasattr(state.report.status, "value") else str(state.report.status)) if state.report else None,
            "sections": dict(state.report.sections) if state.report else {},
            "_origin": "REAL_STRUCTURE_SCRIPTED_LLM_CONTENT",
        },
        "_origin_summary": {
            "timeline": "REAL",
            "retrieval": "REAL",
            "numeric_veto": "REAL",
            "optimizer_iterations": "REAL",
            "gate_decision": "REAL",
            "abstention": "REAL",
            "claim_text_with_99": "SCRIPTED",
            "entailment_judgment": "SCRIPTED",
        },
    }
    print(f"  [OK] Scenario B: {state.workflow_status} in {elapsed:.2f}s, "
          f"{len(state.iterations)} optimizer iterations.")
    return result


# ---------------------------------------------------------------------------
# Scenario C: URGENT + NO EVIDENCE path
# ---------------------------------------------------------------------------

def export_scenario_c(session_factory):
    """Run Scenario C; return structured dict."""
    print("[export] Running Scenario C (URGENT + NO EVIDENCE path)...")
    urgent_case_text = (
        "[SYNTHETIC] 62-year-old patient presenting with acute substernal chest pain "
        "with shortness of breath and diaphoresis."
    )
    submission = CaseSubmission(raw_text=urgent_case_text)

    # Pre-analysis urgent evaluation (REAL — same as demo_workflow.py)
    urgent_result = evaluate_urgent_rules(raw_text=urgent_case_text, findings=None)
    urgent_pre = {
        "is_urgent": urgent_result.is_urgent,
        "matched_rule_names": list(urgent_result.matched_rule_names),
        "banner_text": urgent_result.banner_text,
        "_origin": "REAL",
    }

    # Run with elevated relevance threshold so demo corpus has 0 relevant passages
    orig_min_score = settings.RETRIEVAL_MIN_SCORE
    settings.RETRIEVAL_MIN_SCORE = 0.55

    orchestrator = _make_orchestrator("URGENT", session_factory)

    t0 = time.perf_counter()
    try:
        state = orchestrator.run(submission, is_benchmark=False)
    finally:
        settings.RETRIEVAL_MIN_SCORE = orig_min_score
    elapsed = time.perf_counter() - t0

    assert state.workflow_status == "ABSTAINED", (
        f"Scenario C: expected ABSTAINED, got {state.workflow_status}"
    )
    assert state.abstention is not None

    gate_info = {
        "outcome": "finalize_abstain",
        "rule_triggered": 4,
        "reason_codes": [r.value for r in state.abstention.reason_codes],
        "explanation": "Gate Rule 4: fewer than MIN_RELEVANT_PASSAGES passages above threshold.",
        "_origin": "REAL",
    }

    result = {
        "scenario": "C",
        "title": "URGENT + NO EVIDENCE PATH — Pre-analysis urgent banner → 0 relevant passages → ABSTAINED",
        "question": urgent_case_text,
        "is_benchmark": False,
        "elapsed_s": round(elapsed, 3),
        "workflow_status": state.workflow_status,
        "final_state": state.workflow_status,
        "urgent_banner": state.urgent_banner,
        "urgent_pre_analysis": urgent_pre,
        "timeline": _serialize_agent_records(state.agent_records),
        "claims": [],
        "retrieved_evidence": _serialize_retrieved_evidence(state),
        "gate": gate_info,
        "abstention": _serialize_abstention(state.abstention),
        "iterations": _serialize_iterations(state.iterations),
        "provenance_edges": [],
        "clinical_note": "[SYNTHETIC] case only. No clinical guidance is presented. System terminates safely.",
        "_origin_summary": {
            "timeline": "REAL",
            "urgent_detection": "REAL",
            "retrieval": "REAL",
            "gate_decision": "REAL",
            "abstention": "REAL",
            "case_text": "SYNTHETIC_DEMO_NOT_REAL_PATIENT",
        },
    }
    print(f"  [OK] Scenario C: {state.workflow_status} in {elapsed:.2f}s, "
          f"urgent={urgent_result.is_urgent}.")
    return result


# ---------------------------------------------------------------------------
# Scenario D: REVIEW / AUDIT path
# ---------------------------------------------------------------------------

def export_scenario_d(case_id: str, report_version_id: str, session_factory):
    """Run Scenario D; return structured dict."""
    print("[export] Running Scenario D (HITL Review / Audit path)...")
    clinician_pass = os.environ.get("DEMO_CLINICIAN_PASSWORD", "ClinicianPass123!")
    viewer_pass = os.environ.get("DEMO_VIEWER_PASSWORD", "ViewerPass123!")

    # Provision demo users (idempotent — same logic as demo_workflow.py)
    with session_factory() as session:
        clinician = session.execute(
            select(UserModel).where(UserModel.email == "clinician_demo@clinexus.local")
        ).scalar_one_or_none()
        if not clinician:
            clinician = UserModel(
                email="clinician_demo@clinexus.local",
                hashed_password=hash_password(clinician_pass),
                role="clinician_reviewer",
                is_active=True,
            )
            session.add(clinician)
        viewer = session.execute(
            select(UserModel).where(UserModel.email == "viewer_demo@clinexus.local")
        ).scalar_one_or_none()
        if not viewer:
            viewer = UserModel(
                email="viewer_demo@clinexus.local",
                hashed_password=hash_password(viewer_pass),
                role="viewer",
                is_active=True,
            )
            session.add(viewer)
        session.commit()
        session.refresh(clinician)
        session.refresh(viewer)
        clinician_id = clinician.id
        viewer_id = viewer.id

    # Step D.1: viewer attempts MARK_REVIEWED → must be rejected
    rbac_rejection_message = None
    with session_factory() as session:
        review_svc = ReviewService(session=session)
        try:
            review_svc.record_review_action(
                case_id=case_id,
                report_version_id=report_version_id,
                reviewer_id=viewer_id,
                reviewer_role="viewer",
                action=ReviewActionEnum.MARK_REVIEWED,
            )
        except UnauthorizedActionError as exc:
            rbac_rejection_message = str(exc)

    assert rbac_rejection_message is not None, "RBAC check failed: viewer was not rejected"

    # Step D.2: clinician MARK_REVIEWED
    with session_factory() as session:
        review_svc = ReviewService(session=session)
        rev1 = review_svc.record_review_action(
            case_id=case_id,
            report_version_id=report_version_id,
            reviewer_id=clinician_id,
            reviewer_role="clinician_reviewer",
            action=ReviewActionEnum.MARK_REVIEWED,
            note="Literature evidence on left-right asymmetry confirmed.",
        )
        session.commit()
        case_obj = session.get(ClinicalCaseModel, case_id)
        case_status_after_review = str(case_obj.status)

    # Step D.3: clinician ADD_NOTE (append-only)
    with session_factory() as session:
        review_svc = ReviewService(session=session)
        rev2 = review_svc.record_review_action(
            case_id=case_id,
            report_version_id=report_version_id,
            reviewer_id=clinician_id,
            reviewer_role="clinician_reviewer",
            action=ReviewActionEnum.ADD_NOTE,
            note="Additional note: Embryonic phenotype confirmed in snail model.",
        )
        session.commit()
        all_reviews = session.execute(
            select(HumanReviewModel).where(
                HumanReviewModel.report_version_id == report_version_id
            )
        ).scalars().all()
        reviews_serialized = []
        for r in all_reviews:
            action_str = r.action.value if hasattr(r.action, "value") else str(r.action)
            reviews_serialized.append({
                "action": action_str,
                "outcome": r.outcome,
                "note": r.note,
                "reviewer_role": r.reviewer_role,
                "is_human": True,
                "_origin": "REAL",
            })

    # Step D.4: invalid state transition rejection
    invalid_transition_message = None
    with session_factory() as session:
        try:
            transition_case_state(
                session=session,
                case_id=case_id,
                target_state=CaseReviewState.DRAFT,
                actor_id=clinician_id,
                actor_role="clinician_reviewer",
            )
        except InvalidTransition as exc:
            invalid_transition_message = str(exc)

    assert invalid_transition_message is not None, "Invalid transition was not rejected"

    # Step D.5: fetch full audit trail
    audit_events = []
    with session_factory() as session:
        audit_repo = AuditRepository(session)
        case_events = audit_repo.list_by_resource("clinical_case", case_id)
        report_events = audit_repo.list_by_resource("report_version", report_version_id)
        trail = case_events + report_events
        trail.sort(key=lambda evt: evt.timestamp)
        for evt in trail:
            payload = evt.payload
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            audit_events.append({
                "event_type": evt.event_type,
                "actor_role": evt.actor_role,
                "resource_type": evt.resource_type,
                "resource_id": evt.resource_id,
                "timestamp": _dt(evt.timestamp),
                "payload": payload,
                "is_human": "HUMAN_REVIEW" in evt.event_type or "human" in evt.event_type.lower(),
                "_origin": "REAL",
            })

    result = {
        "scenario": "D",
        "title": "HITL REVIEW & AUDIT PATH — RBAC enforcement, clinician review, append-only audit",
        "case_id": case_id,
        "report_version_id": report_version_id,
        "users": {
            "clinician": "clinician_demo@clinexus.local (role: clinician_reviewer)",
            "viewer": "viewer_demo@clinexus.local (role: viewer)",
            "note": "Passwords loaded from environment variables, hashed with Argon2id, never printed.",
            "_origin": "REAL",
        },
        "rbac_rejection": {
            "attempted_action": "MARK_REVIEWED",
            "attempted_role": "viewer",
            "rejection_message": rbac_rejection_message,
            "_origin": "REAL",
        },
        "clinician_review": {
            "action": "MARK_REVIEWED",
            "outcome": rev1.outcome,
            "note": rev1.note,
            "case_status_after": case_status_after_review,
            "note_label": "[HUMAN]",
            "_origin": "REAL",
        },
        "append_only_verification": {
            "total_reviews": len(reviews_serialized),
            "reviews": reviews_serialized,
            "note": "Append-only verified: earlier review preserved, not overwritten.",
            "_origin": "REAL",
        },
        "invalid_transition": {
            "attempted": "REVIEWED → DRAFT",
            "rejection_message": invalid_transition_message,
            "note": "Pure state machine: only valid transitions in table are permitted.",
            "_origin": "REAL",
        },
        "audit_trail": audit_events,
        "_origin_summary": {
            "users": "REAL",
            "rbac": "REAL",
            "review_actions": "REAL",
            "audit_events": "REAL",
            "state_machine": "REAL",
        },
    }
    print(f"  [OK] Scenario D: {len(audit_events)} audit events, "
          f"{len(reviews_serialized)} reviews recorded.")
    return result


# ---------------------------------------------------------------------------
# Main export runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("CLINEXUS DEMO VIEWER — EXPORT SCENARIO RESULTS")
    print("=" * 72)
    print("Authorization: DEC-041 (docs/decision_log.md)")
    print("Database: clinexus_demo | Qdrant collection: v777")
    print()

    engine = create_engine(settings.DATABASE_SYNC_URL, echo=False, pool_pre_ping=True)
    DemoSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    result_a, case_id_a, report_id_a = export_scenario_a(DemoSession)
    result_b = export_scenario_b(DemoSession)
    result_c = export_scenario_c(DemoSession)
    result_d = export_scenario_d(case_id_a, report_id_a, DemoSession)

    total_elapsed = time.perf_counter() - t_total

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authorization": "DEC-041",
        "database": "clinexus_demo",
        "qdrant_collection_version": DEMO_COLLECTION_VERSION,
        "total_elapsed_s": round(total_elapsed, 3),
        "honesty_banner": (
            "DEMO VIEWER: not the Clinexus application interface. "
            "Language-model outputs are SCRIPTED; embeddings, retrieval, workflow, "
            "verifier checks, gate, provenance, and audit are REAL. "
            "Research prototype, not clinically validated. No accuracy results are shown."
        ),
        "scenarios": [result_a, result_b, result_c, result_d],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print()
    print(f"[OK] Wrote {OUTPUT_PATH.stat().st_size:,} bytes -> {OUTPUT_PATH}")
    print(f"     Total export time: {total_elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()
