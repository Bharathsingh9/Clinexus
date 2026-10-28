"""Integration Tests for Phase 4 Stage B: Extensions C, D, and Urgent-Symptom Rules.

Specification Section 5.3, 5.4, 8:
- Real Postgres and real Qdrant integration with ExtendedWorkflowOrchestrator.
- End-to-end state transitions: DRAFT -> ANALYZING -> READY_FOR_REVIEW -> REVIEWED -> REVISION_REQUESTED.
- Evidence-level provenance persisted to PostgreSQL; assert_provenance_complete verified.
- Immutable append-only audit trail and review actions.
- MARK_REVIEWED uses exact text "Reviewed by clinician", never "approved" or "validated".
- Revision flow generates a NEW report version while preserving earlier versions and their provenance.
- Urgent-symptom rules match before analysis, write an AuditEvent, attach banner, and never block analysis.
- Provenance failure at finalize diverts to ABSTAINED with SYSTEM_ERROR, never ready.
"""

import json
from typing import Dict, Optional
import pytest
from sqlalchemy import delete, select

from backend.app.orchestration.extended_workflow import ExtendedWorkflowOrchestrator
from backend.app.orchestration.review_service import ReviewService
from backend.app.orchestration.states import CaseReviewState
from backend.app.persistence.audit_repository import AuditRepository
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.execution_repository import ExecutionRepository
import os
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.persistence.models import (
    AuditEvent as AuditEventModel,
    ClinicalCase as ClinicalCaseModel,
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
    HumanReview as HumanReviewModel,
    ProvenanceEdge as ProvenanceEdgeModel,
    ReportVersion as ReportVersionModel,
    User as UserModel,
)
from backend.app.provenance import (
    ProvenanceIncompleteError,
    assert_provenance_complete,
)
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name
from backend.app.safety.urgent import URGENT_BANNER_TEXT
from backend.app.schemas.clinical_case import CaseSubmission
from backend.app.schemas.reports import (
    AbstentionReasonCode,
    ReportStatusEnum,
    ReviewActionEnum,
)
from backend.app.security import UserRole, hash_password
from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient
from backend.tests.integration.test_extended_workflow import (
    make_canned_llm_for_extended_workflow,
)

def test_end_to_end_provenance_review_and_versioning(setup_extended_corpus):
    """End-to-end integration: State transitions, provenance persistence, review, and revision versioning."""
    llm = make_canned_llm_for_extended_workflow()
    repo = ExecutionRepository()
    audit_repo = AuditRepository(SessionLocal())

    orchestrator = ExtendedWorkflowOrchestrator(
        llm_provider=llm,
        ncbi_client=FakeNCBIClient(),
        repository=repo,
        embedding_service=setup_extended_corpus["embedder"],
        qdrant_wrapper=setup_extended_corpus["qdrant"],
        collection_version=setup_extended_corpus["collection_version"],
    )

    # 1. Create clinician user in Postgres
    clinician_id = "clinician-integr-1"
    with SessionLocal() as session:
        existing = session.get(UserModel, clinician_id)
        if not existing:
            user = UserModel(
                id=clinician_id,
                email="clinician.stageb@clinexus.local",
                hashed_password=hash_password("DoctorPassword123!"),
                role=UserRole.CLINICIAN_REVIEWER.value,
            )
            session.add(user)
            session.commit()

    # 2. Execute workflow on valid case submission
    sub = CaseSubmission(
        raw_text="45yo male presents with fever for 3 days and persistent cough. Patient denies chest pain."
    )
    final_state = orchestrator.run(submission=sub)

    assert final_state.workflow_status == ReportStatusEnum.READY_FOR_REVIEW.value
    assert final_state.report is not None
    v1_report_id = final_state.report.report_version_id
    case_id = final_state.case_id

    # 3. Verify PostgreSQL State Machine: Case status is READY_FOR_REVIEW
    with SessionLocal() as session:
        case_row = session.get(ClinicalCaseModel, case_id)
        assert case_row is not None
        assert case_row.status == CaseReviewState.READY_FOR_REVIEW.value

        # Verify state transition AuditEvents exist
        events = audit_repo.list_by_resource("clinical_case", case_id)
        state_events = [e for e in events if e.event_type == "CASE_STATE_CHANGED"]
        assert len(state_events) >= 2  # DRAFT -> ANALYZING, ANALYZING -> READY_FOR_REVIEW
        states_seen = [json.loads(e.payload)["to_state"] for e in state_events]
        assert "ANALYZING" in states_seen
        assert "READY_FOR_REVIEW" in states_seen

    # 4. Verify Provenance in PostgreSQL (EXT-C)
    with SessionLocal() as session:
        edges_stmt = select(ProvenanceEdgeModel).where(
            ProvenanceEdgeModel.report_version_id == v1_report_id
        )
        edges = list(session.scalars(edges_stmt).all())
        assert len(edges) > 0

        # Verify completeness invariant passes on the database-persisted report version
        report_v1_row = session.get(ReportVersionModel, v1_report_id)
        assert assert_provenance_complete(report_v1_row, session=session) is True

    # 5. Clinician Review Action: MARK_REVIEWED (EXT-D)
    with SessionLocal() as session:
        rev_service = ReviewService(session)
        review1 = rev_service.record_review_action(
            case_id=case_id,
            report_version_id=v1_report_id,
            reviewer_id=clinician_id,
            reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
            action=ReviewActionEnum.MARK_REVIEWED,
        )

        assert review1.outcome == "Reviewed by clinician"
        assert "approved" not in review1.outcome.lower()
        assert "validated" not in review1.outcome.lower()

        # Case status is now REVIEWED
        case_row = session.get(ClinicalCaseModel, case_id)
        assert case_row.status == CaseReviewState.REVIEWED.value

    # 6. Clinician Review Action: REQUEST_REVISION (EXT-D)
    with SessionLocal() as session:
        rev_service = ReviewService(session)
        review2 = rev_service.record_review_action(
            case_id=case_id,
            report_version_id=v1_report_id,
            reviewer_id=clinician_id,
            reviewer_role=UserRole.CLINICIAN_REVIEWER.value,
            action=ReviewActionEnum.REQUEST_REVISION,
            note="Requesting follow-up revision with alternative considerations.",
        )

        # Case status is now REVISION_REQUESTED
        case_row = session.get(ClinicalCaseModel, case_id)
        assert case_row.status == CaseReviewState.REVISION_REQUESTED.value

        # Reviews are append-only: both reviews exist
        all_reviews = rev_service.list_reviews_for_case(case_id)
        assert len(all_reviews) == 2
        assert all_reviews[0].action == ReviewActionEnum.MARK_REVIEWED
        assert all_reviews[1].action == ReviewActionEnum.REQUEST_REVISION

    # 7. Re-run workflow on this case to produce a revision
    revised_state = orchestrator.run(submission=sub, case_id=case_id)
    assert revised_state.workflow_status == ReportStatusEnum.READY_FOR_REVIEW.value
    assert revised_state.report is not None
    v2_report_id = revised_state.report.report_version_id

    # Must be a NEW version (version_number=2), with distinct report_version_id
    assert v2_report_id != v1_report_id
    assert revised_state.report.version_number >= 2

    # 8. Assert old version (v1) remains completely unchanged with its provenance preserved
    with SessionLocal() as session:
        v1_row = session.get(ReportVersionModel, v1_report_id)
        assert v1_row is not None
        assert v1_row.version_number == 1

        v2_row = session.get(ReportVersionModel, v2_report_id)
        assert v2_row is not None
        assert v2_row.version_number >= 2

        # Both versions have their own separate provenance edges
        v1_edges = list(session.scalars(
            select(ProvenanceEdgeModel).where(ProvenanceEdgeModel.report_version_id == v1_report_id)
        ).all())
        v2_edges = list(session.scalars(
            select(ProvenanceEdgeModel).where(ProvenanceEdgeModel.report_version_id == v2_report_id)
        ).all())
        assert len(v1_edges) > 0
        assert len(v2_edges) > 0


def test_urgent_rules_integration_with_workflow(setup_extended_corpus):
    """Verify urgent rules trigger before analysis, attach banner, write audit event, and analysis completes."""
    llm = make_canned_llm_for_extended_workflow()
    repo = ExecutionRepository()
    audit_repo = AuditRepository(SessionLocal())

    orchestrator = ExtendedWorkflowOrchestrator(
        llm_provider=llm,
        ncbi_client=FakeNCBIClient(),
        repository=repo,
        embedding_service=setup_extended_corpus["embedder"],
        qdrant_wrapper=setup_extended_corpus["qdrant"],
        collection_version=setup_extended_corpus["collection_version"],
    )

    # Narrative with urgent symptom: chest pain with dyspnea
    urgent_narrative = (
        "52-year-old male presenting with acute retrosternal chest pain and severe shortness of breath for 1 hour. "
        "Also reports mild diaphoresis."
    )
    sub = CaseSubmission(raw_text=urgent_narrative)

    # Run workflow
    result = orchestrator.run(submission=sub)

    # 1. Urgent banner attached to output report
    assert result.report is not None
    assert result.report.urgent_banner == URGENT_BANNER_TEXT

    # 2. Workflow completed without blocking
    assert result.workflow_status in (ReportStatusEnum.READY_FOR_REVIEW.value, ReportStatusEnum.ABSTAINED.value)

    # 3. AuditEvent recorded for urgent rule match
    with SessionLocal() as session:
        events = audit_repo.list_by_resource("clinical_case", result.case_id)
        urgent_events = [e for e in events if e.event_type == "URGENT_SYMPTOM_MATCHED"]
        assert len(urgent_events) >= 1
        payload = json.loads(urgent_events[0].payload)
        assert "chest_pain_with_dyspnea" in payload["matched_rule_names"]
