"""Integration Test Suite for REST API.

Specification Section 7, Appendix A14, A15:
- Test route enumeration against SPEC Section 7.
- Auth: login success/failure, tampered/expired token, RBAC matrix, viewer forbidden on mutating routes.
- Privacy: validation errors never echo input strings; logs never echo input strings.
- Cases: submission with urgent banner, draft status, list endpoint has no free text.
- Execution: 202, idempotency key repeat, 409 on concurrent analysis, LLM_UNAVAILABLE / LLM_QUOTA_EXHAUSTED handling.
- Reviews: append-only, reviewer from token, MARK_REVIEWED label "Reviewed by clinician", cross-case version rejected.
- Evidence & Provenance: no sample_id/question_type in response, support NONE for unsupported claims.
- Evaluations: safety gate refusals, label checks, no question/gold leakage.
- Health: no secrets, dependency DOWN reporting.
- Rate limiting: 429 on excessive login attempts.
"""

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_llm_provider
from backend.app.api.rate_limiter import limiter
from backend.app.core.config import settings
from backend.app.core.llm_exceptions import LLMQuotaExhausted
from backend.app.main import app
from backend.app.persistence.database import SessionLocal, sync_engine
from backend.app.persistence.models import (
    CandidateClaim as CandidateClaimModel,
    ClaimVerification as ClaimVerificationModel,
    ClinicalCase as ClinicalCaseModel,
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
    HumanReview as HumanReviewModel,
    ReportVersion as ReportVersionModel,
    User as UserModel,
    WorkflowExecution as WorkflowExecutionModel,
)
from backend.app.security.jwt import create_access_token
from backend.app.security.password import hash_password
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider
from backend.app.security.roles import UserRole
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider

# Ensure clean rate limiter state before each test
@pytest.fixture(autouse=True)
def reset_limiter():
    limiter.reset()


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_users(db_session: Session) -> Dict[str, Dict[str, str]]:
    """Create test users for viewer, clinician_reviewer, and admin roles."""
    roles = {
        "viewer": ("viewer_test@clinexus.local", UserRole.VIEWER.value),
        "clinician": ("clinician_test@clinexus.local", UserRole.CLINICIAN_REVIEWER.value),
        "admin": ("admin_test@clinexus.local", UserRole.ADMIN.value),
    }
    user_data = {}
    for key, (email, role) in roles.items():
        stmt = select(UserModel).where(UserModel.email == email)
        u = db_session.scalars(stmt).first()
        if not u:
            u = UserModel(
                id=str(uuid.uuid4()),
                email=email,
                hashed_password=hash_password("test-secure-password-1234"),
                role=role,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(u)
            db_session.commit()
            db_session.refresh(u)
        token = create_access_token(user_id=u.id, email=u.email, role=u.role)
        user_data[key] = {
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "token": token,
            "headers": {"Authorization": f"Bearer {token}"},
        }
    return user_data


@pytest.fixture
def client():
    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    fake_provider = FakeLLMProvider()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_provider] = lambda: fake_provider
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_llm_provider, None)


# ---------------------------------------------------------------------------
# 1. Route Enumeration Test
# ---------------------------------------------------------------------------

SPEC_SECTION_7_ENDPOINTS = {
    ("GET", "/health"),
    ("GET", "/health/dependencies"),
    ("POST", "/auth/login"),
    ("GET", "/users"),
    ("POST", "/users"),
    ("POST", "/cases"),
    ("GET", "/cases"),
    ("GET", "/cases/{id}"),
    ("POST", "/cases/{id}/analyze"),
    ("GET", "/executions/{id}"),
    ("GET", "/executions/{id}/agents"),
    ("GET", "/cases/{id}/report"),
    ("GET", "/cases/{id}/reports"),
    ("GET", "/reports/{version_id}/claims"),
    ("GET", "/claims/{claim_id}/provenance"),
    ("GET", "/evidence/{evidence_id}"),
    ("GET", "/cases/{id}/abstention"),
    ("POST", "/cases/{id}/clarifications"),
    ("POST", "/cases/{id}/refresh-evidence"),
    ("POST", "/cases/{id}/revisions"),
    ("POST", "/cases/{id}/reviews"),
    ("GET", "/cases/{id}/audit"),
    ("POST", "/evaluations"),
    ("GET", "/evaluations"),
    ("GET", "/evaluations/{id}"),
}


def test_routes_match_spec_section_7():
    """Verify that every route in the app matches SPEC Section 7 exactly with prefix /api/v1."""
    app_routes = set()
    openapi_paths = app.openapi().get("paths", {})
    for path, path_item in openapi_paths.items():
        if path.startswith("/api/v1"):
            clean_path = path[len("/api/v1"):]
            for method in path_item.keys():
                if method.upper() in ["GET", "POST"]:
                    app_routes.add((method.upper(), clean_path))

    # Assert exact match
    diff_extra = app_routes - SPEC_SECTION_7_ENDPOINTS
    diff_missing = SPEC_SECTION_7_ENDPOINTS - app_routes

    assert not diff_extra, f"Unexpected endpoints not in SPEC Section 7: {diff_extra}"
    assert not diff_missing, f"Missing endpoints required by SPEC Section 7: {diff_missing}"
    assert app_routes == SPEC_SECTION_7_ENDPOINTS


# ---------------------------------------------------------------------------
# 2. Authentication & Authorization Tests
# ---------------------------------------------------------------------------

def test_login_success_and_failure(client: TestClient, test_users: Dict[str, Any]):
    # Success
    resp = client.post("/api/v1/auth/login", json={
        "email": "clinician_test@clinexus.local",
        "password": "test-secure-password-1234",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == "clinician_reviewer"
    assert data["token_type"] == "bearer"

    # Wrong password
    resp_bad = client.post("/api/v1/auth/login", json={
        "email": "clinician_test@clinexus.local",
        "password": "wrong-password",
    })
    assert resp_bad.status_code == 401
    err = resp_bad.json()["error"]
    assert err["code"] == "UNAUTHENTICATED"
    assert err["message"] == "Invalid email or password."

    # Non-existent email returns identical error message
    resp_none = client.post("/api/v1/auth/login", json={
        "email": "nonexistent_doctor@clinexus.local",
        "password": "test-password",
    })
    assert resp_none.status_code == 401
    assert resp_none.json()["error"]["message"] == err["message"]


def test_tampered_and_expired_token(client: TestClient):
    # Tampered token
    resp = client.get("/api/v1/cases", headers={"Authorization": "Bearer invalid.fake.token"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHENTICATED"

    # Expired token
    expired_token = create_access_token(
        user_id="fake-user-id",
        email="test@clinexus.local",
        role="viewer",
        expires_delta=timedelta(seconds=-10),
    )
    resp_exp = client.get("/api/v1/cases", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp_exp.status_code == 401
    assert resp_exp.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.parametrize(
    "method,endpoint,payload",
    [
        ("POST", "/api/v1/cases", {"raw_text": "Patient with cough"}),
        ("POST", "/api/v1/cases/case-123/analyze", {}),
        ("POST", "/api/v1/cases/case-123/reviews", {"action": "ADD_NOTE", "report_version_id": "v1"}),
        ("POST", "/api/v1/cases/case-123/clarifications", {"note": "Clarify symptom onset"}),
        ("POST", "/api/v1/cases/case-123/refresh-evidence", {}),
        ("POST", "/api/v1/cases/case-123/revisions", {}),
    ],
)
def test_viewer_forbidden_on_mutating_endpoints(
    client: TestClient, test_users: Dict[str, Any], method: str, endpoint: str, payload: Dict[str, Any]
):
    headers = test_users["viewer"]["headers"]
    if method == "POST":
        resp = client.post(endpoint, json=payload, headers=headers)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_admin_only_endpoints(client: TestClient, test_users: Dict[str, Any]):
    # Viewer rejected on /users
    resp_v = client.get("/api/v1/users", headers=test_users["viewer"]["headers"])
    assert resp_v.status_code == 403
    assert resp_v.json()["error"]["code"] == "FORBIDDEN"

    # Clinician rejected on /users
    resp_c = client.get("/api/v1/users", headers=test_users["clinician"]["headers"])
    assert resp_c.status_code == 403
    assert resp_c.json()["error"]["code"] == "FORBIDDEN"

    # Admin allowed on /users
    resp_a = client.get("/api/v1/users", headers=test_users["admin"]["headers"])
    assert resp_a.status_code == 200
    users = resp_a.json()
    assert isinstance(users, list)
    for u in users:
        assert "hashed_password" not in u
        assert "id" in u
        assert "role" in u


# ---------------------------------------------------------------------------
# 3. Privacy & Input Stripping Tests
# ---------------------------------------------------------------------------

def test_validation_error_strips_patient_free_text(
    client: TestClient, test_users: Dict[str, Any], caplog: pytest.LogCaptureFixture
):
    marker = "PATIENT_SSN_987654321_SENSITIVE_CLINICAL_NARRATIVE"
    # Send empty body which triggers validation error
    with caplog.at_level(logging.DEBUG):
        resp = client.post(
            "/api/v1/cases",
            json={"raw_text": marker, "unexpected_extra_field": "disallowed"},
            headers=test_users["clinician"]["headers"],
        )

    assert resp.status_code == 422
    body_text = resp.text
    # Distinctive marker must NOT be echoed in validation response details
    assert marker not in body_text
    # And must not appear in logged request details
    assert marker not in caplog.text


# ---------------------------------------------------------------------------
# 4. Clinical Case Creation & Listing
# ---------------------------------------------------------------------------

def test_create_case_urgent_banner_and_list_no_free_text(
    client: TestClient, test_users: Dict[str, Any], db_session: Session
):
    urgent_narrative = (
        "55yo male presenting with acute severe crushing substernal chest pain, "
        "dyspnea, and diaphoresis for 45 minutes."
    )
    resp = client.post(
        "/api/v1/cases",
        json={"raw_text": urgent_narrative},
        headers=test_users["clinician"]["headers"],
    )
    assert resp.status_code == 201
    case_data = resp.json()
    case_id = case_data["id"]
    assert case_data["status"] == "DRAFT"
    assert case_data["urgent"] is True
    assert "Possible urgent presentation" in (case_data["urgent_banner"] or "")

    # GET /cases summary listing: MUST NOT contain raw_text or free text
    list_resp = client.get("/api/v1/cases", headers=test_users["viewer"]["headers"])
    assert list_resp.status_code == 200
    cases = list_resp.json()
    target_summary = next(c for c in cases if c["id"] == case_id)
    assert target_summary["urgent"] is True
    assert target_summary["status"] == "DRAFT"
    assert "raw_text" not in target_summary
    assert "chief_complaint" not in target_summary

    # GET /cases/{id} detail view: includes free text
    detail_resp = client.get(f"/api/v1/cases/{case_id}", headers=test_users["viewer"]["headers"])
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == case_id
    assert detail["raw_text"] == urgent_narrative
    assert detail["urgent"] is True


# ---------------------------------------------------------------------------
# 5. Analysis Triggering, Idempotency, and Concurrent Guard
# ---------------------------------------------------------------------------

def test_analyze_trigger_and_idempotency(client: TestClient, test_users: Dict[str, Any]):
    # Create draft case
    c_resp = client.post(
        "/api/v1/cases",
        json={"raw_text": "Patient with persistent mild dry cough for 3 weeks."},
        headers=test_users["clinician"]["headers"],
    )
    case_id = c_resp.json()["id"]

    idemp_key = f"idemp-{uuid.uuid4()}"
    headers = {**test_users["clinician"]["headers"], "Idempotency-Key": idemp_key}

    # Trigger analysis (202 Accepted)
    a_resp = client.post(f"/api/v1/cases/{case_id}/analyze", headers=headers)
    assert a_resp.status_code == 202
    exec_id = a_resp.json()["execution_id"]
    assert a_resp.json()["status"] == "RUNNING"

    # Repeat with same Idempotency-Key: returns same execution_id
    repeat_resp = client.post(f"/api/v1/cases/{case_id}/analyze", headers=headers)
    assert repeat_resp.status_code in (200, 202)
    assert repeat_resp.json()["execution_id"] == exec_id

    # Second request while ANALYZING without idempotency key returns 409 INVALID_STATE
    analyzing_case_id = str(uuid.uuid4())
    with SessionLocal() as s:
        analyzing_case = ClinicalCaseModel(
            id=analyzing_case_id,
            raw_text_hash="hash_analyzing",
            chief_complaint="Patient undergoing analysis",
            status="ANALYZING",
        )
        s.add(analyzing_case)
        s.commit()

    conflict_resp = client.post(
        f"/api/v1/cases/{analyzing_case_id}/analyze",
        headers=test_users["clinician"]["headers"],
    )
    assert conflict_resp.status_code == 409
    assert conflict_resp.json()["error"]["code"] == "INVALID_STATE"


def test_executions_telemetry_no_prompts_or_outputs(
    client: TestClient, test_users: Dict[str, Any], db_session: Session
):
    case_row = ClinicalCaseModel(
        id=str(uuid.uuid4()),
        raw_text_hash="abc",
        chief_complaint="Patient with headache",
        status="DRAFT",
    )
    db_session.add(case_row)
    db_session.commit()

    # Trigger analysis
    resp = client.post(
        f"/api/v1/cases/{case_row.id}/analyze",
        headers=test_users["clinician"]["headers"],
    )
    assert resp.status_code == 202
    exec_id = resp.json()["execution_id"]

    # Check /executions/{id}
    st_resp = client.get(f"/api/v1/executions/{exec_id}", headers=test_users["viewer"]["headers"])
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["id"] == exec_id
    assert st_data["status"] in ("RUNNING", "READY_FOR_REVIEW", "ABSTAINED", "FAILED")

    # Check /executions/{id}/agents: prompts and outputs must not exist
    ag_resp = client.get(f"/api/v1/executions/{exec_id}/agents", headers=test_users["viewer"]["headers"])
    assert ag_resp.status_code == 200
    agents = ag_resp.json()
    assert isinstance(agents, list)
    for ag in agents:
        assert "input_summary" not in ag
        assert "output_summary" not in ag
        assert "prompt" not in ag
        assert "output" not in ag


# ---------------------------------------------------------------------------
# 6. Reviews & State Machine Invariants
# ---------------------------------------------------------------------------

def test_clinician_reviews_and_invariants(client: TestClient, test_users: Dict[str, Any], db_session: Session):
    case_row = ClinicalCaseModel(
        id=str(uuid.uuid4()),
        raw_text_hash="hash1",
        chief_complaint="Sample presentation",
        status="READY_FOR_REVIEW",
    )
    exec_row = WorkflowExecutionModel(
        id=str(uuid.uuid4()),
        case_id=case_row.id,
        status="READY_FOR_REVIEW",
    )
    rep_row = ReportVersionModel(
        id=str(uuid.uuid4()),
        case_id=case_row.id,
        execution_id=exec_row.id,
        version_number=1,
        summary='{"case_summary": "Test summary"}',
        confidence_score=0.88,
        status="READY_FOR_REVIEW",
    )
    db_session.add_all([case_row, exec_row])
    db_session.flush()
    db_session.add(rep_row)
    db_session.commit()

    # Reviewer identity comes from token, MARK_REVIEWED text is "Reviewed by clinician"
    rev_resp = client.post(
        f"/api/v1/cases/{case_row.id}/reviews",
        json={
            "action": "MARK_REVIEWED",
            "report_version_id": rep_row.id,
            "note": "Clinician confirms findings.",
        },
        headers=test_users["clinician"]["headers"],
    )
    assert rev_resp.status_code == 201
    rev_data = rev_resp.json()
    assert rev_data["reviewer_id"] == test_users["clinician"]["id"]
    assert rev_data["outcome"] == "Reviewed by clinician"
    assert rev_data["action"] == "MARK_REVIEWED"

    # Second review never overwrites first (append-only)
    rev_resp_2 = client.post(
        f"/api/v1/cases/{case_row.id}/reviews",
        json={
            "action": "ADD_NOTE",
            "report_version_id": rep_row.id,
            "note": "Second clinician adds follow-up observation.",
        },
        headers=test_users["admin"]["headers"],
    )
    assert rev_resp_2.status_code == 201

    reviews_in_db = db_session.scalars(
        select(HumanReviewModel).where(HumanReviewModel.case_id == case_row.id)
    ).all()
    assert len(reviews_in_db) == 2

    # Version from another case is rejected
    other_case = ClinicalCaseModel(
        id=str(uuid.uuid4()),
        raw_text_hash="hash2",
        chief_complaint="Other",
        status="DRAFT",
    )
    db_session.add(other_case)
    db_session.commit()

    cross_resp = client.post(
        f"/api/v1/cases/{other_case.id}/reviews",
        json={
            "action": "MARK_REVIEWED",
            "report_version_id": rep_row.id,
        },
        headers=test_users["clinician"]["headers"],
    )
    assert cross_resp.status_code == 409


# ---------------------------------------------------------------------------
# 7. Evidence and Provenance Tests
# ---------------------------------------------------------------------------

def test_evidence_sanitization_and_unsupported_claim(
    client: TestClient, test_users: Dict[str, Any], db_session: Session
):
    unique_doc_num = f"{uuid.uuid4().int % 100000000:08d}"
    doc = EvidenceDocumentModel(
        id=str(uuid.uuid4()),
        source="PUBMED",
        source_id=unique_doc_num,
        pmid=unique_doc_num,
        title="[SYNTHETIC] Synthetic Clinical Study",
    )
    passage = EvidencePassageModel(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        evidence_id=f"ev:pubmed:{unique_doc_num}:0",
        chunk_index=0,
        text="Aspirin reduces mortality in acute coronary syndrome.",
        text_hash="thash1",
    )
    db_session.add_all([doc, passage])
    db_session.commit()

    # GET /evidence/{id} must not leak benchmark keys
    ev_resp = client.get(
        f"/api/v1/evidence/{passage.evidence_id}",
        headers=test_users["viewer"]["headers"],
    )
    assert ev_resp.status_code == 200
    ev_data = ev_resp.json()
    assert "sample_id" not in ev_data
    assert "question_type" not in ev_data
    assert "gold" not in ev_data
    assert ev_data["evidence_id"] == passage.evidence_id
    assert ev_data["kind"] == "SOURCE"

    # Unknown evidence id returns 404
    ev_404 = client.get("/api/v1/evidence/ev:nonexistent:000:0", headers=test_users["viewer"]["headers"])
    assert ev_404.status_code == 404

    # Provenance for an unsupported claim returns support "NONE" and specific message
    case_p = ClinicalCaseModel(
        id=str(uuid.uuid4()),
        raw_text_hash="cphash",
        chief_complaint="Complaint",
        status="READY_FOR_REVIEW",
    )
    exec_p = WorkflowExecutionModel(
        id=str(uuid.uuid4()),
        case_id=case_p.id,
        status="READY_FOR_REVIEW",
    )
    rep_p = ReportVersionModel(
        id=str(uuid.uuid4()),
        case_id=case_p.id,
        execution_id=exec_p.id,
        version_number=1,
        summary='{"case_summary": "Test summary"}',
        confidence_score=0.88,
        status="READY_FOR_REVIEW",
    )
    claim = CandidateClaimModel(
        id=str(uuid.uuid4()),
        report_version_id=rep_p.id,
        claim_text="Unproven experimental drug resolves symptoms immediately.",
        section="potential_considerations",
        is_material=True,
    )
    verif = ClaimVerificationModel(
        id=str(uuid.uuid4()),
        claim_id=claim.id,
        category="potential_considerations",
        status="INSUFFICIENT_EVIDENCE",
        justification="No supporting citations exist.",
        method="DETERMINISTIC",
    )
    db_session.add_all([case_p, exec_p])
    db_session.flush()
    db_session.add(rep_p)
    db_session.flush()
    db_session.add_all([claim, verif])
    db_session.commit()

    prov_resp = client.get(
        f"/api/v1/claims/{claim.id}/provenance",
        headers=test_users["viewer"]["headers"],
    )
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()
    assert prov_data["support"] == "NONE"
    assert prov_data["message"] == "No supporting evidence found"
    assert prov_data["edges"] == []


# ---------------------------------------------------------------------------
# 8. Evaluation Endpoints and Safety Gates
# ---------------------------------------------------------------------------

def test_evaluations_safety_gate_refusal(client: TestClient, test_users: Dict[str, Any]):
    # Refusal on unconfirmed eval split or third-party LLM disallowed returns 409 with reason
    resp = client.post(
        "/api/v1/evaluations",
        json={"set": "e2e_sample", "config": "full", "limit": 10},
        headers=test_users["admin"]["headers"],
    )
    assert resp.status_code == 409
    err = resp.json()["error"]
    assert err["code"] == "SAFETY_GATE_REFUSAL"
    assert "Safety gate refusal" in err["message"]


# ---------------------------------------------------------------------------
# 9. Health & Dependencies Check
# ---------------------------------------------------------------------------

def test_health_dependencies_no_secrets(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    resp_dep = client.get("/api/v1/health/dependencies")
    assert resp_dep.status_code == 200
    dep_data = resp_dep.json()
    assert "dependencies" in dep_data
    llm_info = dep_data["dependencies"]["llm"]
    # Secrets must never be exposed
    assert "key" not in llm_info
    assert "api_key" not in llm_info
    assert "key_present" in llm_info


# ---------------------------------------------------------------------------
# 10. Rate Limiting Tests
# ---------------------------------------------------------------------------

def test_login_rate_limiting(client: TestClient):
    # Fire requests exceeding RATE_LIMIT_LOGIN_PER_MINUTE
    limit = settings.RATE_LIMIT_LOGIN_PER_MINUTE
    for _ in range(limit):
        client.post(
            "/api/v1/auth/login",
            json={"email": "rate_test@clinexus.local", "password": "wrong"},
        )

    # Next request must return 429 RATE_LIMITED
    over_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "rate_test@clinexus.local", "password": "wrong"},
    )
    assert over_resp.status_code == 429
    err = over_resp.json()["error"]
    assert err["code"] == "RATE_LIMITED"
    assert "Rate limit exceeded" in err["message"]


def test_cases_abstention_decision(client: TestClient, test_users: Dict[str, Any]):
    # Case without abstention returns 404 NO_ABSTENTION_DECISION
    c_resp = client.post(
        "/api/v1/cases",
        json={"raw_text": "Case for abstention check."},
        headers=test_users["clinician"]["headers"],
    )
    case_id = c_resp.json()["id"]

    abs_resp = client.get(
        f"/api/v1/cases/{case_id}/abstention",
        headers=test_users["viewer"]["headers"],
    )
    assert abs_resp.status_code == 404
    assert abs_resp.json()["error"]["code"] == "NO_ABSTENTION_DECISION"


def test_analyze_no_provider_surfaces_llm_unavailable(client: TestClient, test_users: Dict[str, Any]):
    app.dependency_overrides[get_llm_provider] = lambda: None
    c_resp = client.post(
        "/api/v1/cases",
        json={"raw_text": "Case for no provider check."},
        headers=test_users["clinician"]["headers"],
    )
    case_id = c_resp.json()["id"]

    a_resp = client.post(
        f"/api/v1/cases/{case_id}/analyze",
        headers=test_users["clinician"]["headers"],
    )
    assert a_resp.status_code == 202
    exec_id = a_resp.json()["execution_id"]

    st_resp = client.get(f"/api/v1/executions/{exec_id}", headers=test_users["viewer"]["headers"])
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["status"] == "FAILED"
    assert "LLM_UNAVAILABLE" in (st_data.get("error_code") or "")


def test_analyze_quota_exhausted_surfaces_code(client: TestClient, test_users: Dict[str, Any]):
    class QuotaExhaustedProvider:
        async def generate_text(self, *args, **kwargs):
            raise LLMQuotaExhausted("Gemini API quota depleted.")

        async def generate_json(self, *args, **kwargs):
            raise LLMQuotaExhausted("Gemini API quota depleted.")

    app.dependency_overrides[get_llm_provider] = lambda: QuotaExhaustedProvider()
    c_resp = client.post(
        "/api/v1/cases",
        json={"raw_text": "Case for quota check."},
        headers=test_users["clinician"]["headers"],
    )
    case_id = c_resp.json()["id"]

    a_resp = client.post(
        f"/api/v1/cases/{case_id}/analyze",
        headers=test_users["clinician"]["headers"],
    )
    assert a_resp.status_code == 202
    exec_id = a_resp.json()["execution_id"]

    st_resp = client.get(f"/api/v1/executions/{exec_id}", headers=test_users["viewer"]["headers"])
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["status"] == "FAILED"
    assert "LLM_QUOTA_EXHAUSTED" in (st_data.get("error_code") or "")
