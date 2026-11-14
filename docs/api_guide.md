# Clinexus REST API Guide

This document describes the Phase 6 REST API for the Clinexus research-grade clinical decision support platform.

> [!IMPORTANT]
> **Research Tool Disclaimer**: Clinexus is a clinical-research decision support prototype and is not clinically validated or cleared as a medical device. System outputs require independent clinical judgment and evaluation by licensed healthcare professionals.

---

## 1. Overview & Base URL

- **API Base Prefix**: `/api/v1`
- **OpenAPI 3.0 Specification**: `/api/v1/openapi.json`
- **Standard Content-Type**: `application/json`
- **Authentication Header**: `Authorization: Bearer <JWT_ACCESS_TOKEN>`
- **Idempotency Header**: `Idempotency-Key: <UUID_OR_STRING>` (supported on `POST /cases/{id}/analyze`)

---

## 2. Server-Side Authorization Matrix

All authorization checks are enforced server-side. The platform defines three discrete roles:
- `admin`: Full system administration, user provisioning, and evaluation benchmarks.
- `clinician_reviewer`: Licensed clinical users authorized to submit cases, trigger multi-agent analysis, record structured reviews, request revisions, and add clinical notes.
- `viewer`: Read-only observer account. Strictly forbidden from executing mutating operations.

| Endpoint | Method | Permitted Roles | Notes |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | **Public** | Basic liveness probe. |
| `/health/dependencies` | `GET` | **Public** | Diagnostic readiness check. Never returns secrets or credentials. |
| `/auth/login` | `POST` | **Public** | Issues JWT bearer token. Rate-limited. |
| `/users` | `GET` | `admin` | Lists system users. Password hashes are strictly omitted. |
| `/users` | `POST` | `admin` | Provisions new user with password length $\ge 12$ chars. |
| `/cases` | `POST` | `clinician_reviewer`, `admin` | Submits case. Evaluates urgent rules before analysis. |
| `/cases` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Summary case listing. **Strictly omits narrative free text.** |
| `/cases/{id}` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Detailed case view including narrative text and urgency banner. |
| `/cases/{id}/analyze` | `POST` | `clinician_reviewer`, `admin` | Starts background LangGraph analysis. Idempotent. Returns 202. |
| `/executions/{id}` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Top-level execution progress and terminal status. |
| `/executions/{id}/agents` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Agent telemetry. **Strictly omits prompts and model outputs.** |
| `/cases/{id}/report` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Latest synthesized clinical report version with claims. |
| `/cases/{id}/reports` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Chronological report version history for a case. |
| `/reports/{version_id}/claims` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Claims for report version with verification status. |
| `/claims/{claim_id}/provenance` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Evidence edges for claim. Returns `support: "NONE"` if unsupported. |
| `/evidence/{evidence_id}` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Passage text and source metadata. Excludes gold/benchmark keys. |
| `/cases/{id}/abstention` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Explicit abstention decision. Returns 404 if not abstained. |
| `/cases/{id}/reviews` | `POST` | `clinician_reviewer`, `admin` | Records append-only clinician review action. |
| `/cases/{id}/clarifications` | `POST` | `clinician_reviewer`, `admin` | Records `REQUEST_CLARIFICATION` action. Does not trigger analysis. |
| `/cases/{id}/refresh-evidence` | `POST` | `clinician_reviewer`, `admin` | Records `REQUEST_EVIDENCE_REFRESH` action. Transitions to `REVISION_REQUESTED`. |
| `/cases/{id}/revisions` | `POST` | `clinician_reviewer`, `admin` | Records `REQUEST_REVISION` action. Transitions to `REVISION_REQUESTED`. |
| `/cases/{id}/audit` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Chronological immutable audit trail for case and reports. |
| `/evaluations` | `POST` | `admin` | Dispatches BioASQ evaluation run with safety gate enforcement. |
| `/evaluations` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Lists past evaluation benchmark runs. |
| `/evaluations/{id}` | `GET` | Authenticated (`viewer`, `clinician_reviewer`, `admin`) | Aggregated metrics and labels. Never leaks benchmark text. |

---

## 3. Error Response Contract

All errors return a consistent, structured JSON envelope:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable explanation of error.",
    "details": null
  }
}
```

### Standard Error Codes

| HTTP Status | Error Code | Description |
| :--- | :--- | :--- |
| **401** | `UNAUTHENTICATED` | Missing, invalid, or expired bearer token. Login failure also returns this code without leaking whether email exists. |
| **403** | `FORBIDDEN` | Authenticated user lacks permission to execute the action (e.g. viewer invoking analysis). |
| **404** | `NOT_FOUND` | The requested entity (case, execution, report, claim, evidence) does not exist. |
| **404** | `NO_ABSTENTION_DECISION` | Returned by `GET /cases/{id}/abstention` when the case completed without an explicit abstention decision. |
| **409** | `INVALID_STATE` | Illegal state transition requested (e.g. triggering analysis while case is already `ANALYZING`). |
| **409** | `CONFLICT` | Entity conflict (e.g. email already exists, or report version does not belong to the case). |
| **409** | `SAFETY_GATE_REFUSAL` | Evaluation safety gate refused execution (e.g. unconfirmed manifest or missing LLM permission). |
| **422** | `VALIDATION_ERROR` | Request payload schema validation failure. Patient text is stripped from details. |
| **429** | `RATE_LIMITED` | Exceeded in-memory sliding window request threshold. |
| **500** | `INTERNAL_ERROR` | Unhandled internal server exception. Internal stack traces are suppressed. |

---

## 4. Content Kind Labeling

To ensure transparency regarding information origin, response objects carry explicit `kind` annotations:
- `PATIENT`: Patient-submitted narratives and clinical case presentation data.
- `SOURCE`: Retrieved published biomedical evidence passages and document metadata.
- `INTERPRETATION`: AI multi-agent synthesis, claims, candidate considerations, and abstention decisions.
- `HUMAN`: Clinician review actions, verification notes, and audit event logs.

---

## 5. Request & Response Examples

### 5.1 Case Submission (`POST /api/v1/cases`)

**Request**:
```http
POST /api/v1/cases HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "raw_text": "A 54-year-old patient presents with acute shortness of breath and substernal chest pressure for 2 hours."
}
```

**Response (201 Created)**:
```json
{
  "id": "e8a93e32-2d1f-4b05-a81d-e08d6c70b8a1",
  "status": "DRAFT",
  "urgent": true,
  "urgent_banner": "Possible urgent presentation: do not delay appropriate urgent human assessment",
  "created_at": "2026-11-03T08:30:00Z"
}
```

### 5.2 Trigger Analysis (`POST /api/v1/cases/{id}/analyze`)

**Request**:
```http
POST /api/v1/cases/e8a93e32-2d1f-4b05-a81d-e08d6c70b8a1/analyze HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_TOKEN>
Idempotency-Key: req-case-001-attempt-1
```

**Response (202 Accepted)**:
```json
{
  "execution_id": "c1a2b3c4-d5e6-4f7a-8b9c-0d1e2f3a4b5c",
  "status": "RUNNING"
}
```

### 5.3 Evidence Provenance (`GET /api/v1/claims/{claim_id}/provenance`)

**Response (Supported Claim)**:
```json
{
  "claim_id": "clm-101",
  "support": "SUPPORTED",
  "message": null,
  "edges": [
    {
      "edge_id": "edg-501",
      "relation": "SUPPORTS",
      "evidence_id": "ev:bioasq:E000001:0",
      "passage_text": "Elevated troponin levels with ST-segment elevation are diagnostic criteria for myocardial infarction.",
      "source_metadata": {
        "pmid": "12345678",
        "title": "Clinical management of acute coronary syndromes",
        "source": "PUBMED",
        "journal": "New England Journal of Medicine",
        "publication_date": "2023"
      },
      "retrieval_score": 0.85,
      "retrieval_timestamp": "2026-11-03T08:31:00Z",
      "verification_justification": "Direct textual entailment supported by cited passage.",
      "evidence_spans": [],
      "kind": "SOURCE"
    }
  ],
  "kind": "INTERPRETATION"
}
```

**Response (Unsupported Claim)**:
```json
{
  "claim_id": "clm-102",
  "support": "NONE",
  "message": "No supporting evidence found",
  "edges": [],
  "kind": "INTERPRETATION"
}
```

### 5.4 Clinician Review (`POST /api/v1/cases/{id}/reviews`)

**Request**:
```http
POST /api/v1/cases/e8a93e32-2d1f-4b05-a81d-e08d6c70b8a1/reviews HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "action": "MARK_REVIEWED",
  "report_version_id": "rep-v1-uuid",
  "note": "Verified considerations align with acute clinical presentation."
}
```

**Response (201 Created)**:
```json
{
  "review_id": "rev-301",
  "case_id": "e8a93e32-2d1f-4b05-a81d-e08d6c70b8a1",
  "report_version_id": "rep-v1-uuid",
  "reviewer_id": "usr-admin-01",
  "reviewer_role": "clinician_reviewer",
  "action": "MARK_REVIEWED",
  "outcome": "Reviewed by clinician",
  "note": "[HUMAN] Verified considerations align with acute clinical presentation.",
  "created_at": "2026-11-03T08:35:00Z",
  "kind": "HUMAN"
}
```
