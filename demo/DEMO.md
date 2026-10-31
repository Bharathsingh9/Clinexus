# Clinexus Live Project Review Demonstration Runbook

> **DEMONSTRATION SCOPE & GOVERNANCE DISCLOSURE**
> This demonstration suite is a one-off execution authorized by the project owner for formal review (logged under `DEC-040` in `docs/decision_log.md`). It exercises the real clinical decision support (CDS) backend architecture up through Phase 4 in isolated demo environments without altering existing application code, tests, migrations, or production collections.

---

## 1. Honesty Disclosure & Architecture Reality

Every scenario executed in this suite prints the mandatory honesty banner:
```text
========================================================================================
HONESTY DISCLOSURE:
REAL: embeddings, Qdrant retrieval, PostgreSQL, workflow orchestration, deterministic verifier checks, safety gate, provenance, audit.
SCRIPTED: language-model outputs and the verifier's entailment judgment (demo stand-in, no real LLM connected).
========================================================================================
```

| Component | Status | Implementation Details |
| :--- | :--- | :--- |
| **Biomedical Embeddings** | **REAL** | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-dim, 100-token limit) |
| **Dense Vector Index** | **REAL** | Qdrant collection `evidence_biobert_mnli_snli_scinli_scitail_mednli_stsb_v777` (3,049 vector points) |
| **Relational Storage** | **REAL** | PostgreSQL 16 database `clinexus_demo` (21 tables, Alembic migrations applied) |
| **Workflow Orchestration** | **REAL** | LangGraph 12-node state graph with conditional routing and cycle controls |
| **Deterministic Verifier** | **REAL** | Entity/citation checks and regex numeric extraction veto (`ClaimVerificationAgent`) |
| **Evidence Safety Gate** | **REAL** | Sequential 6-rule gate (`evaluate_gate`, SAF-1, SAF-2, EXT-B) |
| **Urgent Symptom Alert** | **REAL** | Pre-analysis pattern matcher (`evaluate_urgent_rules`, SAF-3) |
| **Provenance Graph** | **REAL** | Complete relational provenance edge persistence (`SPEC 5.3`) |
| **Audit Trail** | **REAL** | Append-only immutable repository (`AuditEventModel`) |
| **HITL Review & RBAC** | **REAL** | 8-state machine transition table and role-based access control (`ReviewService`) |
| **Language Model Outputs** | **SCRIPTED** | `ScriptedDemoLLMProvider` (demo stand-in; claims dynamically cite real retrieved IDs) |
| **Entailment Judgment** | **SCRIPTED** | Synthetic span entailment output (demo stand-in) |
| **Clinical Diagnostic Advice**| **NONE** | System strictly adheres to invariant SAF-1: zero diagnoses, prescriptions, or dosing orders |
| **Accuracy / Recall Numbers** | **NONE** | Closed-world subset demo; unconfirmed labels; not an evaluation result |

---

## 2. Pre-Flight Checklist

Before launching the demonstration, verify all preconditions:

1. **Docker Engine Running & Containers Healthy**:
   ```bash
   docker compose ps
   ```
   *Expected*: `clinexus_postgres` (healthy on port `5432`) and `clinexus_qdrant` (healthy on port `6333`).

2. **Python Virtual Environment Activated**:
   Use the project environment (`C:\Major Project\Clinexus\.venv` or `C:\Users\karth\.conda\envs\clinexus`):
   ```powershell
   & "C:\Major Project\Clinexus\.venv\Scripts\python.exe" --version
   ```
   *Expected*: Python 3.11.x.

3. **BioBERT Model Cache Warm**:
   The HuggingFace model `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` is cached locally under `~/.cache/huggingface/hub/`. Loading takes < 1 second.

4. **Verify Test Suite Integrity**:
   Run the regression test suite to ensure the baseline remains 100% green:
   ```powershell
   & "C:\Major Project\Clinexus\.venv\Scripts\python.exe" -m pytest backend/tests -v
   ```
   *Expected*: **174 passed, 1 deselected** (0 failures).

---

## 3. Demonstration Command Sequence

Execute the following commands in order:

### Step 1: Demo Corpus Ingestion (Pre-computed / Optional Re-run)
*Duration*: ~3 minutes for 2,717 rows.
*Note*: The demo collection `v777` in `clinexus_demo` is already populated. If re-ingestion is required:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/build_demo_data.py
```
*Expected Output*:
- Times 200 BioBERT embeddings (~25 passages/sec).
- Selects 217 candidate passages + 2,500 distractors (seed 42) = 2,717 total rows.
- Runs real `IngestionPipeline` into `clinexus_demo` and Qdrant collection `v777`.
- Source rows: `2,717` | PostgreSQL documents: `2,601` | Passages: `3,049` | Qdrant points: `3,049`.

---

### Step 2: Dense Retrieval Demonstration
*Duration*: ~2.5 seconds.
Demonstrates real BioBERT query embedding and Qdrant dense vector search over the demo corpus:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/demo_retrieval.py
```
To run for a specific candidate question:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/demo_retrieval.py --question-id Q007
```
*Expected Output*:
- Prints 3 contrasting questions (`Q007`, `Q016`, `Q028`).
- Displays retrieval trace fields: `corpus_version: 777`, `embedding_model: pritamdeka/BioBERT-...`, and query text.
- Prints top-5 retrieved passages with `rank`, `score`, `evidence_id`, `PMID`, `source`, and text preview.
- Clearly labeled with `[CLOSED-WORLD SNIPPET CORPUS]` and `illustrative, subset corpus, unconfirmed labels, not an evaluation result`.

---

### Step 3: End-to-End Workflow & Governance Demonstration
*Duration*: ~8 to 10 seconds total (unpaused), or presenter-controlled with `--pause`.
Demonstrates the real LangGraph extended orchestrator, deterministic verification, safety gate, provenance graph, and HITL review:

**Presenter-Paced Run (Waits for Enter between scenarios)**:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/demo_workflow.py --pause
```

**Automated Continuous Run**:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/demo_workflow.py
```

#### What Each Scenario Shows:
- **Scenario A: READY Path (`~2.2s`)**
  - Input: Factual benchmark inquiry on formin protein in the snail (`Q007`).
  - Workflow: 12-node LangGraph orchestration executes real retrieval and deterministic verification.
  - Scripted model synthesizes claims citing real retrieved evidence IDs (`ev:bioasq:E026601:0`).
  - Evidence Safety Gate: Evaluates Rules 1 through 6 in sequence; outcome: `finalize_ready`.
  - State: Transitions to `READY_FOR_REVIEW`.
  - Provenance: Logs 12 relational provenance edges (`GENERATED_IN`, `SUPPORTS`, `RETRIEVED_BY`) per SPEC 5.3.

- **Scenario B: UNSUPPORTED Path (`~2.1s`)**
  - Input: Same inquiry, but scripted claim injects number `"99"` absent from literature passages.
  - Deterministic Numeric Veto: Real Python regex veto catches entity mismatch; downgrades status from `SUPPORTED` to `PARTIALLY_SUPPORTED` despite scripted judge.
  - Adaptive Optimizer: Automatically triggers up to 3 optimization loops before hard budget cap (`MAX_OPTIMIZER_ITERATIONS = 3`).
  - Evidence Safety Gate: Triggers `finalize_abstain` with reason codes `['UNSUPPORTED_MATERIAL_CLAIMS', 'ITERATION_BUDGET_EXHAUSTED']`.
  - Output Sanity: All unverified candidate considerations are suppressed/redacted from final report.

- **Scenario C: URGENT + NO EVIDENCE Path (`~1.8s`)**
  - Input: `[SYNTHETIC] 62-year-old patient presenting with acute substernal chest pain with shortness of breath.`
  - Pre-Analysis Safety: `evaluate_urgent_rules` matches pattern immediately and surfaces urgent red banner before workflow execution.
  - Invariant SAF-3: Urgent banner is displayed immediately; workflow execution continues without blockage.
  - Closed-World Retrieval: Real retrieval yields 0 relevant passages above threshold in demo corpus.
  - Safe Abstention: Gate Rule 4 fires; report abstains safely without presenting any speculative clinical diagnosis or prescription.

- **Scenario D: HITL Review & Audit Path (`~0.8s`)**
  - Users: Creates `clinician_demo@clinexus.local` (`clinician_reviewer`) and `viewer_demo@clinexus.local` (`viewer`) in `clinexus_demo` (Argon2id passwords loaded securely from env).
  - RBAC Enforcement: Viewer attempts `MARK_REVIEWED` -> instantly caught and rejected with `UnauthorizedActionError`.
  - Clinician Review: Clinician executes `MARK_REVIEWED` with note. Case status updates to `REVIEWED` (recorded as "Reviewed by clinician", never "approved").
  - Append-Only Persistence: Clinician adds second note (`ADD_NOTE`). Both reviews preserved; earlier review is never overwritten.
  - State Machine Invariant: Attempting illegal transition (`REVIEWED -> DRAFT`) is rejected with `InvalidTransition`.
  - Audit Trail: Prints immutable chronological audit history for the case (`CASE_STATE_CHANGED`, `HUMAN_REVIEW_RECORDED`).

- **Summary Screen**:
  - Displays comprehensive table breaking down Real vs Scripted components.

---

## 4. Troubleshooting Guide

### Failure 1: Docker Containers Not Running
*Symptom*: Connection refused to `localhost:5432` or `localhost:6333`.
*Remedy*:
```powershell
docker compose up -d
docker compose ps
```
Wait 5 seconds for PostgreSQL healthcheck to report `healthy`.

### Failure 2: HuggingFace Model Download / Offline Failure
*Symptom*: Network timeout when initializing `SentenceTransformer("pritamdeka/BioBERT...")`.
*Root Cause*: Model is not in local cache and external network is offline or unauthenticated.
*Remedy*:
Verify cache directory `C:\Users\karth\.cache\huggingface\hub\models--pritamdeka--BioBERT*`.
If empty, warm cache once while connected to network:
```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')"
```

### Failure 3: PostgreSQL or Qdrant Port Conflict
*Symptom*: `bind: address already in use` for `5432` or `6333`.
*Remedy*:
Check for existing processes binding those ports:
```powershell
Get-NetTCPConnection -LocalPort 5432, 6333 -ErrorAction SilentlyContinue | Select-Object LocalPort, OwningProcess
```
Stop conflicting local PostgreSQL/Qdrant services or stop and restart containers:
```powershell
docker compose down
docker compose up -d
```

---

## 5. Live Demo Fallback Strategy

If any live hardware or environment issue occurs during the presentation:
1. **Container Failure Fallback**:
   If Docker Engine terminates unexpectedly, present the verified terminal logs preserved from the end-to-end run in task logs:
   - Workflow Output: Preserved in conversation transcript.
   - Retrieval Output: Run offline test assertions in `backend/tests/integration/test_stage_b_integration.py`.
2. **Database Re-initialization Fallback**:
   If `clinexus_demo` data is accidentally corrupted:
   ```powershell
   & "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/build_demo_data.py
   ```
   Rebuilding takes under 3 minutes.
3. **Recorded Walkthrough Fallback**:
   All four scenarios execute deterministically; review outputs match the recorded terminal logs in Section 3 of this document.

---

## 6. Honest Project Status (Requirements Traceability)

The following metrics are derived directly from `docs/requirements_traceability.md` without modifying any requirement statuses:

| Area | Prefix | DONE | NOT_STARTED | BLOCKED | Total | Status / Phase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Multi-Agent Clinical CDS** | `AGT` | 7 | 0 | 0 | 7 | 100% Complete (Phase 2 & 4) |
| **Stage B Safety Extensions** | `EXT` | 4 | 0 | 0 | 4 | 100% Complete (Phase 4) |
| **Workflow Orchestration** | `WKF` | 1 | 0 | 0 | 1 | 100% Complete (Phase 4) |
| **Retrieval & Embedding Infrastructure** | `RET` | 8 | 0 | 0 | 8 | 100% Complete (Phase 2 & 3) |
| **Configuration & Model Settings** | `CFG` | 1 | 0 | 0 | 1 | 100% Complete (Phase 1) |
| **Database Persistence & Storage** | `DB` | 5 | 1 | 0 | 6 | Core complete; restart verification pending |
| **Clinical Safety Guardrails** | `SAF` | 2 | 3 | 0 | 5 | Urgent rules & phrase linter complete |
| **Dataset Verification & Auditing** | `DAT` | 8 | 0 | 0 | 8 | 100% Complete (Phase 1) |
| **FastAPI REST Endpoints** | `API` | 0 | 10 | 0 | 10 | Phase 5 (Not Started) |
| **Frontend Interface & Pages** | `UI` | 0 | 8 | 0 | 8 | Phase 5 (Not Started) |
| **Evaluation & Benchmarking Harness** | `EVL` | 0 | 5 | 0 | 5 | Phase 6 (Not Started) |
| **TOTAL** | **ALL** | **36** | **27** | **0** | **63** | **57.1% Overall System Complete** |

### Implementation Boundary Summary
- **Completed Layers (Phases 1-4)**: Data pipeline, BioBERT embeddings, Qdrant index, 5-agent LangGraph workflow, claim-level deterministic verification, 6-rule safety gate, provenance edge tracking, and 8-state HITL review state machine with append-only audit logging.
- **Pending Layers (Phases 5-6)**: FastAPI REST endpoints, React/TypeScript web UI, and formal offline benchmark evaluation harness.

---

## 7. Demo Viewer — Visual Scenario Output (DEC-041)

A read-only, self-contained HTML viewer at `demo/viewer/index.html` visualises the output of
the four demo scenarios. It is **not** the Clinexus application interface (Phase 6).
Data comes from actually running `export_demo_results.py` against `clinexus_demo` and Qdrant v777.
Authorization: DEC-041 (`docs/decision_log.md`).

### Step A: Export scenario results to JSON
*Duration*: ~6–10 seconds.

```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/viewer/export_demo_results.py
```

*Expected output*:
```
[export] Running Scenario A (READY path)...
  [OK] Scenario A: READY_FOR_REVIEW in 2.34s, 12 provenance edges, 1 claims.
[export] Running Scenario B (UNSUPPORTED / numeric veto path)...
  [OK] Scenario B: ABSTAINED in 1.99s, 3 optimizer iterations.
[export] Running Scenario C (URGENT + NO EVIDENCE path)...
  [OK] Scenario C: ABSTAINED in 1.67s, urgent=True.
[export] Running Scenario D (HITL Review / Audit path)...
  [OK] Scenario D: 5 audit events, 2 reviews recorded.

[OK] Wrote 37,537 bytes -> demo/viewer/data/results.json
     Total export time: 6.21s
```

### Step B: Build the self-contained HTML viewer
*Duration*: < 2 seconds.

```powershell
& "C:\Major Project\Clinexus\.venv\Scripts\python.exe" demo/viewer/build_viewer.py
```

*Expected output*:
```
[OK] Loaded results.json: 4 scenarios, generated ...
[OK] Wrote demo/viewer/index.html (79.9 KB)
[OK] No external http resource references found (src=, href=, @import).
[OK] All 4 scenario panels verified in index.html.
```

### Step C: Open in browser

```powershell
start demo\viewer\index.html
```

Or double-click `demo/viewer/index.html` in Explorer. Works completely offline — no CDN, no external fonts.

### Viewer Features

| Feature | Description |
| :--- | :--- |
| Permanent honesty banner | Displayed at top of every page load |
| Scenario tabs A–D | Keyboard-accessible tab navigation |
| Horizontal pipeline | Nodes colored: blue=completed, amber=partial, red=failed/vetoed, purple=abstained |
| Claims table | Click a claim row to expand evidence, verification status, and limitations |
| Gate panel | Shows which gate rule fired and reason codes |
| Abstention panel | Shows reason codes and next steps when workflow abstains |
| Provenance panel (A) | Shows 12 relational provenance edges per SPEC 5.3 |
| Numeric veto panel (B) | SCRIPTED judge vs REAL code veto side-by-side |
| Audit timeline (D) | [HUMAN] notes visually distinct from AI content; append-only trail |
| REAL/SCRIPTED tags | Every panel labeled with its data origin |
| Light / dark mode | Automatic via CSS `prefers-color-scheme` |
| No network requests | Zero CDN, fonts, or external deps — works offline |
