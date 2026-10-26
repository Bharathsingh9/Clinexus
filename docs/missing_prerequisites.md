# Missing Prerequisites & Implementation Plan

## 1. Missing Prerequisites (Per SPEC Section 13 & Phase 0 Audit)

The following items can only be supplied or confirmed by a human reviewer. Until provided, dependent components remain **BLOCKED** or unconfirmed.

| # | Item Description | Impact & Blocked Status | Action Required from Human |
| :- | :--- | :--- | :--- |
| **1** | **LLM Provider, Model Name, and API Key** | Real LLM calls in all agents (`AGT-1..5`, `EXT-A`) are **BLOCKED**. Unit and integration tests proceed using `FakeLLMProvider`. | Provide provider (`openai`, `anthropic`, `gemini`, etc.), model names, and API keys in `.env`. |
| **2** | **Base Paper PDF** | **RESOLVED IN PHASE 0**: Found at `docs/paper/base_paper.pdf` and audited into `docs/paper_summary.md`. | Verified. |
| **3** | **BioASQ Dataset License & Human Confirmation** | Datasets marked `confirmed_by_human: false` in `datasets/MANIFEST.yaml`. Metrics run only as smoke tests and are labeled `UNCONFIRMED`. | Review manifest and confirm license terms for BioASQ files. |
| **4** | **Derived Relevance Labels Approval (SPEC 9.5)** | Corpus-to-question sample_id relevance mapping cannot be treated as official ground truth. | Confirm that linking corpus rows by `sample_id` is approved for benchmark retrieval scoring. |
| **5** | **Evaluation Protocol Confirmation** | Split and scoring rules remain unconfirmed. | Confirm: (a) factoid gold-answer interpretation (any string in gold list is correct), (b) 20% dev / 80% eval split with seed 42, (c) default sample size $N=200$. |
| **6** | **NCBI E-utilities Credentials (Optional)** | Live PubMed scanning defaults to rate-limited mode (3 req/s) or `EXTERNAL_MODE=offline`. | Supply `NCBI_EMAIL` and `NCBI_API_KEY` in `.env` if higher rate limits (10 req/s) are desired. |
| **7** | **Node.js Environment (Phase 6)** | Node runtime is `v24.18.0`. SPEC 2 calls for Node for frontend in Phase 6. Not blocking in Phase 1; recorded for frontend setup. | Verify Node compatibility during Phase 6 setup. |
| **8** | **Docker Container Runtime in CLI Session** | **RESOLVED IN PHASE 1 EXIT GATE**: Docker Desktop 29.9.2 and Docker Compose v5.6.1 operational. PostgreSQL 16 and Qdrant containers running and healthy. Live Alembic migrations and container integration tests passed. | Verified. |

---

## 2. Phase-by-Phase Implementation Plan

This plan details the exact files to create or modify across each phase of the Clinexus platform implementation.

### Phase 1: Contracts and Infrastructure
- **Prerequisites**: Docker, PostgreSQL, Qdrant.
- **Files to Create**:
  - `backend/app/schemas/ids.py`: Platform evidence ID builder and parser (`ev:<source>:<source_id>:<chunk_index>`).
  - `backend/app/schemas/clinical_case.py`: Pydantic v2 `ClinicalCase`, `StructuredSymptom`, `PatientContext`, `ClarificationRequest`.
  - `backend/app/schemas/evidence.py`: `RetrievalQuery`, `EvidenceDocument`, `EvidenceChunk`, `RetrievedEvidenceSet`, `ExternalEvidenceSet`.
  - `backend/app/schemas/reports.py`: `CandidateClaim`, `ClaimVerification`, `CandidateClinicalReport`, `AbstentionDecision`.
  - `backend/app/schemas/executions.py`: `AgentExecutionRecord`, `OptimizationIteration`, `AuditEvent`, `WorkflowResult`, `WorkflowState`.
  - `backend/app/core/config.py`: `pydantic-settings` environment configuration with strict validation.
  - `backend/app/core/logging_config.py`: Privacy-compliant JSON logger (logs IDs/lengths, never patient text).
  - `backend/app/persistence/database.py`: SQLAlchemy 2.x async engine and session factory.
  - `backend/app/persistence/models.py`: PostgreSQL relational ORM models (users, roles, clinical_cases, etc.).
  - `backend/migrations/env.py`, `backend/migrations/script.py.mako`, `backend/alembic.ini`: Alembic migration configuration.
  - `backend/migrations/versions/001_initial_schema.py`: Initial schema migration.
  - `backend/app/retrieval/qdrant_client.py`: Qdrant client wrapper for collection lifecycle and search.
  - `backend/app/core/llm_provider.py`: `LLMProvider` protocol definition.
  - `backend/tests/fixtures/fake_llm_provider.py`: `FakeLLMProvider` implementation for deterministic test mocking.
  - `scripts/check_dependencies.py`: Real dependency validation script for DB, Qdrant, and env variables.
  - `docker-compose.yml`: Container definitions for PostgreSQL 16 and Qdrant.
  - `.env.example`: Safe environment configuration template.
- **Files to Modify**: `docs/requirements_traceability.md`, `docs/decision_log.md`.

### Phase 2: The Five Agents and Ingestion (Independently)
- **Files to Create**:
  - `backend/app/ingestion/chunker.py`: Text chunker implementing recursive character splitting.
  - `backend/app/ingestion/bioasq_loader.py`: Streaming BioASQ corpus loader with idempotency and hash checks.
  - `backend/app/retrieval/embedding_service.py`: `SentenceTransformer` wrapper.
  - `backend/app/agents/clinical_text_clarifier.py`: Agent 1 (`AGT-1`) module.
  - `backend/app/agents/data/symptom_vocab.json`: Local symptom vocabulary.
  - `backend/app/agents/symptom_rag_analyzer.py`: Agent 2 (`AGT-2`) module.
  - `backend/app/evidence_sources/ncbi_client.py`: Agent 3 (`AGT-3`) NCBI E-utilities client.
  - `backend/app/agents/evidence_based_scanner.py`: Agent 3 (`AGT-3`) module.
  - `backend/app/safety/forbidden_phrases.py`: Imperative dosing/treatment regex linting.
  - `backend/app/agents/clinical_data_fusion.py`: Agent 4 (`AGT-4`) module.
  - `backend/app/agents/adaptive_optimizer.py`: Agent 5 (`AGT-5`) module.
  - `backend/tests/unit/test_clinical_text_clarifier.py`: Agent 1 unit tests.
  - `backend/tests/unit/test_symptom_rag_analyzer.py`: Agent 2 unit tests.
  - `backend/tests/unit/test_evidence_based_scanner.py`: Agent 3 unit tests.
  - `backend/tests/unit/test_clinical_data_fusion.py`: Agent 4 unit tests.
  - `backend/tests/unit/test_adaptive_optimizer.py`: Agent 5 unit tests.
  - `backend/tests/integration/test_ingestion_and_retrieval.py`: Corpus ingestion and Qdrant search integration tests.
- **Files to Modify**: `docs/requirements_traceability.md`, `docs/decision_log.md`.

### Phase 3: Baseline Workflow (Base-Paper Pipeline)
- **Files to Create**:
  - `backend/app/orchestration/state.py`: LangGraph state TypedDict.
  - `backend/app/orchestration/nodes/validate_case.py`: Case submission validation node.
  - `backend/app/orchestration/nodes/clarification_check.py`: Clarification check routing node.
  - `backend/app/orchestration/workflow.py`: Baseline LangGraph workflow graph definition.
  - `backend/tests/integration/test_baseline_workflow.py`: End-to-end integration test of 5-agent baseline loop.
- **Files to Modify**: `docs/architecture.md`, `docs/requirements_traceability.md`, `docs/decision_log.md`.

### Phase 4: Extensions A, B, C, D
- **Files to Create**:
  - **Extension A**: `backend/app/agents/claim_verification_agent.py`, `backend/tests/unit/test_claim_verification_agent.py`.
  - **Extension B**: `backend/app/safety/abstention.py` (`evaluate_gate`, `AbstentionDecision`), `backend/app/safety/urgent_rules.yaml`, `backend/app/safety/urgency_checker.py`, `backend/tests/unit/test_safety_gate.py`.
  - **Extension C**: `backend/app/provenance/provenance_tracker.py`, `backend/app/provenance/assertion.py` (`assert_provenance_complete`), `backend/tests/unit/test_provenance.py`.
  - **Extension D**: `backend/app/orchestration/states.py` (state transitions), `backend/app/security/auth.py` (JWT & Argon2), `backend/tests/unit/test_state_machine.py`.
- **Files to Modify**: `backend/app/orchestration/workflow.py`, `docs/requirements_traceability.md`, `docs/decision_log.md`.

### Phase 5: Real Ingestion, Dataset Adapters, Evaluation Harness
- **Files to Create**:
  - `scripts/ingest_documents.py`: CLI ingestion script for full 75,919 corpus.
  - `backend/app/evaluation/dataset_split.py`: Stratified 20/80 seed 42 split generator (`splits_v1.json`).
  - `backend/app/evaluation/bioasq_scoring.py`: Scorers for factoid, yesno, list, summary.
  - `backend/app/evaluation/answer_extractor.py`: Separate extractor prompt and parser.
  - `backend/app/evaluation/evaluation_harness.py`: Benchmark harness running ablations and outputting run JSON.
  - `scripts/evaluate_bioasq.py`: CLI evaluation runner.
  - `backend/tests/unit/test_bioasq_evaluation.py`: Unit tests for scoring, splits, and leakage prevention.
- **Files to Modify**: `docs/requirements_traceability.md`, `docs/evaluation_protocol.md`, `docs/decision_log.md`.

### Phase 6: API Completion and Frontend
- **Files to Create**:
  - `backend/app/api/router.py`: FastAPI API v1 routing.
  - `backend/app/api/endpoints/{health,auth,users,cases,executions,reports,evidence,claims,abstention,reviews,audit,evaluations}.py`: Complete REST endpoints.
  - `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/vite.config.ts`, `frontend/package.json`.
  - `frontend/src/pages/{Dashboard,CaseSubmission,AnalysisWorkflow,ExplainableReport,EvidenceExplorer,HumanReview,EvaluationDashboard}Page.tsx`: Exactly 7 pages.
  - `frontend/src/components/{ResearchBanner,ClaimList,EvidenceCard,AuditTimeline}.tsx`.
  - `frontend/tests/*.test.tsx`: Vitest & React Testing Library unit tests.
  - `frontend/e2e/workflow.spec.ts`: Playwright E2E test.
- **Files to Modify**: `docs/requirements_traceability.md`, `docs/api_guide.md`, `docs/decision_log.md`.

### Phase 7: Hardening and Release
- **Files to Create / Finalize**:
  - `README.md`: Complete reproducible setup instructions.
  - `docs/safety_and_limitations.md`: Final limitations, disclaimer, and safety analysis.
  - `docs/api_guide.md`: Complete OpenAPI guide.
  - `docs/evaluation_protocol.md`: Measured benchmark metrics report.
  - `docs/known_gaps.md`: Final catalog of all deferred or unconfirmed items.
- **Files to Modify**: `docs/requirements_traceability.md`.

---

## 3. Environment Setup (2026-10-26)

- **Conda Environment**: `clinexus`
- **Interpreter Path**: `C:\Users\karth\.conda\envs\clinexus\python.exe`
- **Python Version**: `Python 3.11.17`
- **Docker Status**: `working` (Docker Desktop 29.9.2, Docker Compose v5.6.1, WSL2 backend running, `docker run --rm hello-world` verified)
- **Remaining Blockers**:
  - Human review confirmations detailed in Section 1 (LLM API keys, BioASQ license confirmation, derived relevance labels, evaluation protocol).

