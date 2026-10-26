Run one phase per conversation, in order. Each phase prompt assumes the agent has read .agent/rules/clinexus.md and docs/SPEC.md. A phase is finished only when its exit gate has been demonstrated by running the listed commands. If a gate cannot be met because of something in SPEC section 13, mark the dependent item BLOCKED, finish everything independent of it, and say so in the report.
Phase 0: Audit (no application code)
Goal: understand what exists and fix the plan before building.
Do:
1. Inventory the repository (tree, dependencies, existing code, tests, configs) into docs/repo_audit.md. Confirm or correct the note that the top level holds only datasets/, notebook/, and .gitignore.
2. Find the base-paper PDF. If found, extract the five agents, their inputs and outputs, the confidence mechanism, the iteration mechanism, and the evaluation protocol into docs/paper_summary.md, every item tagged per SPEC section 1. If not found, record it in docs/missing_prerequisites.md.
3. Create docs/requirements_traceability.md with columns Req ID | Requirement | Source | Module | Test file | Status. IDs: AGT-1..5, EXT-A..D, RET-n, API-n, UI-n, DB-n, SAF-n, EVL-n, DAT-n.
4. Run the dataset audit in SPEC 9.11: docs/dataset_audit.md, datasets/MANIFEST.yaml, scripts/check_dataset_pii.py, docs/notebook_audit.md. Reproduce every number in SPEC section 9 by running code, and list any difference.
5. Write docs/missing_prerequisites.md (SPEC section 13 plus anything you find) and a phase-by-phase plan listing the exact files you will create or modify.
Do not: write backend, frontend, or agent code; modify or move any data file or notebook.
Exit gate: all documents above exist; the dataset audit reproduces the section 9 numbers (or lists the differences); the traceability table covers every requirement in the SPEC.
Phase 1: Contracts and infrastructure
Goal: a running skeleton with validated schemas and persistence.
Do: implement the Pydantic models and the evidence-ID module (SPEC 6); settings and validation (SPEC 2); PostgreSQL models and Alembic migrations including evaluation_labels (SPEC 6); the Qdrant client wrapper; the LLMProvider protocol with FakeLLMProvider in tests; JSON logging that never records patient text; docker-compose.yml (api, postgres, qdrant, frontend placeholder only if a frontend service is required by compose, otherwise omit); .env.example; scripts/check_dependencies.py that reports the real status of each dependency.
Do not: write any agent, retrieval, API endpoint, or UI code.
Exit gate: docker compose up starts Postgres and Qdrant; alembic upgrade head succeeds on an empty database and a repeat run is a no-op; schema unit tests pass (including ID build/parse, required fields, extra=forbid); check_dependencies.py prints real results.
Phase 2: The five agents and ingestion (independently)
Goal: each original agent works alone and rejects invalid input and output.
Do: implement the ingestion pipeline (SPEC 9.3 rules: streaming, batched, idempotent by content hash, resumable, failed-record JSONL, --reindex creating a new collection, corpus_version) with the BioASQ corpus loader, because Agent 2 needs real retrieval; then implement Agents 1 to 5 exactly as in SPEC section 4, each with its unit-test file, using FakeLLMProvider and FakeNCBIClient only in tests. Test ingestion and retrieval against a real Qdrant container using [SYNTHETIC] fixtures.
Do not: connect agents to each other; build the orchestration graph, the verifier, the gate, or any API or UI.
Exit gate: each agent's unit tests pass; each agent rejects an invalid input and an invalid output; ingestion run twice on the fixture corpus creates no duplicates; a retrieval test returns passages with all required metadata.
Phase 3: Baseline workflow (the base-paper pipeline)
Goal: the five agents run end to end in the LangGraph workflow, before any extension.
Do: implement the workflow state and the nodes validate_case, clinical_text_clarifier, clarification_check, symptom_rag_analyzer, evidence_based_scanner, clinical_data_fusion, adaptive_optimizer, persist_and_finish, with AgentExecutionRecord persistence, failure semantics (SPEC 3), and the iteration cap. In this phase the verifier and the safety gate are not connected, so the optimizer works only from structural findings (empty retrieval, DEGRADED scanner, claims citing missing evidence). Record clearly in docs/architecture.md which parts are baseline.
Do not: add the verifier, gate, abstention, provenance edges, review states, API, or UI.
Exit gate: an integration test runs a fixture case through every node; each stage consumes the previous stage's real output; evidence IDs are consistent end to end; a scanner outage yields DEGRADED, not success; the optimizer stops at the iteration cap and reports BUDGET_EXHAUSTED without implying success.
Phase 4: Extensions A, B, C, D (in this order)
Goal: add the four approved contributions on real workflow outputs.
Do: (A) Claim Verification Agent (SPEC 5.1) and connect claim_verification; (B) abstention and evaluate_gate (SPEC 3 and 5.2) and connect evidence_safety_gate, finalize_ready, finalize_abstain, with every revision re-verified; (C) provenance edges, assert_provenance_complete, and the append-only audit trail (SPEC 5.3); (D) the state machine, JWT auth with roles, review actions, and report versioning (SPEC 5.4). Finish and test each extension before starting the next.
Do not: build the frontend or evaluation harness; do not let the optimizer or confidence score override a gate rule.
Exit gate: unit and integration tests pass for each extension; a failed material-claim verification can never reach READY_FOR_REVIEW; insufficient evidence ends ABSTAINED; assert_provenance_complete passes on real workflow output; every allowed and disallowed state transition is tested; audit events cannot be updated or deleted.
Phase 5: Real ingestion, dataset adapters, evaluation harness
Goal: reproducible evaluation on the BioASQ files, within what is confirmed.
Do: ingest the full corpus (SPEC 9.3) and verify that the source, Qdrant, and PostgreSQL counts are equal; implement the questions-file adapter with the accepted shapes (9.4), the derived labels and coverage report (9.5), the deterministic split and the three evaluation sets (9.6), the leakage guards (9.7), benchmark mode (9.8), the AnswerExtractor and scorers (9.9), the ablation flags and the run-JSON writer (SPEC 10), and scripts/evaluate_bioasq.py. Metrics that need an unconfirmed manifest entry run only as smoke tests and are labeled UNCONFIRMED; metrics that need an LLM stay BLOCKED until a provider is configured.
Do not: report any metric from an unconfirmed manifest entry; use the files listed in SPEC 9.2; build a MedQA adapter unless MedQA files were found in Phase 0; hardcode any target number.
Exit gate: ingestion counts are equal; all dataset tests in SPEC section 11 pass; an evaluation run on fixtures writes valid run JSON; the smoke30 retrieval run completes and labels its output CLOSED_WORLD_SNIPPET_CORPUS and NON_OFFICIAL_SPLIT.
Phase 6: API completion and frontend
Goal: a usable application wired to the real backend.
Do: complete exactly the endpoints in SPEC 7 and the 7 pages with the UI rules; connect every control to a real endpoint; show real agent status and errors; make unsupported claims and abstention visible; implement the review workflow in the UI.
Do not: add pages, endpoints, or statistics beyond SPEC 7; animate progress that the backend has not reported.
Exit gate: Playwright E2E passes the full flow (submit, watch real status, open report, open claim evidence, view verification, abstention scenario, record review, view audit history); grep finds no unwired control; frontend unit tests pass.
Phase 7: Hardening and release
Goal: a reproducible release candidate with no falsely completed features.
Do: run every test suite from a clean checkout following only the README; fix integration failures; verify restart persistence, authorization, and privacy controls (no patient text in logs); check dependency pinning; write docs/safety_and_limitations.md, docs/api_guide.md, docs/evaluation_protocol.md, and the evaluation report containing only results actually measured; finalize docs/known_gaps.md and the traceability table.
Do not: mark a requirement DONE without a passing test; include any unmeasured metric or any claim of clinical validity.
Exit gate (final report): every requirement ID is DONE with a passing test or BLOCKED with a documented reason; the report contains the test execution summary, the measured evaluation results only, the list of BLOCKED and incomplete requirements, and the known limitations.


I updated Appendix A in docs/SPEC.md (items A1 to A10). Re-read it. Then update only these files: docs/change_proposals.md, docs/decision_log.md, docs/paper_summary.md, docs/requirements_traceability.md. Do not edit SPEC.md and do not start Phase 1.

1. change_proposals.md: set CP-01 to CP-06 to APPROVED or DEFERRED with date and the Appendix item. Correct CP-05: the paper's workflow is sequential (its Scanner consumes the RAG Analyzer's hypotheses), not parallel. Correct CP-06: replace the quoted SPEC text with the actual wording of SPEC section 3 and Appendix A3. Add CP-07 for candidate hypotheses (A5), status APPROVED.
2. paper_summary.md: fix section 4, row "Termination on Budget" (the gate decides, ready or abstain, per A3). Reclassify the paper's treatment generation as [DEVIATION-FROM-PAPER: SAFETY], not [PAPER-GAP]. Fix section 5: the split is dev/eval (SPEC 9.6), and BioASQ scorers follow the published guidelines or are labeled approximations (SPEC 9.9). Add the differences in A5 to A9.
3. requirements_traceability.md: add requirements for CandidateHypothesis (AGT-2), per-hypothesis scanning (AGT-3), the embedding model resolution (RET-n), and per-agent model settings.
4. Resolve the Hugging Face ID of the paper's embedding model by checking it exists (read-only lookup, no downloads of weights) and record the result in docs/decision_log.md.

Reply with the list of files you changed.