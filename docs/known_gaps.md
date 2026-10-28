# Known Gaps

In accordance with Rule **R5** and Rule **R10**, any incomplete item, missing prerequisite, or deferred capability is cataloged here. Work is explicitly stopped on each item until prerequisites are provided.

---

## Catalog of Known Gaps (Phase 0 Baseline)

| Gap ID | Topic | Description | Status & Action |
| :--- | :--- | :--- | :--- |
| **GAP-01** | LLM Provider Integration | Provider, model, and API credentials are unset (SPEC Section 13). | **BLOCKED**: Real LLM calls in all agents are deferred. Tests use `FakeLLMProvider`. |
| **GAP-02** | Base Paper Confidence Formula | Base paper is silent on the mathematical formula for confidence calculation. | Documented in `docs/paper_summary.md` and `docs/change_proposals.md` (CP-04). Engineering formula from SPEC Section 2 selected. |
| **GAP-03** | Missing Context Snippets in BioASQ Corpus | The corpus contains 75,919 snippets against 80,058 context snippets in the questions file (4,139 snippets omitted). | Cause is **UNKNOWN** per SPEC 9.10; documented in `docs/dataset_audit.md`. |
| **GAP-04** | MedQA, MedBullets, PubMedQA Code Adapters | Datasets exist in repository but are outside SPEC Section 9 scope. | Marked `AUDITED_NOT_SPECIFIED` in manifest. Adapters/loaders deferred per CP-01, CP-02, CP-03. |
| **GAP-05** | Official BioASQ Split Absence | `training14b.json` and `bioasq_normalized.csv` do not contain official train/test splits. | Handled via deterministic 20% dev / 80% eval stratified split (seed 42) in Phase 5 per SPEC 9.6; tagged `NON_OFFICIAL_SPLIT`. |
| **GAP-06** | Hardware / Container Infrastructure | PostgreSQL and Qdrant services are not running yet. | Scheduled for startup in Phase 1 via `docker-compose.yml`. |
| **GAP-07** | External PubMed Retrieval Searchability | External PubMed evidence is stored only in PostgreSQL (`evidence_documents` and `evidence_passages`) and not in Qdrant; therefore, external passages are not searchable by vector retrieval refinement (`symptom_rag_analyzer`). | Architectural decision DEC-032 preserving BioASQ vector index purity. In feedback iterations, external evidence is passed via in-memory/relational state rather than vector similarity search. |
| **GAP-08** | Scope Check Limitation for General Non-Biomedical Requests | Deterministic scope check in `backend/app/safety/scope_rules.py` detects requests for treatment, prescription, medication ordering, or dosing via curated regexes (SPEC Rule 1, `[ENGINEERING-DECISION: SCOPE-REGEX]`). It does not attempt to detect arbitrary non-biomedical queries (e.g. general conversational chit-chat). In benchmark mode, biomedical factual queries bypass this check per SPEC Section 9.8. | Intentional engineering scope boundary. Open-domain intent classification is out of scope for Phase 4. |
| **GAP-09** | Urgent-Symptom Pattern Set Clinical Limitations | The urgent rules pattern set in `backend/app/safety/urgent_rules.yaml` and `backend/app/safety/urgent.py` (chest pain with dyspnea, sudden focal neurological deficit, suicidal ideation statements, severe bleeding, anaphylaxis signs) is an uncalibrated engineering prototype heuristic designed for synthetic safety demonstration. It is NOT clinically validated, cannot account for unstated symptoms or complex atypical presentations, and must never be used for clinical triage or emergency department routing without extensive institutional clinical review and regulatory approval (SPEC Section 8, Rule R11). | Prototype ruleset implemented for Phase 4 Stage B (SAF-3); clinical validation deferred to future medical governance and institutional evaluation. |

