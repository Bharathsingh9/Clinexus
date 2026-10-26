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
