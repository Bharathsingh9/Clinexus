# Notebook Audit Report (Phase 0)

## 1. Governance & Usage Policy
Per **SPEC Section 9.11** and Rule **R8**:
- Notebooks under `notebook/` are **historical reference material only**.
- Notebook code is **never imported** by application code.
- Stored notebook outputs are **never copied** or presented as platform benchmarks.
- Any useful logic (e.g. data preprocessing, vector encoding) must be re-implemented as clean, tested, typed application modules.
- Zero modifications were made to `notebook/`.

---

## 2. Notebook Inventory & PII Scan Summary

| Notebook Path | Total Cells | Code Cells | Markdown Cells | PII in Outputs | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `notebook/01_dataset_exploration.ipynb` | 35 | 35 | 0 | **0 violations** | PASS |
| `notebook/02_evidence_retrieval.ipynb` | 79 | 45 | 34 | **0 violations** | PASS |
| `notebook/03_claim_verification.ipynb` | 19 | 9 | 10 | **0 violations** | PASS |

Output cells in all three notebooks were scanned using `scripts/check_dataset_pii.py` pattern matchers for patient names in clinical context, dates of birth, medical record numbers, Social Security numbers, and phone numbers. **Zero patient identifiers were detected.**

---

## 3. Detailed Audit by Notebook

### 3.1 `notebook/01_dataset_exploration.ipynb`
Explores raw biomedical QA datasets (MedQA, MedBullets, PubMedQA, and BioASQ) and serializes them into normalized CSV formats.

- **Dataset Paths Read**:
  - `../datasets/medqa/US/train.jsonl`
  - `../datasets/medqa/US/dev.jsonl`
  - `../datasets/medqa/US/test.jsonl`
  - `../datasets/pubmedqa/ori_pqal.json`
  - `../datasets/medbullets/op4_train-00000-of-00001.parquet`
  - `../datasets/medbullets/op4_eval-00000-of-00001.parquet`
  - `../datasets/medbullets/op4_test-00000-of-00001.parquet`
  - `../datasets/medbullets/op5_train-00000-of-00001.parquet`
  - `../datasets/medbullets/op5_eval-00000-of-00001.parquet`
  - `../datasets/medbullets/op5_test-00000-of-00001.parquet`
  - `../datasets/bioasq/raw/training14b.json`
- **Dataset Paths Written**:
  - `../datasets/processed/medqa/medqa_normalized.csv` (12,723 rows)
  - `../datasets/processed/pubmedqa/pubmedqa_normalized.csv` (1,000 rows)
  - `../datasets/processed/medbullets/medbullets_op4_normalized.csv` (616 rows)
  - `../datasets/processed/bioasq/bioasq_normalized.csv` (5,729 rows)
- **Models Used**: None (pure data inspection and schema structuring).
- **Preprocessing Steps**:
  - Standardized column schema: `sample_id`, `source_dataset`, `split`, `question`, `context`, `options`, `ground_truth_answer`, `explanation`, `source_reference`, `metadata`.
  - Parsed USMLE 4-option and 5-option multiple choice formats into JSON string options.
  - Formatted PubMedQA `ori_pqal.json` into normalized rows with `CONTEXTS` concatenated.
  - Extracted BioASQ `training14b.json` into tabular structure with JSON-serialized exact and ideal answers.
- **Printed Metrics**: None (data exploration only).

### 3.2 `notebook/02_evidence_retrieval.ipynb`
Prototypes dense embedding, FAISS indexing, and retrieval evaluation over BioASQ snippets.

- **Dataset Paths Read**:
  - `datasets/processed/bioasq/bioasq_normalized.csv`
  - `datasets/processed/bioasq/bioasq_evidence_corpus.csv`
- **Dataset Paths Written / Exported**:
  - `datasets/processed/bioasq/bioasq_evidence_corpus_dedup.csv` (73,062 rows)
  - `datasets/processed/bioasq/evidence_lookup.csv` (75,919 rows)
  - `datasets/processed/bioasq/evidence_lookup_dedup.csv` (73,062 rows)
  - `datasets/processed/bioasq/evidence_id_mapping.csv` (73,062 rows)
  - `datasets/processed/bioasq/retrieval_eval_candidates.csv` (30 rows)
  - `datasets/processed/bioasq/retrieval_eval_gold.csv` (30 rows, invalid)
  - `datasets/processed/bioasq/retrieval_eval_results.csv` (30 rows, invalid)
- **Models Used**:
  - `pritamdeka/S-PubMedBert-MS-MARCO` (via `SentenceTransformer`)
  - `PubMedBERT`
- **Preprocessing & Indexing Steps**:
  - Deduplicated snippets based on verbatim text matching (`drop_duplicates(subset=['snippet_text'])`).
  - Assigned sequential zero-based integer indices for FAISS index alignment (`faiss_position`).
  - Extracted 30 candidate questions based on snippet count heuristics.
  - Attempted to construct gold labels by pairing eval questions with FAISS positions, which introduced the cross-question misalignment bug in `retrieval_eval_gold.csv`.
- **Printed Metrics**:
  - Printed cosine similarity scores (e.g. `Rank 1 [Score: 0.9584] E000002`).
  - Outputted evaluation metrics across the corrupted gold file (`Recall@5: 0.70`, `Recall@10: 0.76`, `MRR: 0.75`). Per SPEC 9.2, these numbers are ungrounded artifacts and must never be referenced as system performance.

### 3.3 `notebook/03_claim_verification.ipynb`
Prototypes LLM-based claim verification against retrieved evidence passages.

- **Dataset Paths Read**:
  - `datasets/processed/bioasq/evidence_lookup_dedup.csv`
- **Models Referenced / Loaded**:
  - Explores local LLM loading for verification using PyTorch `bfloat16` and device map `auto` (noting GPU VRAM constraints).
- **Preprocessing Steps**:
  - Inspects deduplicated evidence lookup table.
  - Sets up prompt templates for fact-checking claims against evidence snippets.
- **Printed Metrics**:
  - Hardware / precision diagnostics (`Selected Precision: torch.bfloat16`, `Device Map: auto`).
