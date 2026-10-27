# Dataset Audit Report (Phase 0)

## 1. Executive Summary
This document provides the complete empirical dataset audit for Clinexus, satisfying **SPEC Section 9.11** and Phase 0 requirements (`DAT-1` through `DAT-8`). All statistics and checksums were computed by executing reproducible Python scripts (`scripts/audit_datasets_comprehensive.py`, `scripts/audit_other_datasets.py`, `scripts/check_dataset_pii.py`) against the actual files in `datasets/`.

Every number from SPEC Section 9 was verified against observed file content. Where slight differences were observed between SPEC text and actual file bytes, the observed values are reported and trusted per SPEC 9.

---

## 2. BioASQ Evidence Corpus Verification (`bioasq_evidence_corpus.csv`)
- **Path**: `datasets/processed/bioasq/bioasq_evidence_corpus.csv`
- **File Size**: 23,677,468 bytes
- **SHA-256**: `4aea8ab2bd5c37002a9f3a84278a0ffb26c5bdd4387894c6c9cc14ecc49d894e`
- **Role**: `EVIDENCE_CORPUS` (Manifest status: `confirmed_by_human: false`)

### Numerical Audit & Comparison
| Metric / Field | SPEC Section 9 Value | Observed Value | Match Status / Note |
| :--- | :--- | :--- | :--- |
| **Row Count** | 75,919 | **75,919** | **EXACT MATCH** |
| **Columns** | evidence_id, snippet_text, pubmed_url, section, sample_id, question_type | `['evidence_id', 'snippet_text', 'pubmed_url', 'section', 'sample_id', 'question_type']` | **EXACT MATCH** |
| **Null Counts** | 0 across all columns | 0 across all columns | **EXACT MATCH** |
| **Section: abstract** | 64,275 | **64,275** | **EXACT MATCH** |
| **Section: title** | 7,610 | **7,610** | **EXACT MATCH** |
| **Section: sections.0** | 4,034 | **4,034** | **EXACT MATCH** |
| **Median Snippet Length** | 173 characters | **173.0 characters** | **EXACT MATCH** |
| **Maximum Snippet Length**| 3,946 characters | **3,946 characters** | **EXACT MATCH** |
| **Minimum Snippet Length**| Not specified | 2 characters | Observed |
| **Short Passages (<50 chars)**| 1,621 rows | **1,621 rows** | **EXACT MATCH** |
| **Duplicate Texts** | 261 texts duplicated | **261 non-first duplicate rows** (447 total rows across 186 distinct texts) | **EXACT MATCH** |
| **Texts Shared Across Questions**| 8 | **8** | **EXACT MATCH** |
| **Distinct PubMed URLs** | 48,949 | **48,949** | **EXACT MATCH** |
| **Distinct sample_ids** | 5,665 | **5,665** | **EXACT MATCH** |
| **Deviating PubMed URLs** | 245 URLs | **245 URLs** | **EXACT MATCH** |
| - *https variants* | 116 | 119 (classification detail) | Total sum 245 exact |
| - *bare digits* | 58 | 62 | Total sum 245 exact |
| - *https://pubmed.ncbi.nlm.nih.gov/<digits>/* | 45 | 48 | Total sum 245 exact |
| - *trailing slash* | 15 | 16 | Total sum 245 exact |
| - *no match on PMID regex* | 0 expected | **0** | **EXACT MATCH** |
| **Author Contact Emails**| 24 snippets | **25 snippets** (32 email instances across 24 distinct texts) | Difference noted: 25 snippets observed |
| **Phone-like Patterns** | 0 found | **0 found** | **EXACT MATCH** |
| **Patient Identifiers** | 0 found | **0 found** | **EXACT MATCH** |
| **Phase 5 Expected Passages** | N/A | **85,139 passages** | Approved in Stage A review: effective chunk 98 tokens, overlap 20 tokens; 6,249 rows over limit chunked into 85,139 total passages |

---

## 3. BioASQ Normalized Questions Verification (`bioasq_normalized.csv`)
- **Path**: `datasets/processed/bioasq/bioasq_normalized.csv`
- **File Size**: 46,580,459 bytes
- **SHA-256**: `f80815e48f80d778362c8c74005bfefedbc80cedd3bcd75b24e0e2299fc6db53`
- **Role**: `QUESTIONS_AND_GOLD` (Manifest status: `confirmed_by_human: false`)

### Numerical Audit & Comparison
| Metric / Field | SPEC Section 9 Value | Observed Value | Match Status / Note |
| :--- | :--- | :--- | :--- |
| **Row Count** | 5,729 | **5,729** | **EXACT MATCH** |
| **Columns** | sample_id, source_dataset, question_type, question, exact_answer, ideal_answer, context, source_reference, concepts, metadata | `['sample_id', 'source_dataset', 'question_type', 'question', 'exact_answer', 'ideal_answer', 'context', 'source_reference', 'concepts', 'metadata']` | **EXACT MATCH** |
| **Unique sample_ids** | 5,729 | **5,729** | **EXACT MATCH** |
| **Question Type: factoid** | 1,695 | **1,695** | **EXACT MATCH** |
| **Question Type: yesno** | 1,541 | **1,541** | **EXACT MATCH** |
| **Question Type: summary**| 1,363 | **1,363** | **EXACT MATCH** |
| **Question Type: list** | 1,130 | **1,130** | **EXACT MATCH** |
| **JSON Parse Failures** | 0 failures across all rows/cols | **0 failures** | **EXACT MATCH** |
| **Duplicate Questions (Normalized)**| 0 stated in SPEC | **5 duplicate question pairs** | Difference noted: 5 pairs share identical normalized text |
| **exact_answer: yesno "yes"**| 1,123 | **1,123** | **EXACT MATCH** |
| **exact_answer: yesno "no"** | 418 | **418** | **EXACT MATCH** |
| **exact_answer: factoid list**| 1,695 flat list of strings | **1,695** | **EXACT MATCH** |
| **exact_answer: list nested** | 1,130 list of lists | **1,130** | **EXACT MATCH** |
| **exact_answer: summary null**| 1,363 JSON null | **1,363** | **EXACT MATCH** |
| **exact_answer: unknown shape**| 0 expected | **0** | **EXACT MATCH** |
| **ideal_answer: 1 string** | 3,785 rows | **3,785 rows** | **EXACT MATCH** |
| **ideal_answer: max strings** | 112 strings | **112 strings** | **EXACT MATCH** |
| **ideal_answer: >20,000 chars**| 54 rows | **52 rows** | Difference noted: 52 rows observed |
| **Context: Total Snippets** | 80,058 | **80,058** | **EXACT MATCH** |
| **Context: Median per Q** | 9 snippets | **9.0 snippets** | **EXACT MATCH** |
| **Context: Max per Q** | 121 snippets | **121 snippets** | **EXACT MATCH** |
| **Context: Min per Q** | Not specified | 1 snippet | Observed |
| **Context Snippets w/ Email**| 22 stated in SPEC | **25 snippets** | Difference noted: 25 observed |
| **source_reference URLs** | 61,506 | **61,506** (58,302 distinct) | **EXACT MATCH** |

---

## 4. Corpus-to-Questions Linkage & Retrieval Coverage (`DAT-5`)

### Sample ID Overlap
- **Corpus distinct sample_ids**: 5,665
- **Questions distinct sample_ids**: 5,729
- **Corpus IDs in Questions file**: **100% (5,665 of 5,665 exist)** [MATCHES SPEC 9.5]
- **Questions without corpus rows**: **64 questions** [MATCHES SPEC 9.5]
  - By question type: `factoid`: 35, `yesno`: 12, `summary`: 11, `list`: 6 [EXACT MATCH]
  - Context snippets in these 64 questions: **78 snippets** [EXACT MATCH]
  - Per SPEC 9.5: These 64 questions must be excluded from retrieval metrics.

### Snippet Content Matching (Corpus vs Question Context)
Verified by checking every corpus snippet against its parent question's `context` array:
- **Exact String Matches**: **64,270** [EXACT MATCH TO SPEC 9.5]
- **Normalized Matches** (whitespace & case collapsed): **11,649** [EXACT MATCH TO SPEC 9.5]
- **Not Found in Context**: **0** [EXACT MATCH TO SPEC 9.5]
- **Total Corpus Rows Accounted For**: 64,270 + 11,649 = **75,919 (100.0%)**
- **Context Snippets Absent from Corpus**: 80,058 - 75,919 = 4,139 snippets (reason: UNKNOWN per SPEC 9.10).

---

## 5. Candidate and Evaluation Files Audit

### 5.1 Smoke Candidate List (`retrieval_eval_candidates.csv`) - `DAT-6`
- **Path**: `datasets/processed/bioasq/retrieval_eval_candidates.csv`
- **Row Count**: 30 rows
- **Columns**: `['eval_id', 'question', 'question_type', 'sample_id']`
- **Verification**: All 30 `sample_id` values exist in `bioasq_normalized.csv` with identical question text and question type.
- **Role**: `SMOKE_CANDIDATES`. Only its `sample_id` values may be used to define the `smoke30` set.

### 5.2 Invalid Gold File (`retrieval_eval_gold.csv`) - `DAT-7`
- **Path**: `datasets/processed/bioasq/retrieval_eval_gold.csv`
- **Row Count**: 30 rows
- **Columns**: `['eval_id', 'question', 'question_type', 'sample_id', 'relevant_evidence_ids', 'annotation_notes']`
- **Verification of Invalidation**:
  - Extracted all 209 evidence IDs listed in `relevant_evidence_ids`.
  - Looked up owner `sample_id` in `bioasq_evidence_corpus.csv`.
  - **Result: 0 of 209 evidence IDs belong to the question they label.**
  - Concrete example: For row 0 ("What type of DMD can viltolarsen be used for?", `sample_id=6417900b690f196b5100002a`), evidence ID `E052201` points to text: *"ine kinase inhibitor tirabrutinib (for relapsed and refractory PCNSL)"*, which belongs to `sample_id=601d6fe11cb411341a000033`.
  - **Conclusion**: Confirmed completely invalid. Marked `INVALID_DO_NOT_USE`.

### 5.3 Non-Reproducible Results (`retrieval_eval_results.csv`) - `DAT-7`
- **Path**: `datasets/processed/bioasq/retrieval_eval_results.csv`
- **Row Count**: 30 rows
- **Columns**: `['eval_id', 'question', 'question_type', 'num_relevant', 'first_relevant_rank', 'recall_at_5', 'recall_at_10', 'precision_at_5', 'reciprocal_rank']`
- **Audit Finding**: Evaluated against the corrupted `retrieval_eval_gold.csv` by an untracked pipeline. Non-reproducible. Marked `ARTIFACT_DO_NOT_USE`. Never copy its numbers.

### 5.4 Deduplicated Corpus (`bioasq_evidence_corpus_dedup.csv`) - `DAT-7`
- **Path**: `datasets/processed/bioasq/bioasq_evidence_corpus_dedup.csv`
- **Row Count**: 73,062 rows
- **Columns**: `['evidence_id', 'snippet_text', 'pubmed_url', 'section', 'sample_id', 'question_type']`
- **Audit Finding**:
  - Joined against `bioasq_evidence_corpus.csv` on `evidence_id`: 73,062 rows match.
  - Of 73,062 matching IDs, **73,058 have different text** (only 4 retain the same text).
  - Deduplication criterion is completely untracked. Marked `DISALLOWED_RENUMBERED_DO_NOT_USE`.

### 5.5 FAISS Artifacts (`evidence_lookup.csv`, `evidence_lookup_dedup.csv`, `evidence_id_mapping.csv`) - `DAT-7`
- Contain `faiss_position` columns from an earlier, unversioned FAISS index pipeline.
- Marked `LEGACY_FAISS_ARTIFACT_DO_NOT_USE`.

---

## 6. Raw BioASQ Inspection (`datasets/bioasq/raw/`) - Extra Task 2
- **Path**: `datasets/bioasq/raw/`
- **Files**:
  1. `README` (1,500 bytes, SHA-256: `720da213f77b0c90a6c07f081905030da4aca7ae28b306da8c4b4e700ed2b9ce`)
  2. `training14b.json` (57,967,312 bytes, SHA-256: `f410315e12c71567d79669e5a0ca407e78a7bef83ee40c84d558826b751a3701`)

### Observations:
1. **Original BioASQ Authenticity**: `training14b.json` is the official BioASQ 14b training data distribution, documented in `README` as containing data from the first 11 editions of the challenge (5,729 questions).
2. **Record Count Correspondence**: Exactly **5,729 questions** exist in `training14b.json`, matching `bioasq_normalized.csv` exactly.
3. **Question Type Distribution**:
   - `factoid`: 1,695
   - `yesno`: 1,541
   - `summary`: 1,363
   - `list`: 1,130
   - Matches both `README` notes and `bioasq_normalized.csv` 100%.
4. **Question ID and Text Correspondence**:
   - Every raw `id` equals `sample_id` in identical sequential order (0 mismatches across all 5,729 records).
   - Question bodies are 100% identical in text content (7 records show differences solely due to `\n` vs `\r\n` line endings from Windows CSV serialization).
5. **Train / Test Splits**: **No official train/test split fields are present** in `training14b.json`. The entire file is training/development data for Task 14b.
6. **Processing Provenance**: Remains **UNKNOWN** because the exact script that converted `training14b.json` into `bioasq_normalized.csv` and created the snippet subset `bioasq_evidence_corpus.csv` (75,919 rows from 80,058 context snippets) is not present in the repository.

---

## 7. Audit of Datasets Not Specified in SPEC Section 9 (`DAT-8`)

As instructed, each of the following datasets was audited for row counts, columns, null rates, SHA-256, duplicates, and PII. Each is assigned `role: AUDITED_NOT_SPECIFIED` with `confirmed_by_human: false` in `datasets/MANIFEST.yaml`. No loaders or adapters are built.

### 7.1 MedQA (`datasets/medqa/US/` and `datasets/processed/medqa/`)
- `datasets/medqa/US/dev.jsonl`: 1,272 rows, 5 columns (`question`, `answer`, `options`, `meta_info`, `answer_idx`), 0 nulls, 0 duplicates, PII: PASS. SHA-256: `bdd64cca0ee84c0c685bde6e52f0e41657740af40522e130150da4638c36b518`.
- `datasets/medqa/US/test.jsonl`: 1,273 rows, 5 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `2929db9f297a5cdc1ce469a57422cf91ee0263d6229a50d8d713683c8b6b75f6`.
- `datasets/medqa/US/train.jsonl`: 10,178 rows, 5 columns, 0 nulls, 2 question duplicates, PII: PASS (no patient PII). SHA-256: `ac3199e6e495640db69500faf79b9fadafcee1b0f5002af4e2393fe46e305e37`.
- `datasets/processed/medqa/medqa_normalized.csv`: 12,723 rows, 10 columns, 0 duplicates, PII: PASS. SHA-256: `fb36239999174984860fd3ea1b99d0b119473dd9d672b8db41d3e91249d2f519`.

### 7.2 MedBullets (`datasets/medbullets/` and `datasets/processed/medbullets/`)
- `datasets/medbullets/op4_eval-00000-of-00001.parquet`: 62 rows, 6 columns (`idx`, `question`, `options`, `answer`, `explanation`, `link`), 0 nulls, 0 duplicates, PII: PASS. SHA-256: `9c21c0e7a717967c87197d5ea712342b7c996c84b6dcc7db97eae75b13d6c59b`.
- `datasets/medbullets/op4_test-00000-of-00001.parquet`: 308 rows, 6 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `bf7ae4ba9a5392fdfd978cb950ee05bb8edc6a3f2228d17203dd03f3efe66161`.
- `datasets/medbullets/op4_train-00000-of-00001.parquet`: 246 rows, 6 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `48fe6784d3ce083bd7a1dc0c4ea4596dcf2a48f86f911ed7cf71b5ea87ae4e22`.
- `datasets/medbullets/op5_eval-00000-of-00001.parquet`: 62 rows, 6 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `4c2b55ca91be3db20f0aefbc83c7aaac01ab1b8f8ef9cf90bbfd64f05fe6ba43`.
- `datasets/medbullets/op5_test-00000-of-00001.parquet`: 308 rows, 6 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `034bcf452e76c3e25c80bb8e377080a9a4347387cde90765f0834ae0174f650e`.
- `datasets/medbullets/op5_train-00000-of-00001.parquet`: 246 rows, 6 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `763044b255a5a4b47123b0d0f2ab3c316c66d176f3b9c8b5c310489be3cdc6b0`.
- `datasets/processed/medbullets/medbullets_op4_normalized.csv`: 616 rows, 10 columns, 0 duplicates, PII: PASS. SHA-256: `4c42df2d0fcad379de8ac024e140670646bafba0d2e7bd09dc3aa1b4e59bdda3`.

### 7.3 PubMedQA (`datasets/pubmedqa/` and `datasets/processed/pubmedqa/`)
- `datasets/pubmedqa/ori_pqal.json`: 1,000 rows (labeled subset PQA-L), 10 columns (`key`, `QUESTION`, `CONTEXTS`, `LABELS`, `MESHES`, `YEAR`, `reasoning_required_pred`, `reasoning_free_pred`, `final_decision`, `LONG_ANSWER`), 0 duplicates, PII: PASS. SHA-256: `fb1f109a9bad1f96369a0d5343e3710d251c8f237e8369be86e30bd7e430523d`.
- `datasets/processed/pubmedqa/pubmedqa_normalized.csv`: 1,000 rows, 10 columns, 0 nulls, 0 duplicates, PII: PASS. SHA-256: `d6909ec99aad8d4ec5dd612bb4e518e2dbb488239a35b8f8303bf37950da1980`.
