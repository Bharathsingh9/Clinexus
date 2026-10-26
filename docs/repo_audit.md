# Repository Audit (Phase 0)

## 1. Executive Summary
This document provides a complete inventory and audit of the Clinexus repository prior to any application implementation, fulfilling the requirements of **Phase 0** and requirement **ARCH-0**.

- **Repository Root**: `c:\Major Project\Clinexus`
- **Audit Date**: 2026-10-26
- **Git Commit**: `069e572` (HEAD: "Add BioASQ processed datasets, evidence retrieval and claim verification notebooks")
- **Active Branch**: `main`

## 2. Directory Tree and File Inventory

### 2.1 Top-Level Inventory & Correction
**Specification Note Review**:
The prompt and specification initially noted that the top level holds only `datasets/`, `notebook/`, and `.gitignore`.
**Audit Finding / Correction**:
The top level contains the following directories and files:
1. `.agent/`: Contains `.agent/rules/clinexus.md` (the mandatory always-on governance rules R1–R12).
2. `.git/`: Git repository metadata.
3. `.gitignore`: Git ignore patterns for Python, environments, IDEs, and binary embeddings.
4. `datasets/`: Raw and processed datasets (24 distinct data files).
5. `docs/`: System documentation, specifications (`docs/SPEC.md`, `docs/PHASES.md`), and the base paper PDF (`docs/paper/base_paper.pdf`).
6. `notebook/`: Three exploratory Jupyter notebooks (`01_dataset_exploration.ipynb`, `02_evidence_retrieval.ipynb`, `03_claim_verification.ipynb`).
7. `scripts/`: Directory added during Phase 0 containing dataset and PII audit utilities (`check_dataset_pii.py`, `audit_datasets_comprehensive.py`, `audit_other_datasets.py`, `audit_notebooks.py`, `generate_manifest.py`).

### 2.2 Complete File Tree
```
Clinexus/
├── .agent/
│   └── rules/
│       └── clinexus.md
├── .gitignore
├── datasets/
│   ├── MANIFEST.yaml (Generated Phase 0)
│   ├── bioasq/
│   │   └── raw/
│   │       ├── README
│   │       └── training14b.json
│   ├── medbullets/
│   │   ├── op4_eval-00000-of-00001.parquet
│   │   ├── op4_test-00000-of-00001.parquet
│   │   ├── op4_train-00000-of-00001.parquet
│   │   ├── op5_eval-00000-of-00001.parquet
│   │   ├── op5_test-00000-of-00001.parquet
│   │   └── op5_train-00000-of-00001.parquet
│   ├── medqa/
│   │   └── US/
│   │       ├── dev.jsonl
│   │       ├── test.jsonl
│   │       └── train.jsonl
│   ├── processed/
│   │   ├── bioasq/
│   │   │   ├── bioasq_evidence_corpus.csv
│   │   │   ├── bioasq_evidence_corpus_dedup.csv
│   │   │   ├── bioasq_normalized.csv
│   │   │   ├── evidence_id_mapping.csv
│   │   │   ├── evidence_lookup.csv
│   │   │   ├── evidence_lookup_dedup.csv
│   │   │   ├── retrieval_eval_candidates.csv
│   │   │   ├── retrieval_eval_gold.csv
│   │   │   └── retrieval_eval_results.csv
│   │   ├── medbullets/
│   │   │   └── medbullets_op4_normalized.csv
│   │   ├── medqa/
│   │   │   └── medqa_normalized.csv
│   │   └── pubmedqa/
│   │       └── pubmedqa_normalized.csv
│   └── pubmedqa/
│       └── ori_pqal.json
├── docs/
│   ├── PHASES.md
│   ├── SPEC.md
│   └── paper/
│       └── base_paper.pdf
├── notebook/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_evidence_retrieval.ipynb
│   └── 03_claim_verification.ipynb
└── scripts/
    ├── check_dataset_pii.py
    ├── audit_datasets_comprehensive.py
    ├── audit_other_datasets.py
    ├── audit_notebooks.py
    └── generate_manifest.py
```

## 3. Environment & Runtime Inventory

- **Host Operating System**: Windows 11 (64-bit)
- **Primary Python Runtime (System)**: Python 3.12.4 (`C:\Program Files\Python312\python.exe`)
- **Secondary Python Runtime (Data/ML)**: Python 3.13.9 (`C:\ProgramData\anaconda3\python.exe`), equipped with `pyarrow 21.0.0`, `pyyaml 6.0.3`, `pandas 2.2.2`.
- **Existing Application Code**: None (clean baseline; no backend, frontend, or agent code exists in the repository).
- **Existing Test Suites**: None (no `backend/tests/` or `frontend/tests/` created yet).
- **Configuration Files**: `.gitignore` is present; no `.env` or `docker-compose.yml` present yet (scheduled for Phase 1).

## 4. Git Ignore Audit
Inspection of `.gitignore` against all repository paths:
- Python cache patterns (`__pycache__/`, `*.py[cod]`, `*$py.class`) are ignored.
- Virtual environment patterns (`.env`, `.venv/`, `env/`, `venv/`, `ENV/`) are ignored.
- IDE patterns (`.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`) are ignored.
- Binary embedding and index patterns (`*.npy`, `*.index`) are ignored.
- **Data File Check**: None of the 24 files under `datasets/` are blocked by `.gitignore`. No files are Git LFS pointers (verified text headers).

## 5. Scope Boundary Conformance
In strict adherence to rule **R2** and **R8**:
- Zero backend application code written.
- Zero frontend application code written.
- Zero agent code written.
- Zero raw data files modified or moved.
- Zero notebook files modified.
