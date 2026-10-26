#!/usr/bin/env python3
"""
scripts/audit_other_datasets.py
Audits MedQA, MedBullets, PubMedQA files.
"""

import hashlib
import json
import os
import re
import pandas as pd
from check_dataset_pii import scan_file

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

files_to_audit = [
    # MedQA
    "datasets/medqa/US/dev.jsonl",
    "datasets/medqa/US/test.jsonl",
    "datasets/medqa/US/train.jsonl",
    "datasets/processed/medqa/medqa_normalized.csv",
    # MedBullets
    "datasets/medbullets/op4_eval-00000-of-00001.parquet",
    "datasets/medbullets/op4_test-00000-of-00001.parquet",
    "datasets/medbullets/op4_train-00000-of-00001.parquet",
    "datasets/medbullets/op5_eval-00000-of-00001.parquet",
    "datasets/medbullets/op5_test-00000-of-00001.parquet",
    "datasets/medbullets/op5_train-00000-of-00001.parquet",
    "datasets/processed/medbullets/medbullets_op4_normalized.csv",
    # PubMedQA
    "datasets/pubmedqa/ori_pqal.json",
    "datasets/processed/pubmedqa/pubmedqa_normalized.csv",
]

results = []

for path in files_to_audit:
    size = os.path.getsize(path)
    sha = sha256_file(path)
    pii = scan_file(path)
    ext = os.path.splitext(path)[1].lower()
    
    row_count = 0
    columns = []
    null_rates = {}
    duplicates = 0
    
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        row_count = len(df)
        columns = list(df.columns)
        null_rates = {c: float((df[c] == "").sum()) / row_count if row_count > 0 else 0.0 for c in columns}
        # Check dups on sample_id or first col
        id_col = "sample_id" if "sample_id" in df.columns else columns[0]
        duplicates = int(df[id_col].duplicated().sum())
    elif ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        row_count = len(rows)
        if rows:
            columns = list(rows[0].keys())
            # null rates
            for c in columns:
                n_null = sum(1 for r in rows if r.get(c) is None or r.get(c) == "")
                null_rates[c] = float(n_null) / row_count
            # dups
            q_texts = [r.get("question", "") for r in rows]
            duplicates = len(q_texts) - len(set(q_texts))
    elif ext == ".parquet":
        df = pd.read_parquet(path)
        row_count = len(df)
        columns = list(df.columns)
        null_rates = {c: float(df[c].isna().sum()) / row_count if row_count > 0 else 0.0 for c in columns}
        id_col = "idx" if "idx" in df.columns else columns[0]
        duplicates = int(df[id_col].duplicated().sum())
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            row_count = len(data)
            first_key = next(iter(data.keys()))
            first_val = data[first_key]
            if isinstance(first_val, dict):
                columns = ["key"] + list(first_val.keys())
                for c in first_val.keys():
                    n_null = sum(1 for v in data.values() if v.get(c) is None or v.get(c) == "")
                    null_rates[c] = float(n_null) / row_count
            else:
                columns = ["key", "value"]
            duplicates = 0
            
    res = {
        "path": path,
        "size": size,
        "sha256": sha,
        "row_count": row_count,
        "columns": columns,
        "null_rates": null_rates,
        "duplicates": duplicates,
        "pii_status": pii["status"],
        "records_with_email": pii["records_with_email"],
        "records_with_patient_pii": pii["records_with_patient_pii"],
    }
    results.append(res)
    print(f"Audited {path}: rows={row_count}, cols={len(columns)}, PII={pii['status']}")

with open("scripts/audit_other_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nAudit written to scripts/audit_other_results.json")
