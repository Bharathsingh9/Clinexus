#!/usr/bin/env python3
"""
scripts/audit_notebooks.py
Extracts dataset paths, preprocessing steps, models, printed metrics,
and scans output cells for patient identifiers across notebook/*.ipynb.
"""

import sys
import codecs
import json
import glob
import os
import re

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import os
import re
from check_dataset_pii import scan_text

notebook_files = sorted(glob.glob("notebook/*.ipynb"))

for nb_file in notebook_files:
    print(f"\n=================== {nb_file} ===================")
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    print(f"Total cells: {len(nb.get('cells', []))}, Code cells: {len(code_cells)}")
    
    paths_read = set()
    models_used = set()
    preprocessing_steps = []
    printed_metrics = []
    pii_violations = []
    
    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        
        # Paths read
        for m in re.finditer(r"(?:read_csv|read_parquet|open)\s*\(\s*['\"]([^'\"]+)['\"]", src):
            paths_read.add(m.group(1))
        for m in re.finditer(r"([a-zA-Z0-9_\-\./\\]+\.(?:csv|json|jsonl|parquet))", src):
            if not m.group(1).startswith("http"):
                paths_read.add(m.group(1))
                
        # Models
        for m in re.finditer(r"SentenceTransformer\s*\(\s*['\"]([^'\"]+)['\"]", src):
            models_used.add(m.group(1))
        for m in re.finditer(r"(?:BioBERT|pubmedbert|text-embedding|gemini|deepseek|qwen|gpt-4)[a-zA-Z0-9_\-\./]*", src, re.I):
            models_used.add(m.group(0))
            
        # Preprocessing hints
        for line in src.splitlines():
            line_str = line.strip()
            if any(k in line_str.lower() for k in ["normalize", "clean", "drop", "dedup", "split", "chunk", "faiss"]):
                if not line_str.startswith("#"):
                    preprocessing_steps.append(f"Cell {idx}: {line_str[:120]}")
                    
        # Outputs
        for out in cell.get("outputs", []):
            out_texts = []
            if "text" in out:
                out_texts.extend(out["text"])
            if "data" in out and "text/plain" in out["data"]:
                out_texts.extend(out["data"]["text/plain"])
                
            for ot in out_texts:
                # check pii
                f_res = scan_text(ot)
                if any(f_res[k] for k in ["ssn", "dob", "mrn", "patient_names"]):
                    pii_violations.append((idx, ot[:100], {k: f_res[k] for k in f_res if f_res[k]}))
                    
                # printed metrics
                for line in ot.splitlines():
                    if any(m in line.lower() for m in ["recall", "precision", "mrr", "accuracy", "f1", "map", "score", "ndcg", "loss"]):
                        printed_metrics.append(line.strip()[:140])
                        
    print("Paths Read:")
    for p in sorted(paths_read):
        print(f"  - {p}")
        
    print("Models Used:")
    for m in sorted(models_used):
        print(f"  - {m}")
        
    print(f"Preprocessing steps found: {len(preprocessing_steps)}")
    for ps in preprocessing_steps[:8]:
        print(f"  - {ps}")
    if len(preprocessing_steps) > 8:
        print(f"  ... and {len(preprocessing_steps)-8} more")
        
    print(f"Printed Metrics lines: {len(printed_metrics)}")
    for pm in printed_metrics[:10]:
        print(f"  - {pm}")
    if len(printed_metrics) > 10:
        print(f"  ... and {len(printed_metrics)-10} more")
        
    print(f"PII Violations in outputs: {len(pii_violations)}")
    for pv in pii_violations:
        print(f"  - Cell {pv[0]}: {pv[1]} -> {pv[2]}")
