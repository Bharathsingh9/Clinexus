#!/usr/bin/env python3
"""
scripts/check_dataset_pii.py
Requirement: DAT-2, SAF-0

Scans dataset texts for:
1. Author contact details (email addresses classified as AUTHOR_CONTACT - permitted in raw text, never logged).
2. Phone-like patterns.
3. Patient identifier patterns (dates of birth, medical record numbers, SSNs, clinical patient names).

Usage:
    python scripts/check_dataset_pii.py [path_or_glob]
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Regex definitions per SPEC section 9.3 and 9.11
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)

# Standard phone numbers with delimiters (e.g. 555-123-4567, (555) 123-4567, +1-555-123-4567)
PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)

SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

DOB_PATTERN = re.compile(
    r"\b(?:dob|date\s+of\s+birth|d\.o\.b\.|born\s+on)\s*[:=-]\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b",
    re.IGNORECASE,
)

MRN_PATTERN = re.compile(
    r"\b(?:mrn|medical\s+record\s+number|record\s*#|chart\s*#|patient\s+id)\s*[:=-]\s*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)

PATIENT_NAME_CLINICAL_PATTERN = re.compile(
    r"\b(?:patient|pt\.?)\s+(?:name\s*[:=-]\s*)?[A-Z][a-z]+,\s+[A-Z][a-z]+\b"
)


def scan_text(text: str) -> Dict[str, List[str]]:
    """Scan a single text string for PII and contact patterns."""
    findings: Dict[str, List[str]] = {
        "author_contact_emails": [],
        "phone_numbers": [],
        "ssn": [],
        "dob": [],
        "mrn": [],
        "patient_names": [],
    }
    if not text or not isinstance(text, str):
        return findings

    emails = EMAIL_PATTERN.findall(text)
    if emails:
        findings["author_contact_emails"].extend(emails)

    phones = PHONE_PATTERN.findall(text)
    if phones:
        findings["phone_numbers"].extend(phones)

    ssns = SSN_PATTERN.findall(text)
    if ssns:
        findings["ssn"].extend(ssns)

    dobs = DOB_PATTERN.findall(text)
    if dobs:
        findings["dob"].extend(dobs)

    mrns = MRN_PATTERN.findall(text)
    if mrns:
        findings["mrn"].extend(mrns)

    pt_names = PATIENT_NAME_CLINICAL_PATTERN.findall(text)
    if pt_names:
        findings["patient_names"].extend(pt_names)

    return findings


def scan_file(file_path: str) -> Dict[str, Any]:
    """Scan a file for PII based on file type."""
    ext = os.path.splitext(file_path)[1].lower()
    total_records = 0
    records_with_email = 0
    records_with_patient_pii = 0
    counts: Dict[str, int] = {
        "author_contact_emails": 0,
        "phone_numbers": 0,
        "ssn": 0,
        "dob": 0,
        "mrn": 0,
        "patient_names": 0,
    }

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")
        total_records = len(df)
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ["text", "snippet", "context", "question", "answer"])]
        if not text_cols:
            text_cols = list(df.columns)
        for _, row in df.iterrows():
            row_has_email = False
            row_has_pii = False
            for col in text_cols:
                val = str(row[col])
                f = scan_text(val)
                for k in counts:
                    if f[k]:
                        counts[k] += len(f[k])
                        if k == "author_contact_emails":
                            row_has_email = True
                        else:
                            row_has_pii = True
            if row_has_email:
                records_with_email += 1
            if row_has_pii:
                records_with_patient_pii += 1

    elif ext in [".json", ".jsonl"]:
        if ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    total_records += 1
                    f = scan_text(line)
                    has_email = bool(f["author_contact_emails"])
                    has_pii = any(bool(f[k]) for k in counts if k != "author_contact_emails")
                    for k in counts:
                        counts[k] += len(f[k])
                    if has_email:
                        records_with_email += 1
                    if has_pii:
                        records_with_patient_pii += 1
        else:
            with open(file_path, "r", encoding="utf-8") as fp:
                content = json.load(fp)
            if isinstance(content, list):
                total_records = len(content)
                for item in content:
                    text_str = json.dumps(item)
                    f = scan_text(text_str)
                    has_email = bool(f["author_contact_emails"])
                    has_pii = any(bool(f[k]) for k in counts if k != "author_contact_emails")
                    for k in counts:
                        counts[k] += len(f[k])
                    if has_email:
                        records_with_email += 1
                    if has_pii:
                        records_with_patient_pii += 1
            elif isinstance(content, dict):
                # e.g. training14b or ori_pqal
                if "questions" in content:
                    items = content["questions"]
                else:
                    items = list(content.values())
                total_records = len(items)
                for item in items:
                    text_str = json.dumps(item)
                    f = scan_text(text_str)
                    has_email = bool(f["author_contact_emails"])
                    has_pii = any(bool(f[k]) for k in counts if k != "author_contact_emails")
                    for k in counts:
                        counts[k] += len(f[k])
                    if has_email:
                        records_with_email += 1
                    if has_pii:
                        records_with_patient_pii += 1

    elif ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(file_path)
        total_records = len(df)
        for _, row in df.iterrows():
            row_str = " ".join(str(v) for v in row.values)
            f = scan_text(row_str)
            has_email = bool(f["author_contact_emails"])
            has_pii = any(bool(f[k]) for k in counts if k != "author_contact_emails")
            for k in counts:
                counts[k] += len(f[k])
            if has_email:
                records_with_email += 1
            if has_pii:
                records_with_patient_pii += 1
    else:
        # text files like README
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read()
        total_records = 1
        f = scan_text(text)
        for k in counts:
            counts[k] += len(f[k])
        if f["author_contact_emails"]:
            records_with_email = 1
        if any(f[k] for k in counts if k != "author_contact_emails"):
            records_with_patient_pii = 1

    return {
        "file_path": file_path,
        "total_records": total_records,
        "records_with_email": records_with_email,
        "records_with_patient_pii": records_with_patient_pii,
        "counts": counts,
        "status": "PASS" if records_with_patient_pii == 0 else "FAIL_PATIENT_PII_DETECTED",
    }


def main():
    parser = argparse.ArgumentParser(description="Check dataset files for PII and contact info.")
    parser.add_argument("target", nargs="?", default="datasets/**/*.*", help="File or glob pattern to scan")
    args = parser.parse_args()

    files = [f for f in glob.glob(args.target, recursive=True) if os.path.isfile(f)]
    if not files:
        print(f"No files found matching: {args.target}")
        sys.exit(0)

    any_pii_failure = False
    print(f"Scanning {len(files)} files for PII and author contact details...\n")
    for f in sorted(files):
        res = scan_file(f)
        status_msg = res["status"]
        email_cnt = res["counts"]["author_contact_emails"]
        phone_cnt = res["counts"]["phone_numbers"]
        pii_cnt = sum(res["counts"][k] for k in ["ssn", "dob", "mrn", "patient_names"])
        print(f"File: {f}")
        print(f"  Records: {res['total_records']}, Status: {status_msg}")
        print(f"  Author Emails: {email_cnt} in {res['records_with_email']} records")
        print(f"  Phones: {phone_cnt}, Patient Identifiers: {pii_cnt}")
        if res["status"] != "PASS":
            any_pii_failure = True
            print("  [CRITICAL ALERT] Patient PII detected in dataset! Cease ingestion per SPEC 9.3.")
        print()

    if any_pii_failure:
        print("PII scan complete: VIOLATIONS DETECTED.")
        sys.exit(1)
    else:
        print("PII scan complete: All datasets PASS (0 patient identifiers found).")
        sys.exit(0)


if __name__ == "__main__":
    main()
