#!/usr/bin/env python3
"""
scripts/audit_datasets_comprehensive.py
Full reproducibility audit for SPEC Section 9 and repository data files.
"""

import hashlib
import json
import os
import re
import pandas as pd
from typing import Dict, Any, List

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit_bioasq_corpus():
    path = "datasets/processed/bioasq/bioasq_evidence_corpus.csv"
    print("=== Auditing BioASQ Evidence Corpus ===")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    row_count = len(df)
    columns = list(df.columns)
    null_counts = {col: (df[col] == "").sum() for col in columns}
    
    # Section counts
    section_counts = df["section"].value_counts().to_dict()
    
    # Text lengths
    lens = df["snippet_text"].str.len()
    median_len = lens.median()
    max_len = lens.max()
    min_len = lens.min()
    under_50 = (lens < 50).sum()
    
    # Duplicates
    dup_texts = df["snippet_text"].duplicated(keep=False).sum()
    # Unique texts duplicated
    unique_dups = df[df["snippet_text"].duplicated()]["snippet_text"].nunique()
    # Shared across different questions
    text_q_counts = df.groupby("snippet_text")["sample_id"].nunique()
    shared_questions = (text_q_counts > 1).sum()
    
    # Distinct PubMed URLs
    distinct_urls = df["pubmed_url"].nunique()
    
    # URL patterns and PMID extraction
    pmid_re = re.compile(r"(?:^|/)(\d{1,9})/?$")
    url_deviations = {
        "https_variants": 0,
        "bare_digits": 0,
        "https_pubmed_ncbi_slash": 0,
        "trailing_slash": 0,
        "other": 0,
        "no_match": 0
    }
    
    for url in df["pubmed_url"]:
        m = pmid_re.search(url)
        if not m:
            url_deviations["no_match"] += 1
        else:
            if url.isdigit():
                url_deviations["bare_digits"] += 1
            elif url.startswith("https://pubmed.ncbi.nlm.nih.gov/") and url.endswith("/"):
                url_deviations["https_pubmed_ncbi_slash"] += 1
            elif url.startswith("https://"):
                url_deviations["https_variants"] += 1
            elif url.endswith("/") and not url.isdigit():
                url_deviations["trailing_slash"] += 1
                
    # Distinct sample_ids
    distinct_sample_ids = df["sample_id"].nunique()
    
    return {
        "row_count": row_count,
        "columns": columns,
        "null_counts": null_counts,
        "section_counts": section_counts,
        "median_len": median_len,
        "max_len": max_len,
        "min_len": min_len,
        "under_50": under_50,
        "duplicate_text_rows": int(dup_texts),
        "unique_duplicate_texts": int(unique_dups),
        "texts_shared_across_questions": int(shared_questions),
        "distinct_urls": distinct_urls,
        "distinct_sample_ids": distinct_sample_ids,
        "url_deviations": url_deviations
    }

def audit_bioasq_normalized():
    path = "datasets/processed/bioasq/bioasq_normalized.csv"
    print("=== Auditing BioASQ Normalized Questions ===")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    row_count = len(df)
    columns = list(df.columns)
    null_counts = {col: (df[col] == "").sum() for col in columns}
    
    q_type_counts = df["question_type"].value_counts().to_dict()
    unique_sample_id = df["sample_id"].nunique()
    
    # Normalized question text dups
    def norm_q(q):
        return re.sub(r"\s+", " ", q.strip().lower())
    norm_questions = df["question"].apply(norm_q)
    dup_norm_questions = norm_questions.duplicated().sum()
    
    # JSON parsing and shape check
    json_cols = ["exact_answer", "ideal_answer", "context", "source_reference", "concepts", "metadata"]
    json_parse_failures = {c: 0 for c in json_cols}
    exact_shapes = {"yesno_yes": 0, "yesno_no": 0, "factoid_flat_list": 0, "list_nested_list": 0, "summary_null": 0, "unknown": 0}
    
    ideal_counts = {"count_1": 0, "max_strings": 0, "over_20k_chars": 0}
    context_stats = {"total_snippets": 0, "min": 9999, "max": 0, "snippets_per_q": [], "snippets_with_email": 0}
    source_ref_urls = 0
    all_source_refs = set()
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    
    for _, row in df.iterrows():
        qtype = row["question_type"]
        for col in json_cols:
            val = row[col]
            try:
                parsed = json.loads(val)
            except Exception:
                json_parse_failures[col] += 1
                continue
                
            if col == "exact_answer":
                if qtype == "yesno":
                    if parsed == "yes":
                        exact_shapes["yesno_yes"] += 1
                    elif parsed == "no":
                        exact_shapes["yesno_no"] += 1
                    else:
                        exact_shapes["unknown"] += 1
                elif qtype == "factoid":
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        exact_shapes["factoid_flat_list"] += 1
                    else:
                        exact_shapes["unknown"] += 1
                elif qtype == "list":
                    if isinstance(parsed, list) and all(isinstance(x, list) for x in parsed):
                        exact_shapes["list_nested_list"] += 1
                    else:
                        exact_shapes["unknown"] += 1
                elif qtype == "summary":
                    if parsed is None:
                        exact_shapes["summary_null"] += 1
                    else:
                        exact_shapes["unknown"] += 1
                        
            elif col == "ideal_answer":
                if isinstance(parsed, list):
                    if len(parsed) == 1:
                        ideal_counts["count_1"] += 1
                    if len(parsed) > ideal_counts["max_strings"]:
                        ideal_counts["max_strings"] = len(parsed)
                    total_chars = sum(len(s) for s in parsed if isinstance(s, str))
                    if total_chars > 20000:
                        ideal_counts["over_20k_chars"] += 1
                        
            elif col == "context":
                if isinstance(parsed, list):
                    n_snip = len(parsed)
                    context_stats["total_snippets"] += n_snip
                    context_stats["snippets_per_q"].append(n_snip)
                    if n_snip > context_stats["max"]:
                        context_stats["max"] = n_snip
                    if n_snip < context_stats["min"]:
                        context_stats["min"] = n_snip
                    for snip in parsed:
                        if isinstance(snip, dict) and "text" in snip:
                            if email_re.search(snip["text"]):
                                context_stats["snippets_with_email"] += 1
                                
            elif col == "source_reference":
                if isinstance(parsed, list):
                    source_ref_urls += len(parsed)
                    all_source_refs.update(parsed)

    med_snippets = pd.Series(context_stats["snippets_per_q"]).median()
    
    return {
        "row_count": row_count,
        "columns": columns,
        "null_counts": null_counts,
        "q_type_counts": q_type_counts,
        "unique_sample_id": unique_sample_id,
        "dup_norm_questions": int(dup_norm_questions),
        "json_parse_failures": json_parse_failures,
        "exact_shapes": exact_shapes,
        "ideal_counts": ideal_counts,
        "context_stats": {
            "total_snippets": context_stats["total_snippets"],
            "median_per_q": med_snippets,
            "max_per_q": context_stats["max"],
            "min_per_q": context_stats["min"],
            "snippets_with_email": context_stats["snippets_with_email"]
        },
        "source_reference_urls": source_ref_urls,
        "distinct_source_reference_urls": len(all_source_refs)
    }

def audit_linkage():
    print("=== Auditing Corpus-Questions Linkage ===")
    corpus_df = pd.read_csv("datasets/processed/bioasq/bioasq_evidence_corpus.csv", dtype=str, keep_default_na=False)
    q_df = pd.read_csv("datasets/processed/bioasq/bioasq_normalized.csv", dtype=str, keep_default_na=False)
    
    corpus_sample_ids = set(corpus_df["sample_id"].unique())
    q_sample_ids = set(q_df["sample_id"].unique())
    
    all_corpus_ids_in_q = corpus_sample_ids.issubset(q_sample_ids)
    missing_in_corpus = q_sample_ids - corpus_sample_ids
    
    # Missing questions by type
    missing_by_type = q_df[q_df["sample_id"].isin(missing_in_corpus)]["question_type"].value_counts().to_dict()
    
    # Missing snippets count
    missing_snippets_count = 0
    for _, row in q_df[q_df["sample_id"].isin(missing_in_corpus)].iterrows():
        ctx = json.loads(row["context"])
        missing_snippets_count += len(ctx)
        
    print(f"Corpus sample_ids: {len(corpus_sample_ids)}")
    print(f"All corpus IDs in questions: {all_corpus_ids_in_q}")
    print(f"Questions missing in corpus: {len(missing_in_corpus)}, by type: {missing_by_type}, snippets: {missing_snippets_count}")
    
    return {
        "n_corpus_sample_ids": len(corpus_sample_ids),
        "all_in_questions": all_corpus_ids_in_q,
        "n_questions_missing_corpus": len(missing_in_corpus),
        "missing_by_type": missing_by_type,
        "missing_context_snippets": missing_snippets_count
    }

if __name__ == "__main__":
    def json_default(obj):
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        return str(obj)

    res_corpus = audit_bioasq_corpus()
    print("Corpus Results:", json.dumps(res_corpus, indent=2, default=json_default))
    res_q = audit_bioasq_normalized()
    print("Questions Results:", json.dumps(res_q, indent=2, default=json_default))
    res_link = audit_linkage()
    print("Linkage Results:", json.dumps(res_link, indent=2, default=json_default))
