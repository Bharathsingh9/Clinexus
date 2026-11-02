"""Deterministic Evaluation Split and Sample Generation for BioASQ Questions.

Specification Section 9.6, Appendix A11, A12 (Req ID: EVL-3, DAT-6):
1. Eligible questions: exactly those with corpus rows (5,665 questions). Excludes questions with 0 corpus rows.
2. The 30 smoke questions (retrieval_eval_candidates.csv) are pre-assigned to dev in file order.
3. Remaining questions are assigned to dev or eval using random.Random(42) on lexicographically sorted IDs,
   stratified by question_type to achieve ~20% dev per type overall.
4. Generates datasets/processed/splits_v1.json (IDs only, seed, method, SHA-256 checksum, counts).
5. Sample builders:
   - smoke30: 30 candidate IDs in file order.
   - e2e_sample: drawn strictly from eval, stratified proportionally by type, size from EVAL_SAMPLE_SIZE setting.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.app.core.config import settings


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for checksum calculation: {filepath}")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_smoke30_candidate_ids(
    candidates_csv_path: str,
    expected_count: Optional[int] = 30,
) -> List[str]:
    """Read the smoke candidate question IDs in exact file order."""
    if not os.path.exists(candidates_csv_path):
        raise FileNotFoundError(f"Candidates file not found: {candidates_csv_path}")

    candidate_ids: List[str] = []
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with open(candidates_csv_path, mode="r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sid = r.get("sample_id", "").strip()
                    if sid:
                        candidate_ids.append(sid)
            break
        except UnicodeDecodeError:
            continue

    if expected_count is not None and len(candidate_ids) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} candidate IDs, found {len(candidate_ids)}")
    return candidate_ids


def create_deterministic_splits(
    questions_csv_path: str,
    corpus_csv_path: str,
    candidates_csv_path: str,
    seed: int = 42,
    expected_smoke_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Create deterministic 20% dev / 80% eval split stratified by question_type.
    
    Invariants (SPEC 9.6):
    - Questions with 0 corpus rows are excluded.
    - smoke30 candidates are pre-assigned to dev.
    - Remaining question IDs are sorted lexicographically before random.Random(seed) stratification.
    - No question ID exists in both dev and eval.
    - Exactly deterministic given the same inputs and seed.
    """
    source_checksum = compute_file_sha256(questions_csv_path)
    smoke30_ids = load_smoke30_candidate_ids(candidates_csv_path, expected_count=expected_smoke_count)
    smoke30_set = set(smoke30_ids)

    # 1. Determine eligible sample_ids from corpus
    eligible_samples: Set[str] = set()
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with open(corpus_csv_path, mode="r", encoding=encoding, newline="") as f:
                for r in csv.DictReader(f):
                    sid = r.get("sample_id", "").strip()
                    if sid:
                        eligible_samples.add(sid)
            break
        except UnicodeDecodeError:
            continue

    # 2. Group eligible questions by type
    questions_by_type: Dict[str, List[str]] = defaultdict(list)
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            with open(questions_csv_path, mode="r", encoding=encoding, newline="") as f:
                for r in csv.DictReader(f):
                    sid = r.get("sample_id", "").strip()
                    qtype = r.get("question_type", "").strip()
                    if sid in eligible_samples and qtype:
                        questions_by_type[qtype].append(sid)
            break
        except UnicodeDecodeError:
            continue

    total_eligible = sum(len(ids) for ids in questions_by_type.values())
    if total_eligible != len(eligible_samples):
        raise ValueError(
            f"Eligible questions count mismatch: {total_eligible} mapped vs {len(eligible_samples)} corpus samples"
        )

    # 3. Stratify per question_type
    rng = random.Random(seed)
    dev_by_type: Dict[str, List[str]] = {}
    eval_by_type: Dict[str, List[str]] = {}
    smoke_counts_by_type: Counter = Counter()

    for qtype in sorted(questions_by_type.keys()):
        all_type_ids = questions_by_type[qtype]
        all_type_set = set(all_type_ids)

        # Smoke candidates belonging to this question type
        smoke_for_type = [sid for sid in smoke30_ids if sid in all_type_set]
        smoke_counts_by_type[qtype] = len(smoke_for_type)

        # Target dev size = 20%
        target_dev = round(len(all_type_ids) * 0.20)
        needed_from_rem = target_dev - len(smoke_for_type)
        if needed_from_rem < 0:
            raise ValueError(f"Smoke questions exceed 20% target for question type: {qtype}")

        # Remaining question IDs sorted lexicographically
        rem_ids = sorted([sid for sid in all_type_ids if sid not in smoke30_set])
        shuffled_rem = list(rem_ids)
        rng.shuffle(shuffled_rem)

        dev_rem = shuffled_rem[:needed_from_rem]
        eval_rem = shuffled_rem[needed_from_rem:]

        dev_by_type[qtype] = smoke_for_type + sorted(dev_rem)
        eval_by_type[qtype] = sorted(eval_rem)

    # Aggregate and sort split lists
    dev_ids: List[str] = []
    eval_ids: List[str] = []
    dev_counts: Dict[str, int] = {}
    eval_counts: Dict[str, int] = {}

    for qtype in sorted(questions_by_type.keys()):
        dev_ids.extend(dev_by_type[qtype])
        eval_ids.extend(eval_by_type[qtype])
        dev_counts[qtype] = len(dev_by_type[qtype])
        eval_counts[qtype] = len(eval_by_type[qtype])

    # Invariant checks
    dev_set = set(dev_ids)
    eval_set = set(eval_ids)
    assert len(dev_set.intersection(eval_set)) == 0, "Leakage invariant violated: ID in both dev and eval"
    assert smoke30_set.issubset(dev_set), "Invariant violated: Not all smoke30 candidates assigned to dev"
    assert len(dev_ids) + len(eval_ids) == total_eligible, "Total split IDs must equal eligible count"

    return {
        "dataset": "bioasq_normalized",
        "source_checksum": source_checksum,
        "seed": seed,
        "method_description": (
            "Deterministic 20% dev / 80% eval split stratified by question_type using random.Random(42) "
            "on lexicographically sorted question IDs, with all 30 candidate smoke questions pre-assigned to dev."
        ),
        "total_eligible_questions": total_eligible,
        "total_dev": len(dev_ids),
        "total_eval": len(eval_ids),
        "smoke30_pre_assigned_count": len(smoke30_ids),
        "smoke30_counts_by_type": dict(smoke_counts_by_type),
        "counts_by_type": {
            "dev": dev_counts,
            "eval": eval_counts,
            "total": {qt: len(questions_by_type[qt]) for qt in questions_by_type},
        },
        "dev_ids": dev_ids,
        "eval_ids": eval_ids,
        "eval_by_type": eval_by_type,
    }


def build_smoke30_sample(
    candidates_csv_path: str,
    expected_count: Optional[int] = 30,
) -> List[str]:
    """Return the smoke candidate IDs in file order."""
    return load_smoke30_candidate_ids(candidates_csv_path, expected_count=expected_count)


def build_e2e_sample(
    eval_by_type: Dict[str, List[str]],
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Draw a deterministic sample strictly from the eval split, stratified proportionally by type.
    
    Size is taken from EVAL_SAMPLE_SIZE setting (default 200).
    Uses the largest remainder method to ensure exact integer apportionment across question types.
    """
    if sample_size is None:
        sample_size = settings.EVAL_SAMPLE_SIZE
    if sample_size <= 0:
        raise ValueError(f"EVAL_SAMPLE_SIZE must be > 0, got {sample_size}")

    total_eval = sum(len(ids) for ids in eval_by_type.values())
    if sample_size > total_eval:
        raise ValueError(f"Sample size {sample_size} cannot exceed total eval count {total_eval}")

    # Largest Remainder Method (Hamilton method)
    exact_shares = {qt: (len(ids) / total_eval) * sample_size for qt, ids in eval_by_type.items()}
    allocated = {qt: int(v) for qt, v in exact_shares.items()}
    remainders = {qt: exact_shares[qt] - allocated[qt] for qt in exact_shares}

    extra_needed = sample_size - sum(allocated.values())
    # Sort types by remainder descending, breaking ties by type name for determinism
    sorted_by_remainder = sorted(remainders.keys(), key=lambda qt: (remainders[qt], qt), reverse=True)
    for i in range(extra_needed):
        allocated[sorted_by_remainder[i]] += 1

    assert sum(allocated.values()) == sample_size, "Allocated sample size must equal target"

    rng = random.Random(seed)
    sampled_ids: List[str] = []
    for qt in sorted(eval_by_type.keys()):
        available = sorted(eval_by_type[qt])
        count = allocated[qt]
        shuffled = list(available)
        rng.shuffle(shuffled)
        sampled_ids.extend(sorted(shuffled[:count]))

    return sampled_ids


def write_splits_file(
    output_path: str,
    split_data: Dict[str, Any],
) -> str:
    """Serialize split to JSON with IDs only (no question text)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_payload = {
        "dataset": split_data["dataset"],
        "source_checksum": split_data["source_checksum"],
        "seed": split_data["seed"],
        "method_description": split_data["method_description"],
        "total_eligible_questions": split_data["total_eligible_questions"],
        "total_dev": split_data["total_dev"],
        "total_eval": split_data["total_eval"],
        "counts_by_type": split_data["counts_by_type"],
        "smoke30_counts_by_type": split_data["smoke30_counts_by_type"],
        "dev_ids": split_data["dev_ids"],
        "eval_ids": split_data["eval_ids"],
    }
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(clean_payload, f, indent=2)
    return output_path
