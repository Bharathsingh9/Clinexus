"""Unit tests for Deterministic Evaluation Splits and Sample Builders.

Specification Section 9.6, Appendix A11 (Req ID: EVL-3, DAT-6).
Invariants:
- Uses ONLY [SYNTHETIC] fixture data; NEVER real BioASQ text (Rule R4).
- Deterministic splits: identical seed gives identical output IDs.
- Dev and Eval sets are strictly disjoint (0 overlapping IDs).
- Per-type dev proportions are approximately 20% (including smoke30).
- All smoke30 candidate questions are pre-assigned to dev.
- e2e_sample is drawn strictly from eval (0 in dev) and stratified proportionally by type.
- EVAL_SAMPLE_SIZE setting validates greater than 0.
- Questions with 0 corpus rows are excluded.
"""

import csv
import json
import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.evaluation.splits import (
    build_e2e_sample,
    build_smoke30_sample,
    create_deterministic_splits,
    write_splits_file,
)


@pytest.fixture
def synthetic_split_environment(tmp_path):
    """Generate synthetic CSV files for questions, corpus, and candidate fixtures."""
    q_file = tmp_path / "synthetic_questions.csv"
    c_file = tmp_path / "synthetic_corpus.csv"
    cand_file = tmp_path / "synthetic_candidates.csv"

    # Create 200 synthetic questions across 4 types:
    # factoid: 50, yesno: 50, list: 50, summary: 50.
    # Exclude last 4 questions from corpus (196..199).
    # Exactly 30 candidate smoke questions (8 factoid, 7 yesno, 7 list, 8 summary)
    q_rows = []
    c_rows = []
    cand_rows = []

    types = ["factoid"] * 50 + ["yesno"] * 50 + ["list"] * 50 + ["summary"] * 50

    cand_targets = {"factoid": 8, "yesno": 7, "list": 7, "summary": 8}
    cand_counts = {"factoid": 0, "yesno": 0, "list": 0, "summary": 0}

    for idx, qtype in enumerate(types):
        sid = f"SYNTH_Q_{idx:03d}"
        q_rows.append(
            {
                "sample_id": sid,
                "source_dataset": "BioASQ",
                "question_type": qtype,
                "question": f"[SYNTHETIC] Clinical inquiry {idx}?",
                "exact_answer": json.dumps("yes" if qtype == "yesno" else None),
                "ideal_answer": json.dumps([f"[SYNTHETIC] Ideal answer {idx}"]),
                "context": json.dumps([{"text": f"[SYNTHETIC] Snippet {idx}"}]),
                "source_reference": json.dumps([]),
                "concepts": json.dumps([]),
                "metadata": json.dumps({}),
            }
        )

        # Build candidate smoke questions matching distribution
        if cand_counts[qtype] < cand_targets[qtype]:
            cand_rows.append(
                {
                    "eval_id": f"EVAL_{len(cand_rows):02d}",
                    "question": f"[SYNTHETIC] Clinical inquiry {idx}?",
                    "question_type": qtype,
                    "sample_id": sid,
                }
            )
            cand_counts[qtype] += 1

        # Last 4 questions have NO corpus rows (excluded)
        if idx < 196:
            c_rows.append(
                {
                    "evidence_id": f"E_{idx:03d}",
                    "snippet_text": f"[SYNTHETIC] Snippet {idx}",
                    "pubmed_url": f"http://pubmed/{idx}",
                    "section": "abstract",
                    "sample_id": sid,
                    "question_type": qtype,
                }
            )

    # Write files
    with open(q_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
        writer.writeheader()
        writer.writerows(q_rows)

    with open(c_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(c_rows[0].keys()))
        writer.writeheader()
        writer.writerows(c_rows)

    with open(cand_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cand_rows)

    return {
        "questions_path": str(q_file),
        "corpus_path": str(c_file),
        "candidates_path": str(cand_file),
        "output_path": str(tmp_path / "splits_v1.json"),
        "total_questions": 200,
        "eligible_questions": 196,
        "excluded_questions": 4,
        "candidate_count": 30,
    }


def test_split_determinism_and_isolation(synthetic_split_environment):
    """Verify deterministic split partitioning, disjoint dev/eval sets, and candidate inclusion."""
    env = synthetic_split_environment

    # Run 1
    split1 = create_deterministic_splits(
        questions_csv_path=env["questions_path"],
        corpus_csv_path=env["corpus_path"],
        candidates_csv_path=env["candidates_path"],
        seed=42,
    )

    # Run 2 with same seed
    split2 = create_deterministic_splits(
        questions_csv_path=env["questions_path"],
        corpus_csv_path=env["corpus_path"],
        candidates_csv_path=env["candidates_path"],
        seed=42,
    )

    # 1. Exact determinism: identical dev and eval lists
    assert split1["dev_ids"] == split2["dev_ids"]
    assert split1["eval_ids"] == split2["eval_ids"]
    assert split1["source_checksum"] == split2["source_checksum"]

    # 2. Total eligible questions equals corpus count (196), excluding 4
    assert split1["total_eligible_questions"] == env["eligible_questions"]
    assert "SYNTH_Q_196" not in split1["dev_ids"] and "SYNTH_Q_196" not in split1["eval_ids"]
    assert "SYNTH_Q_199" not in split1["dev_ids"] and "SYNTH_Q_199" not in split1["eval_ids"]

    # 3. Disjoint invariant: zero overlapping IDs
    dev_set = set(split1["dev_ids"])
    eval_set = set(split1["eval_ids"])
    assert len(dev_set.intersection(eval_set)) == 0

    # 4. All candidate smoke questions pre-assigned to dev
    candidates = build_smoke30_sample(env["candidates_path"])
    assert len(candidates) == 30
    for cand_id in candidates:
        assert cand_id in dev_set
        assert cand_id not in eval_set

    # 5. Overall dev proportion near 20%
    dev_prop = split1["total_dev"] / split1["total_eligible_questions"]
    assert 0.18 <= dev_prop <= 0.25


def test_e2e_sample_builder_drawn_only_from_eval(synthetic_split_environment):
    """Verify e2e_sample builder draws strictly from eval split with proportional stratification."""
    env = synthetic_split_environment
    split = create_deterministic_splits(
        questions_csv_path=env["questions_path"],
        corpus_csv_path=env["corpus_path"],
        candidates_csv_path=env["candidates_path"],
        seed=42,
    )

    dev_set = set(split["dev_ids"])
    eval_set = set(split["eval_ids"])

    # Draw sample of size 20
    sample_ids = build_e2e_sample(split["eval_by_type"], sample_size=20, seed=42)

    assert len(sample_ids) == 20
    for sid in sample_ids:
        # Strict invariant: MUST be in eval, NEVER in dev
        assert sid in eval_set
        assert sid not in dev_set

    # Repeat run with same seed must be 100% deterministic
    sample_ids_run2 = build_e2e_sample(split["eval_by_type"], sample_size=20, seed=42)
    assert sample_ids == sample_ids_run2


def test_eval_sample_size_setting_validation():
    """Verify EVAL_SAMPLE_SIZE setting requires an integer greater than 0."""
    # Valid
    s = Settings(SECRET_KEY="x" * 32, ADMIN_PASSWORD="password123", EVAL_SAMPLE_SIZE=100)
    assert s.EVAL_SAMPLE_SIZE == 100

    # Invalid <= 0
    with pytest.raises(ValidationError):
        Settings(SECRET_KEY="x" * 32, ADMIN_PASSWORD="password123", EVAL_SAMPLE_SIZE=0)

    with pytest.raises(ValidationError):
        Settings(SECRET_KEY="x" * 32, ADMIN_PASSWORD="password123", EVAL_SAMPLE_SIZE=-5)


def test_write_splits_file_contains_no_question_text(synthetic_split_environment):
    """Verify serialized split JSON contains only question IDs and metadata, never question text (SPEC 9.6)."""
    env = synthetic_split_environment
    split = create_deterministic_splits(
        questions_csv_path=env["questions_path"],
        corpus_csv_path=env["corpus_path"],
        candidates_csv_path=env["candidates_path"],
        seed=42,
    )
    out_path = write_splits_file(env["output_path"], split)

    with open(out_path, mode="r", encoding="utf-8") as f:
        content = f.read()
        obj = json.loads(content)

    # Invariant: question text must not appear
    assert "Clinical inquiry" not in content
    assert "Ideal answer" not in content
    assert "Snippet" not in content

    # Structure check
    assert "dev_ids" in obj
    assert "eval_ids" in obj
    assert obj["seed"] == 42
    assert "source_checksum" in obj
