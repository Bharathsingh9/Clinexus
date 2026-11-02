"""Unit tests for BioASQ Questions Adapter and Derived Relevance Labels.

Specification Section 9.4, 9.5, 9.7, and Appendix A11 (Req ID: DAT-4, DAT-5, DB-1).
Invariants:
- Uses ONLY [SYNTHETIC] fixture data; NEVER real BioASQ text (Rule R4).
- Each exact_answer shape parses cleanly; unknown shapes rejected with reasons.
- Invalid ideal_answer rejected with reasons.
- Questions with no corpus rows excluded and counted.
- Row-level relevance: a corpus row split into multiple chunks makes all its chunks relevant.
- Coverage and mismatch reporting verified on fixture where n_corpus_rows < n_context.
- Persistence into evaluation_labels is idempotent.
"""

import csv
import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.evaluation.bioasq_questions import (
    build_derived_relevance_mapping,
    compute_coverage_report,
    extract_pmids_from_source_reference,
    parse_bioasq_questions,
    persist_evaluation_labels,
    validate_exact_answer_shape,
    validate_ideal_answer,
)
from backend.app.persistence.database import Base
from backend.app.persistence.models import EvaluationLabel, EvidencePassage
from backend.app.schemas.evaluation import BioASQGoldLabel, WorkflowInput


@pytest.fixture
def synthetic_sqlite_db():
    """Create an in-memory SQLite database session for unit testing persistence."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_validate_exact_answer_shapes():
    """Verify each allowed exact_answer shape succeeds and invalid shapes are rejected."""
    # 1. yesno: string 'yes' or 'no'
    ok, err = validate_exact_answer_shape("yesno", "yes")
    assert ok is True and err is None
    ok, err = validate_exact_answer_shape("yesno", "no")
    assert ok is True and err is None
    ok, err = validate_exact_answer_shape("yesno", "maybe")
    assert ok is False and "must be string 'yes' or 'no'" in err
    ok, err = validate_exact_answer_shape("yesno", ["yes"])
    assert ok is False

    # 2. factoid: flat list of strings
    ok, err = validate_exact_answer_shape("factoid", ["[SYNTHETIC] Entity A", "[SYNTHETIC] Entity B"])
    assert ok is True and err is None
    ok, err = validate_exact_answer_shape("factoid", "[SYNTHETIC] String only")
    assert ok is False and "flat list of strings" in err
    ok, err = validate_exact_answer_shape("factoid", [["[SYNTHETIC] Nested"]])
    assert ok is False

    # 3. list: list of lists of strings
    ok, err = validate_exact_answer_shape("list", [["[SYNTHETIC] Ent1", "[SYNTHETIC] Syn1"], ["[SYNTHETIC] Ent2"]])
    assert ok is True and err is None
    ok, err = validate_exact_answer_shape("list", ["[SYNTHETIC] Flat string"])
    assert ok is False and "list of lists of strings" in err
    ok, err = validate_exact_answer_shape("list", "not a list")
    assert ok is False

    # 4. summary: null (None)
    ok, err = validate_exact_answer_shape("summary", None)
    assert ok is True and err is None
    ok, err = validate_exact_answer_shape("summary", "[SYNTHETIC] Summary text")
    assert ok is False and "JSON null" in err

    # 5. Unknown question type
    ok, err = validate_exact_answer_shape("multiple_choice", None)
    assert ok is False and "Unknown question_type" in err


def test_validate_ideal_answer():
    """Verify ideal_answer must be a non-empty list of non-empty strings."""
    # Valid
    ok, err = validate_ideal_answer(["[SYNTHETIC] Ideal answer paragraph 1."])
    assert ok is True and err is None
    ok, err = validate_ideal_answer(["[SYNTHETIC] Ans 1", "[SYNTHETIC] Ans 2"])
    assert ok is True and err is None

    # Invalid
    ok, err = validate_ideal_answer([])
    assert ok is False and "cannot be empty" in err
    ok, err = validate_ideal_answer("[SYNTHETIC] Raw string")
    assert ok is False and "must be a list" in err
    ok, err = validate_ideal_answer(["[SYNTHETIC] Non-empty", "   "])
    assert ok is False and "must be a non-empty string" in err
    ok, err = validate_ideal_answer([123])
    assert ok is False


def test_extract_pmids_from_urls():
    """Verify PMID extraction using SPEC 9.3 regex on multiple URL formats."""
    urls = [
        "http://www.ncbi.nlm.nih.gov/pubmed/111001",
        "https://pubmed.ncbi.nlm.nih.gov/111002/",
        "111003",
        "https://ncbi.nlm.nih.gov/pubmed/111004?param=1",  # deviation handled
        "invalid_url_with_no_digits",
    ]
    pmids = extract_pmids_from_source_reference(urls)
    assert "111001" in pmids
    assert "111002" in pmids
    assert "111003" in pmids
    assert "invalid_url_with_no_digits" not in pmids


def test_parse_bioasq_questions_synthetic_csv(tmp_path):
    """Verify parsing synthetic CSV creates WorkflowInput and BioASQGoldLabel and counts rejections."""
    csv_file = tmp_path / "synthetic_questions.csv"
    rows = [
        {
            "sample_id": "SYNTH_Q01",
            "source_dataset": "BioASQ",
            "question_type": "yesno",
            "question": "[SYNTHETIC] Is drug X effective for condition Y?",
            "exact_answer": json.dumps("yes"),
            "ideal_answer": json.dumps(["[SYNTHETIC] Yes, clinical trials indicate efficacy."]),
            "context": json.dumps([{"text": "[SYNTHETIC] Context snippet 1", "document": "http://pubmed/111"}]),
            "source_reference": json.dumps(["http://pubmed/111"]),
            "concepts": json.dumps(["http://concept/1"]),
            "metadata": json.dumps({"_type": "yesno"}),
        },
        {
            "sample_id": "SYNTH_Q02",
            "source_dataset": "BioASQ",
            "question_type": "factoid",
            "question": "[SYNTHETIC] Which gene is mutated?",
            "exact_answer": json.dumps(["[SYNTHETIC] RET", "[SYNTHETIC] MEN2"]),
            "ideal_answer": json.dumps(["[SYNTHETIC] Mutations in RET are causative."]),
            "context": json.dumps([{"text": "[SYNTHETIC] Context snippet 2", "document": "http://pubmed/222"}]),
            "source_reference": json.dumps(["http://pubmed/222"]),
            "concepts": json.dumps([]),
            "metadata": json.dumps({}),
        },
        # Invalid row 1: bad exact answer shape for summary
        {
            "sample_id": "SYNTH_Q03_INVALID",
            "source_dataset": "BioASQ",
            "question_type": "summary",
            "question": "[SYNTHETIC] Summarize finding Z",
            "exact_answer": json.dumps("not null"),
            "ideal_answer": json.dumps(["[SYNTHETIC] Ideal summary."]),
            "context": json.dumps([]),
            "source_reference": json.dumps([]),
            "concepts": json.dumps([]),
            "metadata": json.dumps({}),
        },
        # Invalid row 2: empty ideal answer
        {
            "sample_id": "SYNTH_Q04_INVALID",
            "source_dataset": "BioASQ",
            "question_type": "list",
            "question": "[SYNTHETIC] List receptors",
            "exact_answer": json.dumps([["[SYNTHETIC] RecA"]]),
            "ideal_answer": json.dumps([]),
            "context": json.dumps([]),
            "source_reference": json.dumps([]),
            "concepts": json.dumps([]),
            "metadata": json.dumps({}),
        },
    ]

    with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    wf_inputs, gold_labels, rejections = parse_bioasq_questions(str(csv_file))

    assert len(wf_inputs) == 2
    assert len(gold_labels) == 2
    assert wf_inputs[0].sample_id == "SYNTH_Q01"
    assert wf_inputs[0].question == "[SYNTHETIC] Is drug X effective for condition Y?"
    assert wf_inputs[0].question_type == "yesno"
    assert not hasattr(wf_inputs[0], "exact_answer")
    assert not hasattr(wf_inputs[0], "ideal_answer")

    assert gold_labels["SYNTH_Q01"].exact_answer == "yes"
    assert gold_labels["SYNTH_Q01"].gold_pmids == ["111"]

    assert len(rejections) == 2
    assert any("invalid_exact_answer_shape" in k for k in rejections)
    assert any("invalid_ideal_answer" in k for k in rejections)


def test_row_level_relevance_multi_chunk(synthetic_sqlite_db, tmp_path):
    """Verify row-level relevance: a corpus row split into multiple chunks makes all chunks relevant (SPEC A11)."""
    # 1. Populate mock DB documents and passages with multi-chunk row E999
    from backend.app.persistence.models import EvidenceDocument
    doc = EvidenceDocument(
        id="doc-1",
        source="BIOASQ",
        source_id="111999",
        title="[SYNTHETIC] Document for multi-chunk test",
    )
    p1 = EvidencePassage(
        id="p-1",
        document_id="doc-1",
        evidence_id="ev:bioasq:E999:0",
        chunk_index=0,
        text="[SYNTHETIC] Chunk 0 text",
        text_hash="hash_chunk_0",
    )
    p2 = EvidencePassage(
        id="p-2",
        document_id="doc-1",
        evidence_id="ev:bioasq:E999:1",
        chunk_index=1,
        text="[SYNTHETIC] Chunk 1 text",
        text_hash="hash_chunk_1",
    )
    synthetic_sqlite_db.add_all([doc, p1, p2])
    synthetic_sqlite_db.commit()

    # 2. Write corpus fixture where SYNTH_Q_ROW carries E999
    corpus_file = tmp_path / "synthetic_corpus.csv"
    with open(corpus_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["evidence_id", "snippet_text", "pubmed_url", "section", "sample_id", "question_type"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "evidence_id": "E999",
                "snippet_text": "[SYNTHETIC] Source text spanning multiple chunks.",
                "pubmed_url": "http://pubmed/999",
                "section": "abstract",
                "sample_id": "SYNTH_Q_ROW",
                "question_type": "factoid",
            }
        )

    mapping, stats = build_derived_relevance_mapping(str(corpus_file), synthetic_sqlite_db)

    # Question's relevant passages MUST contain BOTH chunk 0 and chunk 1
    assert "SYNTH_Q_ROW" in mapping
    assert mapping["SYNTH_Q_ROW"] == ["ev:bioasq:E999:0", "ev:bioasq:E999:1"]
    assert stats["missing_legacy_ids_count"] == 0


def test_coverage_and_mismatch_reporting(tmp_path):
    """Verify coverage report identifies questions where n_corpus_rows < n_context and excluded questions."""
    q_file = tmp_path / "questions.csv"
    c_file = tmp_path / "corpus.csv"

    # Q1: 3 context snippets, but only 2 corpus rows (n_corpus < n_context)
    # Q2: 2 context snippets, 0 corpus rows (EXCLUDED)
    q_rows = [
        {
            "sample_id": "Q1",
            "source_dataset": "BioASQ",
            "question_type": "factoid",
            "question": "[SYNTHETIC] Q1?",
            "exact_answer": json.dumps(["[SYNTHETIC] A1"]),
            "ideal_answer": json.dumps(["[SYNTHETIC] Ideal 1"]),
            "context": json.dumps([
                {"text": "[SYNTHETIC] Snippet One"},
                {"text": "[SYNTHETIC] Snippet Two"},
                {"text": "[SYNTHETIC] Snippet Three"},
            ]),
            "source_reference": json.dumps([]),
            "concepts": json.dumps([]),
            "metadata": json.dumps({}),
        },
        {
            "sample_id": "Q2_EXCLUDED",
            "source_dataset": "BioASQ",
            "question_type": "yesno",
            "question": "[SYNTHETIC] Q2?",
            "exact_answer": json.dumps("yes"),
            "ideal_answer": json.dumps(["[SYNTHETIC] Ideal 2"]),
            "context": json.dumps([
                {"text": "[SYNTHETIC] Snippet A"},
                {"text": "[SYNTHETIC] Snippet B"},
            ]),
            "source_reference": json.dumps([]),
            "concepts": json.dumps([]),
            "metadata": json.dumps({}),
        },
    ]

    c_rows = [
        {
            "evidence_id": "E1",
            "snippet_text": "[SYNTHETIC] Snippet One",
            "pubmed_url": "http://pubmed/1",
            "section": "abstract",
            "sample_id": "Q1",
            "question_type": "factoid",
        },
        {
            "evidence_id": "E2",
            "snippet_text": "[synthetic]   snippet two  ",  # normalized match
            "pubmed_url": "http://pubmed/2",
            "section": "abstract",
            "sample_id": "Q1",
            "question_type": "factoid",
        },
    ]

    with open(q_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
        writer.writeheader()
        writer.writerows(q_rows)

    with open(c_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(c_rows[0].keys()))
        writer.writeheader()
        writer.writerows(c_rows)

    cov = compute_coverage_report(str(q_file), str(c_file))

    assert cov["total_questions_parsed"] == 2
    assert cov["questions_with_corpus_rows"] == 1
    assert cov["excluded_questions_count"] == 1
    assert cov["excluded_questions_by_type"] == {"yesno": 1}
    assert cov["exact_matches"] == 1
    assert cov["normalized_matches"] == 1
    assert cov["questions_corpus_lt_context_count"] == 1
    assert cov["questions_corpus_lt_context_ids"] == ["Q1"]
    assert cov["total_context_snippets"] == 5
    assert cov["total_corpus_rows"] == 2


def test_persist_evaluation_labels_idempotence(synthetic_sqlite_db):
    """Verify persisting evaluation labels is idempotent (second run creates 0 new rows)."""
    labels = {
        "SYNTH_Q1": BioASQGoldLabel(
            sample_id="SYNTH_Q1",
            question_type="yesno",
            exact_answer="yes",
            ideal_answer=["[SYNTHETIC] Ideal 1"],
            gold_pmids=["1001"],
            relevant_passages=["ev:bioasq:E1:0"],
        ),
        "SYNTH_Q2": BioASQGoldLabel(
            sample_id="SYNTH_Q2",
            question_type="factoid",
            exact_answer=["[SYNTHETIC] Entity"],
            ideal_answer=["[SYNTHETIC] Ideal 2"],
            gold_pmids=["1002"],
            relevant_passages=["ev:bioasq:E2:0", "ev:bioasq:E2:1"],
        ),
    }
    relevance_map = {
        "SYNTH_Q1": ["ev:bioasq:E1:0"],
        "SYNTH_Q2": ["ev:bioasq:E2:0", "ev:bioasq:E2:1"],
    }

    # Run 1: inserts rows
    res1 = persist_evaluation_labels(synthetic_sqlite_db, labels, relevance_map)
    assert res1["inserted"] == 2
    assert res1["updated"] == 0

    rows1 = synthetic_sqlite_db.query(EvaluationLabel).all()
    assert len(rows1) == 2
    rec = synthetic_sqlite_db.query(EvaluationLabel).filter_by(sample_id="SYNTH_Q1").one()
    assert rec.exact_answer == "yes"
    assert rec.gold_pmids == ["1001"]
    assert rec.relevant_passages == ["ev:bioasq:E1:0"]
    assert rec.is_derived is True
    assert rec.confirmed_by_human is False

    # Run 2: idempotent repeat run must insert 0 new rows
    res2 = persist_evaluation_labels(synthetic_sqlite_db, labels, relevance_map)
    assert res2["inserted"] == 0
    assert res2["updated"] == 2

    rows2 = synthetic_sqlite_db.query(EvaluationLabel).all()
    assert len(rows2) == 2
