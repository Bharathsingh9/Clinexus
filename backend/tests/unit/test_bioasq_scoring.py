"""Unit Tests for Pure BioASQ Metric Scorers (Phase 5 Stage C).

Specification Section 9.9, Appendix A11 (Req ID: EVL-2):
- All tests use synthetic values only; no real BioASQ text.
- Every scorer has hand-computed expected values written out in the test with arithmetic shown in comments.
- Tests cover:
    * text normalization (lowercase, NFKC, punctuation removal, whitespace collapse)
    * yes/no accuracy and macro-F1 on an imbalanced set
    * factoid strict, lenient, and MRR with case, punctuation, and synonyms
    * list precision, recall, and F1 including duplicates, partial matches, multi-synonym gold
    * abstention handling in both views (abstention changes all-questions view, not answered-only)
    * empty gold and empty prediction edge cases
    * retrieval metrics with chunk collapsing, duplicate groups, excluded empty-gold questions, and question counts
    * gold_document_citation_rate pure function over PMID sets
"""

import math
import pytest

from backend.app.evaluation.bioasq_scoring import (
    SCORING_BASIS,
    normalize_text,
    score_factoid,
    score_gold_document_citation_rate,
    score_list,
    score_retrieval,
    score_summary,
    score_yesno,
)


def test_normalize_text_properties():
    """Verify lowercase, Unicode NFKC, punctuation stripping, and whitespace collapsing."""
    # Case and whitespace
    assert normalize_text("  ASPIRIN  ") == "aspirin"
    # Punctuation stripping
    assert normalize_text("Aspirin, Ibuprofen; and Paracetamol.") == "aspirin ibuprofen and paracetamol"
    assert normalize_text("(Covid-19)!") == "covid19"
    # Unicode NFKC normalization: e.g. ligature ﬁ -> fi
    assert normalize_text("ﬁndings") == "findings"
    # Empty string / None
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_yesno_scoring_hand_computed_arithmetic():
    """Verify yes/no accuracy and macro-F1 on an imbalanced synthetic dataset.

    ARITHMETIC BREAKDOWN:
    Dataset of 4 synthetic questions:
      - Q1: gold="yes", predicted="yes"  (Answered, correct 'yes')
      - Q2: gold="yes", predicted="no"   (Answered, incorrect 'no', false negative for 'yes')
      - Q3: gold="no",  predicted="no"   (Answered, correct 'no')
      - Q4: gold="no",  predicted="ABSTAIN" (Abstained question)

    COUNTS:
      - Answered: 3 (Q1, Q2, Q3)
      - Abstained: 1 (Q4)
      - Failed: 0
      - Total: 4

    VIEW 1: ANSWERED-ONLY (evaluated on Q1, Q2, Q3):
      - Total answered = 3
      - Correct = 2 (Q1, Q3)
      - Accuracy = 2 / 3 = 0.6666666666666666
      - Class 'yes':
          TP_yes = 1 (Q1)
          FP_yes = 0
          FN_yes = 1 (Q2)
          P_yes = 1 / (1 + 0) = 1.0
          R_yes = 1 / (1 + 1) = 0.5
          F1_yes = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 2/3 = 0.6666666666666666
      - Class 'no':
          TP_no = 1 (Q3)
          FP_no = 1 (Q2)
          FN_no = 0
          P_no = 1 / (1 + 1) = 0.5
          R_no = 1 / (1 + 0) = 1.0
          F1_no = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 = 0.6666666666666666
      - Macro-F1 = (F1_yes + F1_no) / 2 = (2/3 + 2/3) / 2 = 2/3 = 0.6666666666666666

    VIEW 2: ALL-QUESTIONS (evaluated across all 4 questions):
      - Total = 4
      - Correct = 2 (Q1, Q3)
      - Accuracy = 2 / 4 = 0.50
      - Class 'yes':
          TP_yes = 1 (Q1)
          FP_yes = 0
          FN_yes = 1 (Q2)
          P_yes = 1.0
          R_yes = 0.5
          F1_yes = 2/3 = 0.6666666666666666
      - Class 'no':
          TP_no = 1 (Q3)
          FP_no = 1 (Q2)
          FN_no = 1 (Q4 gold was 'no', model abstained -> missed!)
          P_no = 1 / (1 + 1) = 0.5
          R_no = 1 / (1 + 1) = 0.5
          F1_no = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5 / 1.0 = 0.50
      - Macro-F1 = (2/3 + 0.50) / 2 = (7/6) / 2 = 7/12 = 0.5833333333333334
    """
    records = [
        {"predicted": "yes", "gold": "yes", "workflow_abstained": False, "failed": False},
        {"predicted": "no", "gold": "yes", "workflow_abstained": False, "failed": False},
        {"predicted": "no", "gold": "no", "workflow_abstained": False, "failed": False},
        {"predicted": "ABSTAIN", "gold": "no", "workflow_abstained": False, "failed": False},
    ]

    res = score_yesno(records)

    assert res.scoring_basis == SCORING_BASIS
    assert res.counts.answered == 3
    assert res.counts.abstained == 1
    assert res.counts.failed == 0
    assert res.counts.total == 4

    # Answered-only assertions
    assert math.isclose(res.answered_only.accuracy, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.answered_only.f1_yes, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.answered_only.f1_no, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.answered_only.macro_f1, 2.0 / 3.0, rel_tol=1e-6)

    # All-questions assertions: abstention lowered accuracy and macro-F1
    assert math.isclose(res.all_questions.accuracy, 2.0 / 4.0, rel_tol=1e-6)
    assert math.isclose(res.all_questions.f1_yes, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.all_questions.f1_no, 0.50, rel_tol=1e-6)
    assert math.isclose(res.all_questions.macro_f1, 7.0 / 12.0, rel_tol=1e-6)


def test_factoid_scoring_hand_computed_arithmetic():
    """Verify factoid strict accuracy, lenient accuracy, and MRR.

    ARITHMETIC BREAKDOWN:
    Dataset of 4 synthetic questions:
      - Q1: gold=["Aspirin", "Acetylsalicylic acid"]
            predicted=["aspirin.", "ibuprofen"]
            Rank 1 matches gold synonym "Aspirin" after normalization.
            Strict = 1.0, Lenient = 1.0, RR = 1/1 = 1.0.

      - Q2: gold=["Imatinib", "Gleevec"]
            predicted=["dasatinib", "Gleevec!"]
            Rank 1 ("dasatinib") does not match.
            Rank 2 ("Gleevec!") matches gold synonym "Gleevec".
            Strict = 0.0, Lenient = 1.0, RR = 1/2 = 0.50.

      - Q3: gold=["Metformin"]
            predicted=["insulin", "sulfonylurea", "glipizide"]
            No match in top 5.
            Strict = 0.0, Lenient = 0.0, RR = 0.0.

      - Q4: gold=["Paracetamol"]
            predicted="ABSTAIN"
            Abstained.

    COUNTS:
      - Answered: 3 (Q1, Q2, Q3)
      - Abstained: 1 (Q4)
      - Failed: 0
      - Total: 4

    VIEW 1: ANSWERED-ONLY (3 questions):
      - Strict accuracy = (1.0 + 0.0 + 0.0) / 3 = 1/3 = 0.3333333333333333
      - Lenient accuracy = (1.0 + 1.0 + 0.0) / 3 = 2/3 = 0.6666666666666666
      - MRR = (1.0 + 0.5 + 0.0) / 3 = 1.5 / 3 = 0.50

    VIEW 2: ALL-QUESTIONS (4 questions, Q4 contributes 0.0):
      - Strict accuracy = (1.0 + 0.0 + 0.0 + 0.0) / 4 = 1/4 = 0.25
      - Lenient accuracy = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 2/4 = 0.50
      - MRR = (1.0 + 0.5 + 0.0 + 0.0) / 4 = 1.5 / 4 = 0.375
    """
    records = [
        {
            "predicted": ["aspirin.", "ibuprofen"],
            "gold": ["Aspirin", "Acetylsalicylic acid"],
            "workflow_abstained": False,
            "failed": False,
        },
        {
            "predicted": ["dasatinib", "Gleevec!"],
            "gold": ["Imatinib", "Gleevec"],
            "workflow_abstained": False,
            "failed": False,
        },
        {
            "predicted": ["insulin", "sulfonylurea", "glipizide"],
            "gold": ["Metformin"],
            "workflow_abstained": False,
            "failed": False,
        },
        {
            "predicted": "ABSTAIN",
            "gold": ["Paracetamol"],
            "workflow_abstained": False,
            "failed": False,
        },
    ]

    res = score_factoid(records)

    assert res.scoring_basis == SCORING_BASIS
    assert res.counts.answered == 3
    assert res.counts.abstained == 1
    assert res.counts.total == 4

    # Answered-only
    assert math.isclose(res.answered_only.strict_accuracy, 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.answered_only.lenient_accuracy, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.answered_only.mrr, 0.50, rel_tol=1e-6)

    # All-questions
    assert math.isclose(res.all_questions.strict_accuracy, 0.25, rel_tol=1e-6)
    assert math.isclose(res.all_questions.lenient_accuracy, 0.50, rel_tol=1e-6)
    assert math.isclose(res.all_questions.mrr, 0.375, rel_tol=1e-6)


def test_list_scoring_hand_computed_arithmetic():
    """Verify list question precision, recall, and F1 with duplicates and multi-synonym gold entities.

    ARITHMETIC BREAKDOWN:
    Dataset of 3 synthetic questions:
      - Q1:
          Gold entities (2):
            Entity 0: ["Epinephrine", "Adrenaline"]
            Entity 1: ["Norepinephrine", "Noradrenaline"]
          Predicted (4 items):
            ["adrenaline.", "noradrenaline!", "dopamine", "adrenaline"]
          Matching process:
            1. "adrenaline." matches Entity 0 -> matched_gold = {0}, TP=1
            2. "noradrenaline!" matches Entity 1 -> matched_gold = {0, 1}, TP=2
            3. "dopamine" matches neither entity -> TP=2
            4. "adrenaline" matches Entity 0, but Entity 0 is already matched! Duplicate prediction counts as extra.
          For Q1:
            TP = 2
            N_pred = 4
            N_gold = 2
            Precision = 2 / 4 = 0.50
            Recall = 2 / 2 = 1.00
            F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3 = 0.6666666666666666

      - Q2:
          Gold entities (2):
            Entity 0: ["Penicillin"]
            Entity 1: ["Amoxicillin"]
          Predicted (2 items):
            ["penicillin", "erythromycin"]
          Matching:
            1. "penicillin" matches Entity 0 -> TP=1
            2. "erythromycin" matches none
          For Q2:
            TP = 1
            N_pred = 2
            N_gold = 2
            Precision = 1 / 2 = 0.50
            Recall = 1 / 2 = 0.50
            F1 = 0.50

      - Q3:
          Gold entities (1):
            Entity 0: ["Cefazolin"]
          Predicted: "ABSTAIN"
          Abstained question.

    COUNTS:
      - Answered: 2 (Q1, Q2)
      - Abstained: 1 (Q3)
      - Failed: 0
      - Total: 3

    VIEW 1: ANSWERED-ONLY (2 questions):
      - Mean Precision = (0.50 + 0.50) / 2 = 0.50
      - Mean Recall = (1.00 + 0.50) / 2 = 0.75
      - Mean F1 = (2/3 + 0.50) / 2 = (7/6) / 2 = 7/12 = 0.5833333333333334

    VIEW 2: ALL-QUESTIONS (3 questions, Q3 contributes 0.0):
      - Mean Precision = (0.50 + 0.50 + 0.0) / 3 = 1.0 / 3 = 0.3333333333333333
      - Mean Recall = (1.00 + 0.50 + 0.0) / 3 = 1.5 / 3 = 0.50
      - Mean F1 = (2/3 + 0.50 + 0.0) / 3 = (7/6) / 3 = 7/18 = 0.3888888888888889
    """
    records = [
        {
            "predicted": ["adrenaline.", "noradrenaline!", "dopamine", "adrenaline"],
            "gold": [
                ["Epinephrine", "Adrenaline"],
                ["Norepinephrine", "Noradrenaline"],
            ],
            "workflow_abstained": False,
            "failed": False,
        },
        {
            "predicted": ["penicillin", "erythromycin"],
            "gold": [["Penicillin"], ["Amoxicillin"]],
            "workflow_abstained": False,
            "failed": False,
        },
        {
            "predicted": "ABSTAIN",
            "gold": [["Cefazolin"]],
            "workflow_abstained": False,
            "failed": False,
        },
    ]

    res = score_list(records)

    assert res.scoring_basis == SCORING_BASIS
    assert res.counts.answered == 2
    assert res.counts.abstained == 1
    assert res.counts.total == 3

    # Answered-only
    assert math.isclose(res.answered_only.mean_precision, 0.50, rel_tol=1e-6)
    assert math.isclose(res.answered_only.mean_recall, 0.75, rel_tol=1e-6)
    assert math.isclose(res.answered_only.mean_f1, 7.0 / 12.0, rel_tol=1e-6)

    # All-questions
    assert math.isclose(res.all_questions.mean_precision, 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(res.all_questions.mean_recall, 0.50, rel_tol=1e-6)
    assert math.isclose(res.all_questions.mean_f1, 7.0 / 18.0, rel_tol=1e-6)


def test_list_scoring_edge_cases():
    """Verify empty gold and empty prediction edge cases for list questions."""
    # Both empty -> perfect match (1.0, 1.0, 1.0)
    res_both_empty = score_list([{"predicted": [], "gold": [], "failed": False}])
    assert res_both_empty.answered_only.mean_precision == 1.0
    assert res_both_empty.answered_only.mean_recall == 1.0
    assert res_both_empty.answered_only.mean_f1 == 1.0

    # Prediction empty with non-empty gold -> 0.0
    res_pred_empty = score_list([{"predicted": [], "gold": [["target"]], "failed": False}])
    assert res_pred_empty.answered_only.mean_precision == 0.0
    assert res_pred_empty.answered_only.mean_recall == 0.0
    assert res_pred_empty.answered_only.mean_f1 == 0.0

    # Prediction non-empty with empty gold -> 0.0
    res_gold_empty = score_list([{"predicted": ["entity"], "gold": [], "failed": False}])
    assert res_gold_empty.answered_only.mean_precision == 0.0
    assert res_gold_empty.answered_only.mean_recall == 0.0
    assert res_gold_empty.answered_only.mean_f1 == 0.0


def test_summary_scoring_not_evaluated():
    """Verify summary scoring reports evaluated=False with reason and correct counts."""
    records = [
        {"predicted": "Valid synthetic summary text", "gold": ["Ideal synthetic answer"], "failed": False},
        {"predicted": "ABSTAIN", "gold": ["Ideal synthetic answer"], "failed": False},
        {"predicted": None, "gold": ["Ideal synthetic answer"], "failed": True},
    ]

    res = score_summary(records)
    assert res.scoring_basis == SCORING_BASIS
    assert res.evaluated is False
    assert "ROUGE" in res.reason
    assert res.counts.answered == 1
    assert res.counts.abstained == 1
    assert res.counts.failed == 1
    assert res.counts.total == 3


def test_retrieval_scoring_hand_computed_arithmetic():
    """Verify retrieval metrics with chunk collapsing, duplicate groups, and excluded questions.

    ARITHMETIC BREAKDOWN:
    3 synthetic questions evaluated:

    - Question Q1:
        Relevant rows: {"row_1", "row_2"}  (total relevant R = 2)
        Duplicate groups: None
        Retrieved chunks:
          * ev:bioasq:row_1:0 with score 0.85
          * ev:bioasq:row_1:1 with score 0.92  -> Collapses to row_1 with score 0.92!
          * ev:bioasq:row_3:0 with score 0.70  -> Collapses to row_3 with score 0.70 (not relevant)
          * ev:bioasq:row_2:0 with score 0.60  -> Collapses to row_2 with score 0.60 (relevant)
        Ranked rows:
          1. row_1 (rel=1)
          2. row_3 (rel=0)
          3. row_2 (rel=1)
        Metrics for Q1:
          Total relevant R = 2
          Top 5: [row_1, row_3, row_2] -> 2 relevant
          Recall@5 = 2 / 2 = 1.0
          Precision@5 = 2 / 5 = 0.40
          Recall@10 = 2 / 2 = 1.0
          MRR: first relevant is rank 1 -> RR = 1 / 1 = 1.0
          MAP (Average Precision normalized by R=2):
            rank 1 (row_1, rel=1): P@1 = 1/1 = 1.0. Contrib = 1.0 * 1 = 1.0
            rank 2 (row_3, rel=0): P@2 = 1/2 = 0.5. Contrib = 0.5 * 0 = 0.0
            rank 3 (row_2, rel=1): P@3 = 2/3. Contrib = (2/3) * 1 = 2/3
            Sum = 1.0 + 2/3 = 5/3
            AP = (5/3) / 2 = 5/6 = 0.8333333333333334

    - Question Q2:
        Relevant rows: {"row_A"}  (total relevant R = 1)
        Duplicate groups: {"row_B": {"row_A", "row_B"}} (row_B is duplicate of row_A)
        Retrieved chunks:
          * ev:bioasq:row_C:0 with score 0.80 (rel=0)
          * ev:bioasq:row_B:0 with score 0.50 (in duplicate group of row_A -> rel=1!)
        Ranked rows:
          1. row_C (rel=0)
          2. row_B (rel=1 via duplicate group)
        Metrics for Q2:
          Total relevant R = 1
          Top 5: [row_C, row_B] -> 1 relevant
          Recall@5 = 1 / 1 = 1.0
          Precision@5 = 1 / 5 = 0.20
          Recall@10 = 1 / 1 = 1.0
          MRR: first relevant is rank 2 -> RR = 1 / 2 = 0.50
          MAP (Average Precision normalized by R=1):
            rank 1 (row_C, rel=0): P@1 = 0/1 = 0.0. Contrib = 0.0
            rank 2 (row_B, rel=1): P@2 = 1/2 = 0.50. Contrib = 0.50 * 1 = 0.50
            AP = 0.50 / 1 = 0.50

    - Question Q3:
        Relevant rows: {}  (empty relevant row set!)
        SPEC / Rule: Questions with no relevant rows are excluded and counted.
        Excluded from metric calculation!

    AGGREGATE METRICS (over included questions Q1 and Q2, num_questions = 2):
      - Mean Recall@5 = (1.0 + 1.0) / 2 = 1.0
      - Mean Recall@10 = (1.0 + 1.0) / 2 = 1.0
      - Mean Precision@5 = (0.40 + 0.20) / 2 = 0.30
      - Mean MRR = (1.0 + 0.50) / 2 = 0.75
      - Mean MAP = (5/6 + 1/2) / 2 = (5/6 + 3/6) / 2 = (8/6) / 2 = 4/6 = 2/3 = 0.6666666666666666
      - num_questions beside every metric = 2
      - excluded_no_relevant_rows = 1
    """
    question_retrievals = [
        {
            "question_id": "Q1",
            "relevant_row_ids": {"row_1", "row_2"},
            "duplicate_groups": None,
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_1:0", "retrieval_score": 0.85},
                {"evidence_id": "ev:bioasq:row_1:1", "retrieval_score": 0.92},
                {"evidence_id": "ev:bioasq:row_3:0", "retrieval_score": 0.70},
                {"evidence_id": "ev:bioasq:row_2:0", "retrieval_score": 0.60},
            ],
        },
        {
            "question_id": "Q2",
            "relevant_row_ids": {"row_A"},
            "duplicate_groups": {"row_B": {"row_A", "row_B"}},
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_C:0", "retrieval_score": 0.80},
                {"evidence_id": "ev:bioasq:row_B:0", "retrieval_score": 0.50},
            ],
        },
        {
            "question_id": "Q3",
            "relevant_row_ids": set(),  # Empty!
            "retrieved_chunks": [
                {"evidence_id": "ev:bioasq:row_Z:0", "retrieval_score": 0.95},
            ],
        },
    ]

    res = score_retrieval(question_retrievals)

    assert res.scoring_basis == SCORING_BASIS
    assert res.included_questions == 2
    assert res.excluded_no_relevant_rows == 1

    # Check that num_questions is returned beside every metric
    assert res.recall_at_5.num_questions == 2
    assert res.recall_at_10.num_questions == 2
    assert res.precision_at_5.num_questions == 2
    assert res.mrr.num_questions == 2
    assert res.map.num_questions == 2

    # Verify arithmetic values
    assert math.isclose(res.recall_at_5.value, 1.0, rel_tol=1e-6)
    assert math.isclose(res.recall_at_10.value, 1.0, rel_tol=1e-6)
    assert math.isclose(res.precision_at_5.value, 0.30, rel_tol=1e-6)
    assert math.isclose(res.mrr.value, 0.75, rel_tol=1e-6)
    assert math.isclose(res.map.value, 2.0 / 3.0, rel_tol=1e-6)


def test_gold_document_citation_rate_toy_set():
    """Verify gold_document_citation_rate calculation on synthetic answered questions.

    Toy dataset of 4 answered questions:
      - Q1: cited={"1001", "1002"}, gold={"1002", "1003"} -> hit (overlap on "1002")
      - Q2: cited={"2001"}, gold={"2002"} -> miss
      - Q3: cited={"3001", "3002"}, gold={"3002"} -> hit (overlap on "3002")
      - Q4: cited={"4001"}, gold=set() -> miss (no gold PMIDs)

    Hits = 2
    Total answered = 4
    Rate = 2 / 4 = 0.50
    """
    records = [
        ({"1001", "1002"}, {"1002", "1003"}),
        ({"2001"}, {"2002"}),
        ({"3001", "3002"}, {"3002"}),
        ({"4001"}, set()),
    ]

    res = score_gold_document_citation_rate(records)
    assert res.scoring_basis == SCORING_BASIS
    assert res.hits == 2
    assert res.total_answered == 4
    assert math.isclose(res.rate, 0.50, rel_tol=1e-6)

    # Empty answered questions edge case
    empty_res = score_gold_document_citation_rate([])
    assert empty_res.hits == 0
    assert empty_res.total_answered == 0
    assert empty_res.rate == 0.0
