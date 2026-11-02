"""BioASQ Metric Scorers and Evaluation Functions (Phase 5 Stage C).

Specification Section 9.9, Appendix A11 (Req ID: EVL-2):
- Pure, tested functions with zero I/O and zero network operations.
- Text normalization: lowercase, Unicode NFKC, strip punctuation, collapse whitespace.
- Official BioASQ measures URL unreachable without participant credentials (404/auth wall);
  all scorers labeled scoring_basis="APPROXIMATION"; never claim official equivalence.
- Question type scorers:
    * yesno: accuracy and macro-F1 over yes and no.
    * factoid: gold list = acceptable surface forms of the one correct answer (SPEC 9.9).
      Strict accuracy (rank 1), lenient accuracy (top 5), MRR (first rank matching).
    * list: per-question precision, recall, F1 against gold entities (synonyms, matched <= 1 time,
      duplicate predictions counted as extra), then mean across questions.
    * summary: ROUGE dependency not added per Stage C decision. Evaluated=False with reason.
- Aggregation:
    * Returns BOTH answered-only view and all-questions view.
    * Per-question counts of answered, abstained, and failed records per type, never pooled.
- Retrieval metrics at corpus-row level (SPEC A11):
    * Map retrieved chunks to corpus rows (legacy_id), collapse keeping best score,
      rank by best score, apply duplicate-group rule.
    * Compute Recall@5, Recall@10, Precision@5, MRR, and MAP (AP normalized by total relevant rows).
    * Questions with no relevant rows excluded and counted.
    * Returns question count beside every metric.
- Grounding:
    * gold_document_citation_rate: answered questions citing >= 1 gold PMID / answered questions.
"""

import string
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

SCORING_BASIS = "APPROXIMATION"


def normalize_text(text: Optional[str]) -> str:
    """Normalize text before matching: lowercase, Unicode NFKC, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    # 1. Unicode NFKC
    norm = unicodedata.normalize("NFKC", str(text))
    # 2. Lowercase
    norm = norm.lower()
    # 3. Strip punctuation
    punc_table = str.maketrans("", "", string.punctuation)
    norm = norm.translate(punc_table)
    # 4. Collapse whitespace
    norm = " ".join(norm.split())
    return norm


class QuestionTypeCounts(BaseModel):
    """Execution counts for a specific question type."""
    model_config = ConfigDict(extra="forbid")

    answered: int = Field(..., ge=0, description="Questions answered by the workflow and extractor")
    abstained: int = Field(..., ge=0, description="Questions abstained by workflow or extractor")
    failed: int = Field(..., ge=0, description="Questions that failed with an error")
    total: int = Field(..., ge=0, description="Total questions submitted")


class YesNoMetrics(BaseModel):
    """Metrics for yesno questions."""
    model_config = ConfigDict(extra="forbid")

    accuracy: float
    macro_f1: float
    f1_yes: float
    f1_no: float


class YesNoScoringResult(BaseModel):
    """Scoring result for yesno questions carrying dual views."""
    model_config = ConfigDict(extra="forbid")

    question_type: str = "yesno"
    scoring_basis: str = SCORING_BASIS
    counts: QuestionTypeCounts
    answered_only: YesNoMetrics
    all_questions: YesNoMetrics


class FactoidMetrics(BaseModel):
    """Metrics for factoid questions."""
    model_config = ConfigDict(extra="forbid")

    strict_accuracy: float
    lenient_accuracy: float
    mrr: float


class FactoidScoringResult(BaseModel):
    """Scoring result for factoid questions carrying dual views."""
    model_config = ConfigDict(extra="forbid")

    question_type: str = "factoid"
    scoring_basis: str = SCORING_BASIS
    counts: QuestionTypeCounts
    answered_only: FactoidMetrics
    all_questions: FactoidMetrics


class ListMetrics(BaseModel):
    """Metrics for list questions."""
    model_config = ConfigDict(extra="forbid")

    mean_precision: float
    mean_recall: float
    mean_f1: float


class ListScoringResult(BaseModel):
    """Scoring result for list questions carrying dual views."""
    model_config = ConfigDict(extra="forbid")

    question_type: str = "list"
    scoring_basis: str = SCORING_BASIS
    counts: QuestionTypeCounts
    answered_only: ListMetrics
    all_questions: ListMetrics


class SummaryScoringResult(BaseModel):
    """Scoring result for summary questions indicating omitted ROUGE dependency."""
    model_config = ConfigDict(extra="forbid")

    question_type: str = "summary"
    evaluated: bool = False
    reason: str = "ROUGE dependency not installed; summary evaluation omitted per SPEC 9.9 and Stage C decision"
    scoring_basis: str = SCORING_BASIS
    counts: QuestionTypeCounts


class MetricWithCount(BaseModel):
    """Metric value paired with the count of evaluating questions."""
    model_config = ConfigDict(extra="forbid")

    value: float
    num_questions: int


class RetrievalScoringResult(BaseModel):
    """Retrieval evaluation metrics at corpus-row level (SPEC A11)."""
    model_config = ConfigDict(extra="forbid")

    scoring_basis: str = SCORING_BASIS
    recall_at_5: MetricWithCount
    recall_at_10: MetricWithCount
    precision_at_5: MetricWithCount
    mrr: MetricWithCount
    map: MetricWithCount
    included_questions: int
    excluded_no_relevant_rows: int


class GoldDocumentCitationRateResult(BaseModel):
    """Grounding citation metric for answered questions citing gold PMIDs."""
    model_config = ConfigDict(extra="forbid")

    scoring_basis: str = SCORING_BASIS
    hits: int
    total_answered: int
    rate: float


def score_yesno(records: List[Dict[str, Any]]) -> YesNoScoringResult:
    """Score yes/no questions returning answered-only and all-questions metrics.

    Each record dict contains:
    - predicted: Optional[str] ("yes", "no", "ABSTAIN", or None)
    - gold: str ("yes" or "no")
    - workflow_abstained: bool (default False)
    - failed: bool (default False)
    """
    answered_records: List[Tuple[str, str]] = []  # (pred, gold)
    abstained_count = 0
    failed_count = 0

    for rec in records:
        if rec.get("failed", False):
            failed_count += 1
            continue
        if rec.get("workflow_abstained", False):
            abstained_count += 1
            continue

        raw_pred = rec.get("predicted")
        if raw_pred is None:
            failed_count += 1
            continue

        pred_norm = normalize_text(str(raw_pred))
        if pred_norm == "abstain":
            abstained_count += 1
            continue
        if pred_norm in ("yes", "no"):
            gold_norm = normalize_text(str(rec.get("gold", "")))
            answered_records.append((pred_norm, gold_norm))
        else:
            failed_count += 1

    answered_count = len(answered_records)
    total_count = answered_count + abstained_count + failed_count
    counts = QuestionTypeCounts(
        answered=answered_count,
        abstained=abstained_count,
        failed=failed_count,
        total=total_count,
    )

    # 1. Answered-only view
    if answered_count == 0:
        ans_metrics = YesNoMetrics(accuracy=0.0, macro_f1=0.0, f1_yes=0.0, f1_no=0.0)
    else:
        correct = sum(1 for p, g in answered_records if p == g)
        acc = correct / float(answered_count)

        tp_yes = sum(1 for p, g in answered_records if p == "yes" and g == "yes")
        fp_yes = sum(1 for p, g in answered_records if p == "yes" and g == "no")
        fn_yes = sum(1 for p, g in answered_records if p == "no" and g == "yes")

        p_yes = (tp_yes / float(tp_yes + fp_yes)) if (tp_yes + fp_yes) > 0 else 0.0
        r_yes = (tp_yes / float(tp_yes + fn_yes)) if (tp_yes + fn_yes) > 0 else 0.0
        f1_yes = (2.0 * p_yes * r_yes / (p_yes + r_yes)) if (p_yes + r_yes) > 0 else 0.0

        tp_no = sum(1 for p, g in answered_records if p == "no" and g == "no")
        fp_no = sum(1 for p, g in answered_records if p == "no" and g == "yes")
        fn_no = sum(1 for p, g in answered_records if p == "yes" and g == "no")

        p_no = (tp_no / float(tp_no + fp_no)) if (tp_no + fp_no) > 0 else 0.0
        r_no = (tp_no / float(tp_no + fn_no)) if (tp_no + fn_no) > 0 else 0.0
        f1_no = (2.0 * p_no * r_no / (p_no + r_no)) if (p_no + r_no) > 0 else 0.0

        macro_f1 = (f1_yes + f1_no) / 2.0
        ans_metrics = YesNoMetrics(accuracy=acc, macro_f1=macro_f1, f1_yes=f1_yes, f1_no=f1_no)

    # 2. All-questions view
    if total_count == 0:
        all_metrics = YesNoMetrics(accuracy=0.0, macro_f1=0.0, f1_yes=0.0, f1_no=0.0)
    else:
        # Correct answers among answered records
        correct = sum(1 for p, g in answered_records if p == g)
        acc_all = correct / float(total_count)

        # In all-questions view, any abstention or failure with gold="yes" counts as false negative for "yes"
        # and with gold="no" counts as false negative for "no".
        tp_yes = sum(1 for p, g in answered_records if p == "yes" and g == "yes")
        fp_yes = sum(1 for p, g in answered_records if p == "yes" and g == "no")

        tp_no = sum(1 for p, g in answered_records if p == "no" and g == "no")
        fp_no = sum(1 for p, g in answered_records if p == "no" and g == "yes")

        fn_yes = 0
        fn_no = 0
        for rec in records:
            g_norm = normalize_text(str(rec.get("gold", "")))
            if rec.get("failed", False) or rec.get("workflow_abstained", False):
                if g_norm == "yes":
                    fn_yes += 1
                elif g_norm == "no":
                    fn_no += 1
            else:
                p_raw = rec.get("predicted")
                p_n = normalize_text(str(p_raw)) if p_raw is not None else ""
                if p_n == "abstain" or p_n not in ("yes", "no"):
                    if g_norm == "yes":
                        fn_yes += 1
                    elif g_norm == "no":
                        fn_no += 1
                else:
                    if g_norm == "yes" and p_n != "yes":
                        fn_yes += 1
                    elif g_norm == "no" and p_n != "no":
                        fn_no += 1

        p_yes_all = (tp_yes / float(tp_yes + fp_yes)) if (tp_yes + fp_yes) > 0 else 0.0
        r_yes_all = (tp_yes / float(tp_yes + fn_yes)) if (tp_yes + fn_yes) > 0 else 0.0
        f1_yes_all = (2.0 * p_yes_all * r_yes_all / (p_yes_all + r_yes_all)) if (p_yes_all + r_yes_all) > 0 else 0.0

        p_no_all = (tp_no / float(tp_no + fp_no)) if (tp_no + fp_no) > 0 else 0.0
        r_no_all = (tp_no / float(tp_no + fn_no)) if (tp_no + fn_no) > 0 else 0.0
        f1_no_all = (2.0 * p_no_all * r_no_all / (p_no_all + r_no_all)) if (p_no_all + r_no_all) > 0 else 0.0

        macro_f1_all = (f1_yes_all + f1_no_all) / 2.0
        all_metrics = YesNoMetrics(
            accuracy=acc_all,
            macro_f1=macro_f1_all,
            f1_yes=f1_yes_all,
            f1_no=f1_no_all,
        )

    return YesNoScoringResult(
        counts=counts,
        answered_only=ans_metrics,
        all_questions=all_metrics,
    )


def score_factoid(records: List[Dict[str, Any]]) -> FactoidScoringResult:
    """Score factoid questions returning strict, lenient accuracy, and MRR.

    Gold list is treated as acceptable surface forms of the one correct answer (SPEC 9.9).
    Each record contains:
    - predicted: Optional[Union[str, List[str]]] (up to 5 ranked strings, or 'ABSTAIN')
    - gold: List[str] (acceptable surface forms)
    - workflow_abstained: bool (default False)
    - failed: bool (default False)
    """
    answered_scores: List[Tuple[float, float, float]] = []  # (strict, lenient, rr)
    abstained_count = 0
    failed_count = 0

    for rec in records:
        if rec.get("failed", False):
            failed_count += 1
            continue
        if rec.get("workflow_abstained", False):
            abstained_count += 1
            continue

        raw_pred = rec.get("predicted")
        if raw_pred is None:
            failed_count += 1
            continue

        if isinstance(raw_pred, str) and normalize_text(raw_pred) == "abstain":
            abstained_count += 1
            continue

        if isinstance(raw_pred, str):
            pred_list = [raw_pred]
        elif isinstance(raw_pred, list):
            pred_list = raw_pred
        else:
            failed_count += 1
            continue

        raw_gold = rec.get("gold", [])
        if isinstance(raw_gold, str):
            gold_list = [raw_gold]
        else:
            gold_list = list(raw_gold)

        gold_norm_set = {normalize_text(g) for g in gold_list if normalize_text(g)}
        pred_norm_list = [normalize_text(p) for p in pred_list if normalize_text(p)][:5]

        # Strict: rank 1 matches any gold string
        strict = 1.0 if (len(pred_norm_list) > 0 and pred_norm_list[0] in gold_norm_set) else 0.0

        # Lenient: any top 5 matches any gold string
        lenient = 1.0 if any(p in gold_norm_set for p in pred_norm_list) else 0.0

        # MRR: reciprocal rank of first match
        rr = 0.0
        for rank_idx, p in enumerate(pred_norm_list, start=1):
            if p in gold_norm_set:
                rr = 1.0 / float(rank_idx)
                break

        answered_scores.append((strict, lenient, rr))

    answered_count = len(answered_scores)
    total_count = answered_count + abstained_count + failed_count
    counts = QuestionTypeCounts(
        answered=answered_count,
        abstained=abstained_count,
        failed=failed_count,
        total=total_count,
    )

    if answered_count == 0:
        ans_metrics = FactoidMetrics(strict_accuracy=0.0, lenient_accuracy=0.0, mrr=0.0)
    else:
        ans_metrics = FactoidMetrics(
            strict_accuracy=sum(s for s, l, r in answered_scores) / float(answered_count),
            lenient_accuracy=sum(l for s, l, r in answered_scores) / float(answered_count),
            mrr=sum(r for s, l, r in answered_scores) / float(answered_count),
        )

    if total_count == 0:
        all_metrics = FactoidMetrics(strict_accuracy=0.0, lenient_accuracy=0.0, mrr=0.0)
    else:
        # Abstained and failed questions contribute 0.0
        all_metrics = FactoidMetrics(
            strict_accuracy=sum(s for s, l, r in answered_scores) / float(total_count),
            lenient_accuracy=sum(l for s, l, r in answered_scores) / float(total_count),
            mrr=sum(r for s, l, r in answered_scores) / float(total_count),
        )

    return FactoidScoringResult(
        counts=counts,
        answered_only=ans_metrics,
        all_questions=all_metrics,
    )


def score_list(records: List[Dict[str, Any]]) -> ListScoringResult:
    """Score list questions returning mean precision, recall, and F1.

    Each prediction matches an entity if it equals any synonym of that entity;
    each gold entity matched at most once; duplicate predictions counted as extra.
    Each record contains:
    - predicted: Optional[Union[str, List[str]]] (list of entity strings, or 'ABSTAIN')
    - gold: List[List[str]] (list of entities, each having synonyms)
    - workflow_abstained: bool (default False)
    - failed: bool (default False)
    """
    answered_scores: List[Tuple[float, float, float]] = []  # (p, r, f1)
    abstained_count = 0
    failed_count = 0

    for rec in records:
        if rec.get("failed", False):
            failed_count += 1
            continue
        if rec.get("workflow_abstained", False):
            abstained_count += 1
            continue

        raw_pred = rec.get("predicted")
        if raw_pred is None:
            failed_count += 1
            continue

        if isinstance(raw_pred, str) and normalize_text(raw_pred) == "abstain":
            abstained_count += 1
            continue

        if isinstance(raw_pred, str):
            pred_items = [raw_pred]
        elif isinstance(raw_pred, list):
            pred_items = raw_pred
        else:
            failed_count += 1
            continue

        raw_gold = rec.get("gold", [])
        gold_entities: List[Set[str]] = []
        for ent in raw_gold:
            if isinstance(ent, (list, tuple)):
                syns = {normalize_text(s) for s in ent if normalize_text(s)}
            elif isinstance(ent, str):
                s = normalize_text(ent)
                syns = {s} if s else set()
            else:
                syns = set()
            if syns:
                gold_entities.append(syns)

        pred_norm_items = [normalize_text(p) for p in pred_items if normalize_text(p)]

        n_gold = len(gold_entities)
        n_pred = len(pred_norm_items)

        if n_gold == 0 and n_pred == 0:
            p, r, f = 1.0, 1.0, 1.0
        elif n_gold == 0 or n_pred == 0:
            p, r, f = 0.0, 0.0, 0.0
        else:
            matched_gold_indices: Set[int] = set()
            for p_item in pred_norm_items:
                for g_idx, syn_set in enumerate(gold_entities):
                    if g_idx not in matched_gold_indices and p_item in syn_set:
                        matched_gold_indices.add(g_idx)
                        break

            tp = len(matched_gold_indices)
            p = tp / float(n_pred)
            r = tp / float(n_gold)
            f = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0

        answered_scores.append((p, r, f))

    answered_count = len(answered_scores)
    total_count = answered_count + abstained_count + failed_count
    counts = QuestionTypeCounts(
        answered=answered_count,
        abstained=abstained_count,
        failed=failed_count,
        total=total_count,
    )

    if answered_count == 0:
        ans_metrics = ListMetrics(mean_precision=0.0, mean_recall=0.0, mean_f1=0.0)
    else:
        ans_metrics = ListMetrics(
            mean_precision=sum(p for p, r, f in answered_scores) / float(answered_count),
            mean_recall=sum(r for p, r, f in answered_scores) / float(answered_count),
            mean_f1=sum(f for p, r, f in answered_scores) / float(answered_count),
        )

    if total_count == 0:
        all_metrics = ListMetrics(mean_precision=0.0, mean_recall=0.0, mean_f1=0.0)
    else:
        # Abstained and failed contribute 0.0
        all_metrics = ListMetrics(
            mean_precision=sum(p for p, r, f in answered_scores) / float(total_count),
            mean_recall=sum(r for p, r, f in answered_scores) / float(total_count),
            mean_f1=sum(f for p, r, f in answered_scores) / float(total_count),
        )

    return ListScoringResult(
        counts=counts,
        answered_only=ans_metrics,
        all_questions=all_metrics,
    )


def score_summary(records: List[Dict[str, Any]]) -> SummaryScoringResult:
    """Report summary questions as Not Evaluated per SPEC 9.9 and Stage C decision."""
    answered = 0
    abstained = 0
    failed = 0

    for rec in records:
        if rec.get("failed", False):
            failed += 1
        elif rec.get("workflow_abstained", False):
            abstained += 1
        else:
            raw_pred = rec.get("predicted")
            if raw_pred is None:
                failed += 1
            elif isinstance(raw_pred, str) and normalize_text(raw_pred) == "abstain":
                abstained += 1
            else:
                answered += 1

    counts = QuestionTypeCounts(
        answered=answered,
        abstained=abstained,
        failed=failed,
        total=answered + abstained + failed,
    )
    return SummaryScoringResult(
        counts=counts,
    )


def score_retrieval(question_retrievals: List[Dict[str, Any]]) -> RetrievalScoringResult:
    """Compute retrieval metrics at corpus-row level (SPEC A11).

    Steps:
    1. Map each retrieved chunk evidence_id to its corpus row (legacy_id).
    2. Collapse chunks of the same row, keeping the best score.
    3. Rank rows by that score (descending, with stable tie-breaking).
    4. Apply duplicate-group rule: row counts as relevant if any member of its duplicate group is relevant.
    5. Compute Recall@5, Recall@10, Precision@5, MRR, and MAP (AP normalized by total relevant rows R).
    6. Exclude and count questions with no relevant rows.
    """
    included_metrics: List[Dict[str, float]] = []
    excluded_count = 0

    for item in question_retrievals:
        raw_rel_rows = item.get("relevant_row_ids", [])
        rel_row_set = {str(r) for r in raw_rel_rows}

        # Questions with no relevant rows are excluded and counted
        if not rel_row_set:
            excluded_count += 1
            continue

        raw_dup_groups = item.get("duplicate_groups") or {}
        dup_groups: Dict[str, Set[str]] = {
            str(k): {str(v) for v in vs} for k, vs in raw_dup_groups.items()
        }

        # 1. Map each chunk to corpus row (legacy_id) and collapse to best score
        chunks = item.get("retrieved_chunks", [])
        best_row_scores: Dict[str, float] = {}

        for ch in chunks:
            if isinstance(ch, dict):
                ev_id = ch["evidence_id"]
                score = float(ch["retrieval_score"])
            elif isinstance(ch, (list, tuple)):
                ev_id = ch[0]
                score = float(ch[1])
            else:
                ev_id = getattr(ch, "evidence_id")
                score = float(getattr(ch, "retrieval_score"))

            # Extract legacy_id from ev:bioasq:<legacy_id>:<chunk_idx>
            parts = ev_id.split(":")
            if len(parts) >= 3 and parts[0] == "ev":
                row_id = parts[2]
            else:
                row_id = ev_id

            if row_id not in best_row_scores or score > best_row_scores[row_id]:
                best_row_scores[row_id] = score

        # 2. Rank rows by best score descending (stable tie-breaking by row_id)
        ranked_rows = sorted(best_row_scores.keys(), key=lambda r: (-best_row_scores[r], r))

        # 3. Duplicate-group rule
        def is_relevant(r: str) -> bool:
            if r in rel_row_set:
                return True
            if r in dup_groups:
                return any(member in rel_row_set for member in dup_groups[r])
            return False

        r_total = len(rel_row_set)

        # Recall@5, Precision@5
        top5 = ranked_rows[:5]
        rel5 = sum(1 for r in top5 if is_relevant(r))
        rec5 = rel5 / float(r_total)
        prec5 = rel5 / 5.0

        # Recall@10
        top10 = ranked_rows[:10]
        rel10 = sum(1 for r in top10 if is_relevant(r))
        rec10 = rel10 / float(r_total)

        # MRR
        mrr_val = 0.0
        for rank_idx, r in enumerate(ranked_rows, start=1):
            if is_relevant(r):
                mrr_val = 1.0 / float(rank_idx)
                break

        # MAP: Average precision over retrieved row list, normalized by number of relevant rows R
        # [ENGINEERING-DECISION: RETRIEVAL-AP-NORMALIZATION]
        running_rel = 0
        prec_sum = 0.0
        for k, r in enumerate(ranked_rows, start=1):
            if is_relevant(r):
                running_rel += 1
                prec_sum += (running_rel / float(k))
        ap_val = prec_sum / float(r_total)

        included_metrics.append({
            "recall_at_5": rec5,
            "recall_at_10": rec10,
            "precision_at_5": prec5,
            "mrr": mrr_val,
            "map": ap_val,
        })

    num_q = len(included_metrics)
    if num_q == 0:
        mean_rec5 = mean_rec10 = mean_prec5 = mean_mrr = mean_map = 0.0
    else:
        mean_rec5 = sum(m["recall_at_5"] for m in included_metrics) / float(num_q)
        mean_rec10 = sum(m["recall_at_10"] for m in included_metrics) / float(num_q)
        mean_prec5 = sum(m["precision_at_5"] for m in included_metrics) / float(num_q)
        mean_mrr = sum(m["mrr"] for m in included_metrics) / float(num_q)
        mean_map = sum(m["map"] for m in included_metrics) / float(num_q)

    return RetrievalScoringResult(
        recall_at_5=MetricWithCount(value=mean_rec5, num_questions=num_q),
        recall_at_10=MetricWithCount(value=mean_rec10, num_questions=num_q),
        precision_at_5=MetricWithCount(value=mean_prec5, num_questions=num_q),
        mrr=MetricWithCount(value=mean_mrr, num_questions=num_q),
        map=MetricWithCount(value=mean_map, num_questions=num_q),
        included_questions=num_q,
        excluded_no_relevant_rows=excluded_count,
    )


def score_gold_document_citation_rate(
    records: List[Union[Dict[str, Any], Tuple[Set[str], Set[str]]]],
) -> GoldDocumentCitationRateResult:
    """Compute gold document citation rate over answered questions.

    gold_document_citation_rate = (answered questions whose citations include >= 1 gold PMID) / answered questions.
    Pure function over PMID sets.
    """
    hits = 0
    total_answered = len(records)

    for rec in records:
        if isinstance(rec, (tuple, list)):
            cited = {str(p) for p in rec[0]}
            gold = {str(p) for p in rec[1]}
        else:
            cited = {str(p) for p in rec.get("cited_pmids", [])}
            gold = {str(p) for p in rec.get("gold_pmids", [])}

        if gold and bool(cited & gold):
            hits += 1

    rate = (hits / float(total_answered)) if total_answered > 0 else 0.0
    return GoldDocumentCitationRateResult(
        hits=hits,
        total_answered=total_answered,
        rate=rate,
    )
