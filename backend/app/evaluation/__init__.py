"""BioASQ Evaluation Module.

Specification Section 9.4, 9.5, 9.6, 9.7 (Req ID: DAT-4, DAT-5, EVL-3, EVL-5).
"""

from backend.app.evaluation.bioasq_questions import (
    build_derived_relevance_mapping,
    compute_coverage_report,
    extract_pmids_from_source_reference,
    parse_bioasq_questions,
    persist_evaluation_labels,
    validate_exact_answer_shape,
    validate_ideal_answer,
)
from backend.app.evaluation.bioasq_scoring import (
    FactoidMetrics,
    FactoidScoringResult,
    GoldDocumentCitationRateResult,
    ListMetrics,
    ListScoringResult,
    MetricWithCount,
    QuestionTypeCounts,
    RetrievalScoringResult,
    SummaryScoringResult,
    YesNoMetrics,
    YesNoScoringResult,
    normalize_text,
    score_factoid,
    score_gold_document_citation_rate,
    score_list,
    score_retrieval,
    score_summary,
    score_yesno,
)
from backend.app.evaluation.splits import (
    build_e2e_sample,
    build_smoke30_sample,
    create_deterministic_splits,
    write_splits_file,
)

__all__ = [
    "parse_bioasq_questions",
    "validate_exact_answer_shape",
    "validate_ideal_answer",
    "extract_pmids_from_source_reference",
    "build_derived_relevance_mapping",
    "compute_coverage_report",
    "persist_evaluation_labels",
    "create_deterministic_splits",
    "build_smoke30_sample",
    "build_e2e_sample",
    "write_splits_file",
    "normalize_text",
    "score_yesno",
    "score_factoid",
    "score_list",
    "score_summary",
    "score_retrieval",
    "score_gold_document_citation_rate",
    "QuestionTypeCounts",
    "YesNoMetrics",
    "YesNoScoringResult",
    "FactoidMetrics",
    "FactoidScoringResult",
    "ListMetrics",
    "ListScoringResult",
    "SummaryScoringResult",
    "RetrievalScoringResult",
    "GoldDocumentCitationRateResult",
]
