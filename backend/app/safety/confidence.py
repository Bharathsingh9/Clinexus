"""Pure Function Confidence Module.

Specification Section 2, Section 4.4, Appendix A1 (Req ID: AGT-4, App A1):
- Tagged [ENGINEERING-DECISION: CONF-FORMULA], [PAPER-GAP].
- Full system formula:
    confidence = 0.50 * claim_support + 0.20 * retrieval_strength + 0.15 * source_availability + 0.15 * info_completeness
- Baseline variant (omits claim_support per App A1):
    confidence = 0.40 * retrieval_strength + 0.30 * source_availability + 0.30 * info_completeness
- retrieval_strength: mean of top 3 retrieval scores normalized to [floor, ceiling],
  or 0.0 when fewer than MIN_RELEVANT_PASSAGES pass RETRIEVAL_MIN_SCORE.
  Uncalibrated lower and upper bounds: RETRIEVAL_SCORE_FLOOR=0.30, RETRIEVAL_SCORE_CEILING=0.90.
  NOTE: retrieval_strength is an uncalibrated engineering score, NEVER described as validated or calibrated.
- source_availability: 1 - (unavailable_sources / configured_sources)
- info_completeness: 1 - (missing_required_case_fields / total_required_case_fields)
- claim_support: (SUPPORTED + 0.5 * PARTIALLY_SUPPORTED) / total_material_claims
- LLM self-reported confidence is NEVER used.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from backend.app.core.config import settings


@dataclass(frozen=True)
class ConfidenceResult:
    """Breakdown of confidence calculation components."""
    confidence_score: float
    retrieval_strength: float
    source_availability: float
    info_completeness: float
    claim_support: Optional[float]
    is_baseline: bool
    weights: Dict[str, float]


def compute_confidence(
    retrieval_scores: List[float],
    claim_support_score: Optional[float] = None,
    unavailable_sources: int = 0,
    configured_sources: int = 2,
    missing_case_fields: int = 0,
    total_case_fields: int = 4,
    is_baseline: bool = False,
    score_floor: Optional[float] = None,
    score_ceiling: Optional[float] = None,
    min_relevant_passages: Optional[int] = None,
    min_score_threshold: Optional[float] = None,
) -> ConfidenceResult:
    """Pure function calculating confidence score adhering strictly to SPEC Section 2 and App A1."""
    floor = score_floor if score_floor is not None else settings.RETRIEVAL_SCORE_FLOOR
    ceiling = score_ceiling if score_ceiling is not None else settings.RETRIEVAL_SCORE_CEILING
    min_passages = min_relevant_passages if min_relevant_passages is not None else settings.MIN_RELEVANT_PASSAGES
    min_threshold = min_score_threshold if min_score_threshold is not None else settings.RETRIEVAL_MIN_SCORE

    # 1. Retrieval Strength Calculation
    valid_scores = [s for s in retrieval_scores if s >= min_threshold]
    if len(valid_scores) < min_passages:
        retrieval_strength = 0.0
    else:
        top_3 = sorted(retrieval_scores, reverse=True)[:3]
        mean_top_3 = sum(top_3) / len(top_3)
        if ceiling <= floor:
            raw_scaled = 1.0 if mean_top_3 >= floor else 0.0
        else:
            raw_scaled = (mean_top_3 - floor) / (ceiling - floor)
        retrieval_strength = max(0.0, min(1.0, float(raw_scaled)))

    # 2. Source Availability Calculation
    cfg_src = max(1, configured_sources)
    unavail = max(0, min(cfg_src, unavailable_sources))
    source_availability = max(0.0, min(1.0, 1.0 - (unavail / cfg_src)))

    # 3. Patient Information Completeness Calculation
    tot_fields = max(1, total_case_fields)
    miss_fields = max(0, min(tot_fields, missing_case_fields))
    info_completeness = max(0.0, min(1.0, 1.0 - (miss_fields / tot_fields)))

    # 4. Weighting and Synthesis
    if is_baseline or claim_support_score is None:
        # Baseline variant: 0.40 retrieval, 0.30 availability, 0.30 completeness (SPEC App A1)
        w_retrieval = 0.40
        w_availability = 0.30
        w_completeness = 0.30
        confidence = (
            w_retrieval * retrieval_strength
            + w_availability * source_availability
            + w_completeness * info_completeness
        )
        weights = {
            "retrieval_strength": w_retrieval,
            "source_availability": w_availability,
            "info_completeness": w_completeness,
        }
        claim_support = None
    else:
        # Full system formula: 0.50 claim, 0.20 retrieval, 0.15 availability, 0.15 completeness
        w_claim = settings.CONF_WEIGHT_CLAIM_SUPPORT
        w_retrieval = settings.CONF_WEIGHT_RETRIEVAL_STRENGTH
        w_availability = settings.CONF_WEIGHT_SOURCE_AVAILABILITY
        w_completeness = settings.CONF_WEIGHT_INFO_COMPLETENESS

        claim_support = max(0.0, min(1.0, float(claim_support_score)))
        confidence = (
            w_claim * claim_support
            + w_retrieval * retrieval_strength
            + w_availability * source_availability
            + w_completeness * info_completeness
        )
        weights = {
            "claim_support": w_claim,
            "retrieval_strength": w_retrieval,
            "source_availability": w_availability,
            "info_completeness": w_completeness,
        }

    return ConfidenceResult(
        confidence_score=round(float(confidence), 6),
        retrieval_strength=round(retrieval_strength, 6),
        source_availability=round(source_availability, 6),
        info_completeness=round(info_completeness, 6),
        claim_support=round(claim_support, 6) if claim_support is not None else None,
        is_baseline=is_baseline or (claim_support_score is None),
        weights=weights,
    )
