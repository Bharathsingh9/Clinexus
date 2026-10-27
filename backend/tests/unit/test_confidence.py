"""Unit tests for Confidence Calculation Module (Req ID: AGT-4, App A1).

Specification Section 2, Section 4.4, Appendix A1:
- Hand-computed ground truth verification.
- Full system weights (0.50, 0.20, 0.15, 0.15).
- Baseline variant weights (0.40, 0.30, 0.30).
- Insufficient passages check (retrieval_strength forced to 0.0).
- Normalization bounds: RETRIEVAL_SCORE_FLOOR=0.30, RETRIEVAL_SCORE_CEILING=0.90.
- Clamping between 0.0 and 1.0.
- Never uses LLM self-reported confidence.
"""

import pytest
from backend.app.safety.confidence import compute_confidence


def test_confidence_full_weights_hand_computed():
    """Verify full formula with exact hand-computed values:
    claim_support = 3.5 / 4 = 0.875
    retrieval_strength = ((0.85+0.75+0.65)/3 - 0.30) / (0.90 - 0.30) = (0.75 - 0.30) / 0.60 = 0.75
    source_availability = 1.0 - (0 / 2) = 1.0
    info_completeness = 1.0 - (1 / 4) = 0.75
    confidence = 0.50 * 0.875 + 0.20 * 0.75 + 0.15 * 1.0 + 0.15 * 0.75
               = 0.4375 + 0.1500 + 0.1500 + 0.1125 = 0.8500
    """
    res = compute_confidence(
        retrieval_scores=[0.85, 0.75, 0.65],
        claim_support_score=0.875,
        unavailable_sources=0,
        configured_sources=2,
        missing_case_fields=1,
        total_case_fields=4,
        is_baseline=False,
        score_floor=0.30,
        score_ceiling=0.90,
        min_relevant_passages=2,
        min_score_threshold=0.30,
    )

    assert res.claim_support == 0.875
    assert res.retrieval_strength == 0.75
    assert res.source_availability == 1.0
    assert res.info_completeness == 0.75
    assert res.confidence_score == 0.85
    assert res.is_baseline is False


def test_confidence_baseline_variant_hand_computed():
    """Verify baseline formula (App A1: 0.40 retrieval, 0.30 availability, 0.30 completeness):
    retrieval_strength = (0.60 - 0.30) / (0.90 - 0.30) = 0.30 / 0.60 = 0.50
    source_availability = 1.0 - (1 / 2) = 0.50
    info_completeness = 1.0 - (0 / 4) = 1.0
    confidence = 0.40 * 0.50 + 0.30 * 0.50 + 0.30 * 1.0
               = 0.20 + 0.15 + 0.30 = 0.6500
    """
    res = compute_confidence(
        retrieval_scores=[0.60, 0.60],
        claim_support_score=None,
        unavailable_sources=1,
        configured_sources=2,
        missing_case_fields=0,
        total_case_fields=4,
        is_baseline=True,
        score_floor=0.30,
        score_ceiling=0.90,
        min_relevant_passages=2,
        min_score_threshold=0.30,
    )

    assert res.claim_support is None
    assert res.retrieval_strength == 0.50
    assert res.source_availability == 0.50
    assert res.info_completeness == 1.0
    assert res.confidence_score == 0.65
    assert res.is_baseline is True


def test_confidence_insufficient_passages_drops_retrieval_strength_to_zero():
    """When fewer than MIN_RELEVANT_PASSAGES pass threshold, retrieval_strength MUST be 0.0."""
    res = compute_confidence(
        retrieval_scores=[0.85],  # Only 1 passage >= 0.30, but min_relevant_passages=2
        claim_support_score=1.0,
        unavailable_sources=0,
        configured_sources=2,
        missing_case_fields=0,
        total_case_fields=4,
        is_baseline=False,
        min_relevant_passages=2,
        min_score_threshold=0.30,
    )

    assert res.retrieval_strength == 0.0
    # confidence = 0.50 * 1.0 + 0.20 * 0.0 + 0.15 * 1.0 + 0.15 * 1.0 = 0.80
    assert res.confidence_score == 0.80


def test_confidence_clamping():
    """Verify scores above ceiling or below floor clamp cleanly between 0.0 and 1.0."""
    res_high = compute_confidence(
        retrieval_scores=[0.98, 0.95, 0.92],  # Above ceiling 0.90
        claim_support_score=1.5,  # Exceeds 1.0
        unavailable_sources=-1,   # Negative
        configured_sources=2,
        missing_case_fields=-2,
        total_case_fields=4,
        score_floor=0.30,
        score_ceiling=0.90,
    )
    assert res_high.retrieval_strength == 1.0
    assert res_high.claim_support == 1.0
    assert res_high.source_availability == 1.0
    assert res_high.info_completeness == 1.0
    assert res_high.confidence_score == 1.0

    res_low = compute_confidence(
        retrieval_scores=[0.10, 0.15],  # Below threshold
        claim_support_score=0.0,
        unavailable_sources=5,  # Exceeds configured
        configured_sources=2,
        missing_case_fields=10,
        total_case_fields=4,
        score_floor=0.30,
        score_ceiling=0.90,
    )
    assert res_low.retrieval_strength == 0.0
    assert res_low.source_availability == 0.0
    assert res_low.info_completeness == 0.0
    assert res_low.confidence_score == 0.0
