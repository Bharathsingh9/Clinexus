"""Retrieval Insufficient Signal Guard.

Specification Section 2, Section 4.2 (Req ID: RET-5):
- Evaluates whether retrieved passages meet relevance thresholds.
- Flags insufficient=True with exact reason codes:
  * VECTOR_DB_UNAVAILABLE
  * EMPTY_CORPUS
  * LOW_SIMILARITY / FEWER_THAN_MIN_PASSAGES
"""

from typing import List, Optional, Tuple
from backend.app.core.config import settings
from backend.app.schemas.evidence import RetrievedPassageResult


def evaluate_retrieval_sufficiency(
    passages: List[RetrievedPassageResult],
    min_score: Optional[float] = None,
    min_passages: Optional[int] = None,
    collection_empty: bool = False,
    vector_db_error: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Determine whether retrieval produced sufficient evidence signal.
    
    Returns (insufficient: bool, reason_code: Optional[str]).
    """
    if vector_db_error:
        return True, "VECTOR_DB_UNAVAILABLE"

    if collection_empty:
        return True, "EMPTY_CORPUS"

    m_score = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
    m_passages = min_passages if min_passages is not None else settings.MIN_RELEVANT_PASSAGES

    relevant = [p for p in passages if p.retrieval_score >= m_score]
    if len(relevant) < m_passages:
        return True, "FEWER_THAN_MIN_RELEVANT_PASSAGES"

    return False, None
