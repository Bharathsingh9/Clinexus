"""Evidence Deduplication.

Specification Section 4.2 (Req ID: RET-3):
- Deduplicate retrieved passages by chunk ID and text hash.
- Preserve highest ranking and score across duplicate groups.
- Keep trace of duplicate group members.
"""

import hashlib
from typing import Dict, List, Optional
from backend.app.schemas.evidence import RetrievedPassageResult


def deduplicate_passages(
    passages: List[RetrievedPassageResult],
    top_k: int = 10,
) -> List[RetrievedPassageResult]:
    """Deduplicate passages by chunk_id and text hash, preserving highest score and keeping member trace."""
    chunk_to_primary: Dict[str, RetrievedPassageResult] = {}
    hash_to_primary: Dict[str, RetrievedPassageResult] = {}
    deduped: List[RetrievedPassageResult] = []

    # Sort primarily by retrieval score descending
    sorted_passages = sorted(passages, key=lambda p: p.retrieval_score, reverse=True)

    for p in sorted_passages:
        t_hash = hashlib.sha256(p.text.strip().lower().encode("utf-8")).hexdigest()

        primary: Optional[RetrievedPassageResult] = (
            chunk_to_primary.get(p.chunk_id) or hash_to_primary.get(t_hash)
        )

        if primary is not None:
            # Append member evidence ID to duplicate trace if not present
            if p.evidence_id not in primary.duplicate_evidence_ids:
                primary.duplicate_evidence_ids.append(p.evidence_id)
            continue

        # If top_k reached for new distinct passages, skip adding new primaries
        if len(deduped) >= top_k:
            continue

        # New distinct passage group
        if not p.duplicate_evidence_ids:
            p.duplicate_evidence_ids = [p.evidence_id]
        elif p.evidence_id not in p.duplicate_evidence_ids:
            p.duplicate_evidence_ids.insert(0, p.evidence_id)

        chunk_to_primary[p.chunk_id] = p
        hash_to_primary[t_hash] = p
        deduped.append(p)

    # Re-assign ranks 1..N
    for i, p in enumerate(deduped):
        p.rank = i + 1

    return deduped
