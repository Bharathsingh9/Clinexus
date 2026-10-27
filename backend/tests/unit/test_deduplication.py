"""Unit tests for Evidence Deduplication (Req ID: RET-3)."""

from backend.app.retrieval.deduplication import deduplicate_passages
from backend.app.schemas.evidence import RetrievedPassageResult


def make_passage(ev_id: str, chunk_id: str, text: str, score: float, rank: int):
    return RetrievedPassageResult(
        evidence_id=ev_id,
        document_id="doc-1",
        chunk_id=chunk_id,
        title="[SYNTHETIC] Test Document",
        source="BIOASQ",
        publication_date="2020",
        source_location="abstract",
        retrieval_score=score,
        rank=rank,
        corpus_version="v1",
        embedding_model="biobert",
        query_id="q-1",
        evidence_level="UNKNOWN",
        text=text,
    )


def test_deduplication_by_chunk_id():
    p1 = make_passage("ev:bioasq:1:0", "chunk-1", "Passage text A", 0.85, 1)
    p2 = make_passage("ev:bioasq:1:0", "chunk-1", "Passage text A", 0.70, 2)
    deduped = deduplicate_passages([p1, p2], top_k=5)
    assert len(deduped) == 1
    assert deduped[0].retrieval_score == 0.85
    assert deduped[0].rank == 1


def test_deduplication_by_text_hash():
    p1 = make_passage("ev:bioasq:1:0", "chunk-1", "Identical snippet text", 0.80, 1)
    p2 = make_passage("ev:bioasq:2:0", "chunk-2", "  identical SNIPPET text  ", 0.75, 2)
    deduped = deduplicate_passages([p1, p2], top_k=5)
    assert len(deduped) == 1
    assert deduped[0].evidence_id == "ev:bioasq:1:0"
    assert deduped[0].retrieval_score == 0.80


def test_deduplication_top_k_and_ranking():
    passages = [
        make_passage(f"ev:bioasq:{i}:0", f"chunk-{i}", f"Distinct text {i}", 0.5 + i * 0.05, i + 1)
        for i in range(10)
    ]
    deduped = deduplicate_passages(passages, top_k=3)
    assert len(deduped) == 3
    # Check descending order of scores
    assert deduped[0].retrieval_score >= deduped[1].retrieval_score >= deduped[2].retrieval_score
    # Ranks renumbered 1, 2, 3
    assert [p.rank for p in deduped] == [1, 2, 3]


def test_deduplication_trace_keeps_all_member_evidence_ids():
    """Verify that a deduplicated group preserves all member evidence IDs in duplicate_evidence_ids."""
    p1 = make_passage("ev:bioasq:101:0", "chunk-101", "Identical snippet text for group", 0.92, 1)
    p2 = make_passage("ev:bioasq:102:0", "chunk-102", "identical snippet text for group", 0.81, 2)
    p3 = make_passage("ev:bioasq:103:0", "chunk-103", "  IDENTICAL SNIPPET TEXT FOR GROUP  ", 0.74, 3)

    deduped = deduplicate_passages([p1, p2, p3], top_k=5)
    assert len(deduped) == 1
    rep = deduped[0]
    assert rep.evidence_id == "ev:bioasq:101:0"
    # Must trace all member evidence IDs across the deduplicated group
    assert set(rep.duplicate_evidence_ids) == {
        "ev:bioasq:101:0",
        "ev:bioasq:102:0",
        "ev:bioasq:103:0",
    }

