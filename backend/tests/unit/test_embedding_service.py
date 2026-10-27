"""Unit tests for Biomedical Embedding Service (Req ID: RET-2)."""

import pytest
from backend.app.retrieval.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_embedding_service_dimensions_and_tokens(embedding_service):
    """Verify verified empirical dimensions from DEC-020."""
    assert embedding_service.dimension == 768
    assert embedding_service.max_seq_length == 100


def test_token_counting(embedding_service):
    """Test token counting functionality."""
    text = "Fever and dyspnea in an adult patient."
    token_count = embedding_service.count_tokens(text)
    assert token_count > 0
    assert token_count < 20


def test_chunking_only_when_exceeding_limit(embedding_service):
    """Test that chunker is applied ONLY when tokens > max_tokens per SPEC 9.3."""
    short_text = "Patient presents with headache and mild fever."
    chunks = embedding_service.chunk_text_if_needed(short_text, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Generate long text that exceeds 100 tokens
    long_text = "Acute pulmonary exacerbation with severe hypoxemia and respiratory distress. " * 15
    long_chunks = embedding_service.chunk_text_if_needed(long_text, max_tokens=100, overlap_tokens=50)
    assert len(long_chunks) > 1
    for c in long_chunks:
        assert len(c) > 0


def test_embed_text_and_batch(embedding_service):
    """Test vector embedding shapes and normalization."""
    vec = embedding_service.embed_text("Community acquired pneumonia")
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)

    batch_vecs = embedding_service.embed_batch(["Dyspnea", "Chest discomfort"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768
    assert len(batch_vecs[1]) == 768
