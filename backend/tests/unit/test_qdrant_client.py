"""Unit Tests for Qdrant Client Wrapper and Payload Invariants.

Specification Section 2, Section 6.
Asserts that:
- Collections are formatted as evidence_<model_slug>_v<n>.
- Point payload validation permits ONLY the six specified fields:
  evidence_id, document_id, chunk_index, source, corpus_version, embedding_model.
- Any extraneous or missing fields raise ValueError.
"""

import pytest
from qdrant_client.models import PointStruct
from backend.app.retrieval.qdrant_client import (
    ALLOWED_PAYLOAD_KEYS,
    QdrantWrapper,
    format_collection_name,
    slugify_model_name,
)


def test_collection_name_formatting():
    """Verify canonical naming: evidence_<model_slug>_v<n>."""
    model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    slug = slugify_model_name(model_name)
    assert slug == "biobert_mnli_snli_scinli_scitail_mednli_stsb"

    collection_name = format_collection_name(model_name, 1)
    assert collection_name == "evidence_biobert_mnli_snli_scinli_scitail_mednli_stsb_v1"


def test_qdrant_payload_validation_success():
    """Verify payload with exactly the 6 allowed fields passes validation."""
    valid_payload = {
        "evidence_id": "ev:pubmed:12345:0",
        "document_id": "doc-001",
        "chunk_index": 0,
        "source": "PUBMED",
        "corpus_version": "v1.0",
        "embedding_model": "pritamdeka/BioBERT",
    }
    wrapper = QdrantWrapper(url="http://localhost:6333")
    cleaned = wrapper.validate_payload(valid_payload)
    assert set(cleaned.keys()) == ALLOWED_PAYLOAD_KEYS


def test_qdrant_payload_rejects_extraneous_fields():
    """Invariant: Text or extra metadata must NOT be stored in Qdrant (lives in Postgres)."""
    payload_with_text = {
        "evidence_id": "ev:pubmed:12345:0",
        "document_id": "doc-001",
        "chunk_index": 0,
        "source": "PUBMED",
        "corpus_version": "v1.0",
        "embedding_model": "pritamdeka/BioBERT",
        "text": "Full passage text that should not be in Qdrant",
    }
    wrapper = QdrantWrapper(url="http://localhost:6333")
    with pytest.raises(ValueError, match="Disallowed fields in Qdrant payload"):
        wrapper.validate_payload(payload_with_text)


def test_qdrant_payload_rejects_missing_fields():
    """Verify missing required fields raise ValueError."""
    incomplete_payload = {
        "evidence_id": "ev:pubmed:12345:0",
        "document_id": "doc-001",
        # chunk_index missing
        "source": "PUBMED",
        "corpus_version": "v1.0",
        "embedding_model": "pritamdeka/BioBERT",
    }
    wrapper = QdrantWrapper(url="http://localhost:6333")
    with pytest.raises(ValueError, match="Missing required fields in Qdrant payload"):
        wrapper.validate_payload(incomplete_payload)
