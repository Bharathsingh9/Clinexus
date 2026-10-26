"""Integration Tests for Qdrant Vector Database Wrapper.

Specification Section 2, Section 6.
Runs against the real Qdrant container (must FAIL, not skip, if unreachable).
"""

import uuid
import pytest
from qdrant_client.models import PointStruct
from backend.app.core.config import settings
from backend.app.retrieval.qdrant_client import (
    ALLOWED_PAYLOAD_KEYS,
    QdrantWrapper,
    format_collection_name,
)


@pytest.mark.integration
def test_qdrant_collection_lifecycle_and_point_operations():
    """Verify Qdrant wrapper creates collection, upserts points with 6-field payload, searches, and cleans up."""
    try:
        wrapper = QdrantWrapper(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        # Check initial connectivity; fail test if service is down
        wrapper.client.get_collections()
    except Exception as e:
        pytest.fail(f"Qdrant container unreachable or failed connection: {e}")

    test_version = 999
    collection_name = format_collection_name(settings.EMBEDDING_MODEL, test_version)

    # Ensure clean starting state
    if wrapper.collection_exists(collection_name):
        wrapper.client.delete_collection(collection_name)

    try:
        # 1. Create collection
        created = wrapper.create_collection(collection_name=collection_name, vector_size=4)
        assert created is True
        assert wrapper.collection_exists(collection_name) is True

        # 2. Upsert points containing ONLY the six allowed fields
        point_id = str(uuid.uuid4())
        point_vector = [0.1, 0.2, 0.3, 0.4]
        payload = {
            "evidence_id": "ev:pubmed:99999999:0",
            "document_id": "doc-test-999",
            "chunk_index": 0,
            "source": "PUBMED",
            "corpus_version": "v1.0",
            "embedding_model": settings.EMBEDDING_MODEL,
        }
        # Invariant check: payload has exactly the 6 allowed keys
        assert set(payload.keys()) == ALLOWED_PAYLOAD_KEYS

        point = PointStruct(id=point_id, vector=point_vector, payload=payload)
        wrapper.upsert_points(collection_name=collection_name, points=[point])

        # 3. Search collection and verify hit
        results = wrapper.search(collection_name=collection_name, query_vector=[0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(results) >= 1
        top_hit = results[0]
        assert top_hit.id == point_id
        assert top_hit.payload["evidence_id"] == "ev:pubmed:99999999:0"
        assert set(top_hit.payload.keys()) == ALLOWED_PAYLOAD_KEYS

    finally:
        # Teardown: delete test collection
        if wrapper.collection_exists(collection_name):
            wrapper.client.delete_collection(collection_name)
        assert wrapper.collection_exists(collection_name) is False
