"""Qdrant Client Wrapper.

Specification Section 2, Section 6:
- Managed collection naming: evidence_<model_slug>_v<n>
- Vector metric: Cosine distance
- Payload storage invariant: payload MUST contain ONLY:
  evidence_id, document_id, chunk_index, source, corpus_version, embedding_model.
  Full text lives strictly in PostgreSQL.
"""

import re
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.models import Distance, PointStruct, VectorParams
from backend.app.core.config import settings

ALLOWED_PAYLOAD_KEYS = {
    "evidence_id",
    "document_id",
    "chunk_index",
    "source",
    "corpus_version",
    "embedding_model",
}


def slugify_model_name(model_name: str) -> str:
    """Generate a clean URL/collection-safe slug from a model ID."""
    clean = model_name.split("/")[-1].lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
    return clean


def format_collection_name(model_name: str, version: int) -> str:
    """Format canonical Qdrant collection name: evidence_<model_slug>_v<n>."""
    slug = slugify_model_name(model_name)
    return f"evidence_{slug}_v{version}"


class QdrantWrapper:
    """Wrapper managing Qdrant collection lifecycle and point operations."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            client_url = url or settings.QDRANT_URL
            client_api_key = api_key or settings.QDRANT_API_KEY
            self.client = QdrantClient(url=client_url, api_key=client_api_key)

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Create a collection if it does not already exist."""
        if self.collection_exists(collection_name):
            return False

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        return True

    def validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and filter payload so it strictly contains ONLY the 6 allowed keys.
        
        Allowed keys (SPEC 6):
        evidence_id, document_id, chunk_index, source, corpus_version, embedding_model.
        """
        payload_keys = set(payload.keys())
        disallowed = payload_keys - ALLOWED_PAYLOAD_KEYS
        if disallowed:
            raise ValueError(
                f"Disallowed fields in Qdrant payload: {disallowed}. "
                f"Payload must contain ONLY: {sorted(list(ALLOWED_PAYLOAD_KEYS))}."
            )

        missing = ALLOWED_PAYLOAD_KEYS - payload_keys
        if missing:
            raise ValueError(
                f"Missing required fields in Qdrant payload: {missing}. "
                f"Payload must contain ALL: {sorted(list(ALLOWED_PAYLOAD_KEYS))}."
            )

        return {k: payload[k] for k in ALLOWED_PAYLOAD_KEYS}

    def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> None:
        """Upsert points validating that each point's payload contains only the 6 allowed keys."""
        validated_points: List[PointStruct] = []
        for point in points:
            clean_payload = self.validate_payload(point.payload or {})
            validated_points.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=clean_payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=validated_points,
            wait=True,
        )

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Any]:
        """Perform vector similarity search against the collection."""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
