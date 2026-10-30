"""Integration test fixtures for Clinexus backend.

Specification Section 2, 6, 9.3 (Req ID: DAT-3, RET-1, WKF-2, EXT-C, EXT-D).
Provides session-scoped synthetic corpus fixture in Qdrant and PostgreSQL.
"""

import os
import pytest
from sqlalchemy import delete

from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import (
    EvidenceDocument as EvidenceDocumentModel,
    EvidencePassage as EvidencePassageModel,
)
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_corpus.csv"
)
TEST_COLLECTION_VERSION = 98


@pytest.fixture(scope="function")
def setup_extended_corpus():
    """Ingest synthetic fixture corpus into test Qdrant collection and clean up afterwards."""
    qdrant = QdrantWrapper()
    embedder = EmbeddingService()
    col_name = format_collection_name(embedder.model_name, TEST_COLLECTION_VERSION)

    # Clean existing records
    with SessionLocal() as session:
        session.execute(delete(EvidencePassageModel).where(EvidencePassageModel.evidence_id.like("ev:bioasq:SYNTH%")))
        session.execute(delete(EvidenceDocumentModel).where(EvidenceDocumentModel.source_id.in_(["111001", "111002", "111003", "111004", "111005"])))
        session.commit()

    if qdrant.collection_exists(col_name):
        qdrant.client.delete_collection(col_name)

    pipeline = IngestionPipeline(
        embedding_service=embedder,
        qdrant_wrapper=qdrant,
        db_session_factory=SessionLocal,
    )
    pipeline.ingest_file(
        file_path=FIXTURE_PATH,
        collection_version=TEST_COLLECTION_VERSION,
        reindex=False,
        batch_size=10,
    )

    yield {
        "qdrant": qdrant,
        "embedder": embedder,
        "collection_version": TEST_COLLECTION_VERSION,
        "collection_name": col_name,
    }

    if qdrant.collection_exists(col_name):
        qdrant.client.delete_collection(col_name)

    with SessionLocal() as session:
        session.execute(delete(EvidencePassageModel).where(EvidencePassageModel.evidence_id.like("ev:bioasq:SYNTH%")))
        session.execute(delete(EvidenceDocumentModel).where(EvidenceDocumentModel.source_id.in_(["111001", "111002", "111003", "111004", "111005"])))
        session.commit()
