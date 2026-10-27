"""Ingestion Pipeline Service.

Specification Section 9.3, Section 6, Section 12 (Req ID: RET-1, DB-2):
- Streams and processes corpus CSV in batches.
- Idempotent by content hash and evidence ID.
- Resumable: skips already persisted passages/vectors.
- Re-index flag: creates a fresh Qdrant collection version.
- Failed-record JSONL logger.
- Invariants: text lives strictly in PostgreSQL; Qdrant has only the 6-field payload.
"""

import hashlib
import json
import logging
import os
import uuid
from typing import Dict, List, Optional
from qdrant_client.models import PointStruct
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.core.config import settings
from backend.app.ingestion.bioasq_loader import process_corpus_row, stream_bioasq_csv
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import EvidenceDocument, EvidencePassage
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file efficiently in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class IngestionPipeline:
    """Manages idempotent streaming ingestion of BioASQ snippets into Postgres and Qdrant."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_wrapper: Optional[QdrantWrapper] = None,
        db_session_factory=SessionLocal,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_wrapper = qdrant_wrapper or QdrantWrapper()
        self.db_session_factory = db_session_factory

    def get_or_create_collection(
        self,
        collection_version: int = 1,
        reindex: bool = False,
    ) -> str:
        """Resolve or create target Qdrant collection."""
        model_name = self.embedding_service.model_name
        collection_name = format_collection_name(model_name, collection_version)

        if reindex and self.qdrant_wrapper.collection_exists(collection_name):
            # Bump collection version if reindexing
            collection_version += 1
            collection_name = format_collection_name(model_name, collection_version)

        if not self.qdrant_wrapper.collection_exists(collection_name):
            self.qdrant_wrapper.create_collection(
                collection_name=collection_name,
                vector_size=self.embedding_service.dimension,
            )
            logger.info("Created Qdrant collection: %s", collection_name)

        return collection_name

    def ingest_file(
        self,
        file_path: str,
        collection_version: int = 1,
        reindex: bool = False,
        batch_size: int = 64,
        max_records: Optional[int] = None,
        failed_records_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """Ingest a BioASQ corpus CSV file.
        
        Returns summary dictionary of processing metrics.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus file not found: {file_path}")

        file_sha256 = compute_file_sha256(file_path)
        model_name = self.embedding_service.model_name
        corpus_version = f"{file_sha256}_{model_name}_v{collection_version}"

        collection_name = self.get_or_create_collection(
            collection_version=collection_version,
            reindex=reindex,
        )

        failed_file = None
        if failed_records_path:
            failed_file = open(failed_records_path, "a", encoding="utf-8")

        stats = {
            "total_rows_read": 0,
            "processed_rows": 0,
            "rejected_rows": 0,
            "documents_created": 0,
            "passages_created": 0,
            "passages_skipped_existing": 0,
            "short_passages_count": 0,
            "qdrant_vectors_upserted": 0,
        }

        try:
            for row_batch in stream_bioasq_csv(file_path, chunk_size=batch_size):
                if max_records and stats["total_rows_read"] >= max_records:
                    break

                with self.db_session_factory() as session:
                    points_to_upsert: List[PointStruct] = []
                    texts_to_embed: List[str] = []
                    passage_metadata_list: List[dict] = []

                    for row in row_batch:
                        if max_records and stats["total_rows_read"] >= max_records:
                            break
                        stats["total_rows_read"] += 1

                        record, chunks, error = process_corpus_row(
                            row=row,
                            embedding_service=self.embedding_service,
                        )

                        if error or not record:
                            stats["rejected_rows"] += 1
                            if failed_file:
                                failed_file.write(
                                    json.dumps({"row": row, "reason": error}) + "\n"
                                )
                            continue

                        stats["processed_rows"] += 1

                        # Resolve or create Document by (source, source_id)
                        doc = session.execute(
                            select(EvidenceDocument).where(
                                EvidenceDocument.source == "BIOASQ",
                                EvidenceDocument.source_id == record.pmid,
                            )
                        ).scalar_one_or_none()

                        if doc is None:
                            doc = EvidenceDocument(
                                id=str(uuid.uuid4()),
                                source="BIOASQ",
                                source_id=record.pmid,
                                pmid=record.pmid,
                                url=record.pubmed_url,
                                title="UNKNOWN",
                                authors="UNKNOWN",
                                journal="UNKNOWN",
                                publication_date="UNKNOWN",
                                evidence_level="UNKNOWN",
                            )
                            session.add(doc)
                            session.flush()
                            stats["documents_created"] += 1

                        # Process passage chunks
                        for ev_id, chunk_idx, c_text, c_hash, is_short, t_count in chunks:
                            if is_short:
                                stats["short_passages_count"] += 1

                            # Idempotency check: see if evidence_id already exists in Postgres
                            existing_p = session.execute(
                                select(EvidencePassage).where(
                                    EvidencePassage.evidence_id == ev_id
                                )
                            ).scalar_one_or_none()

                            if existing_p is not None:
                                stats["passages_skipped_existing"] += 1
                                continue

                            p_id = str(uuid.uuid4())
                            new_p = EvidencePassage(
                                id=p_id,
                                document_id=doc.id,
                                evidence_id=ev_id,
                                chunk_index=chunk_idx,
                                text=c_text,
                                text_hash=c_hash,
                                short_passage=is_short,
                                token_count=t_count,
                            )
                            session.add(new_p)
                            stats["passages_created"] += 1

                            texts_to_embed.append(c_text)
                            passage_metadata_list.append(
                                {
                                    "id": p_id,
                                    "evidence_id": ev_id,
                                    "document_id": doc.id,
                                    "chunk_index": chunk_idx,
                                }
                            )

                    session.commit()

                    # Embed and upsert points into Qdrant in batch
                    if texts_to_embed:
                        embeddings = self.embedding_service.embed_batch(
                            texts_to_embed,
                            batch_size=settings.EMBED_BATCH_SIZE,
                        )

                        for meta, emb in zip(passage_metadata_list, embeddings):
                            q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["evidence_id"]))
                            payload = {
                                "evidence_id": meta["evidence_id"],
                                "document_id": meta["document_id"],
                                "chunk_index": meta["chunk_index"],
                                "source": "BIOASQ",
                                "corpus_version": corpus_version,
                                "embedding_model": model_name,
                            }
                            points_to_upsert.append(
                                PointStruct(
                                    id=q_id,
                                    vector=emb,
                                    payload=payload,
                                )
                            )

                        self.qdrant_wrapper.upsert_points(
                            collection_name=collection_name,
                            points=points_to_upsert,
                        )
                        stats["qdrant_vectors_upserted"] += len(points_to_upsert)

        finally:
            if failed_file:
                failed_file.close()

        return stats
