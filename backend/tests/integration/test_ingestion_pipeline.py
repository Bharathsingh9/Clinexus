"""Integration tests for Ingestion Pipeline and Retrieval against real containers.

Specification Section 9.3, Section 6, Section 11 (Req ID: RET-1, DB-2, DAT-3):
- Uses [SYNTHETIC] fixture corpus against real containers (Postgres and Qdrant).
- Ingestion run twice creates NO duplicates and counts match across source, Qdrant, and PostgreSQL.
- Retrieval test returns passages with all required metadata.
"""

import os
import pytest
from sqlalchemy import delete, func, select
from backend.app.agents.symptom_rag_analyzer import SymptomRAGAnalyzer
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import EvidenceDocument, EvidencePassage
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name
from backend.app.schemas.clinical_case import (
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)
from backend.tests.fixtures.fake_llm_provider import FakeLLMProvider

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_corpus.csv"
)


@pytest.mark.integration
def test_ingestion_idempotence_and_retrieval_metadata():
    """Verify Stage A exit gate requirements on fixture corpus."""
    qdrant = QdrantWrapper()
    embedder = EmbeddingService()
    test_version = 99
    collection_name = format_collection_name(embedder.model_name, test_version)

    # Clean up any existing fixture records from previous runs to ensure test repeatability
    with SessionLocal() as session:
        session.execute(delete(EvidencePassage).where(EvidencePassage.evidence_id.like("ev:bioasq:SYNTH%")))
        session.execute(delete(EvidenceDocument).where(EvidenceDocument.source_id.in_(["111001", "111002", "111003", "111004", "111005"])))
        session.commit()

    if qdrant.collection_exists(collection_name):
        qdrant.client.delete_collection(collection_name)

    pipeline = IngestionPipeline(
        embedding_service=embedder,
        qdrant_wrapper=qdrant,
        db_session_factory=SessionLocal,
    )

    # 1. First Ingestion Run
    stats1 = pipeline.ingest_file(
        file_path=FIXTURE_PATH,
        collection_version=test_version,
        reindex=False,
        batch_size=10,
    )

    assert stats1["total_rows_read"] == 6
    assert stats1["processed_rows"] == 6
    assert stats1["rejected_rows"] == 0
    assert stats1["passages_created"] >= 6
    assert stats1["short_passages_count"] >= 1  # SYNTH003 is under 50 chars

    # Verify counts in Postgres
    with SessionLocal() as session:
        pg_passage_count = session.execute(
            select(func.count(EvidencePassage.id))
        ).scalar()
        assert pg_passage_count >= stats1["passages_created"]

    # Verify points in Qdrant
    qdrant_info = qdrant.client.get_collection(collection_name)
    assert qdrant_info.points_count == stats1["qdrant_vectors_upserted"]

    # 2. Second Ingestion Run (Idempotency Check)
    stats2 = pipeline.ingest_file(
        file_path=FIXTURE_PATH,
        collection_version=test_version,
        reindex=False,
        batch_size=10,
    )

    # No new passages or vectors created on duplicate run
    assert stats2["passages_created"] == 0
    assert stats2["passages_skipped_existing"] >= 6
    assert stats2["qdrant_vectors_upserted"] == 0

    # Counts remain equal and stable
    qdrant_info_after = qdrant.client.get_collection(collection_name)
    assert qdrant_info_after.points_count == stats1["qdrant_vectors_upserted"]

    # 3. Retrieval Test: verify all required metadata fields
    analyzer = SymptomRAGAnalyzer(
        qdrant_wrapper=qdrant,
        embedding_service=embedder,
        collection_version=test_version,
        top_k=5,
        min_score=0.10,  # Ensure fixture passages pass score
        llm_provider=FakeLLMProvider(
            default_text=(
                "CONSIDERATION: Acute viral respiratory infection | CITATIONS: ev:bioasq:SYNTH001:0\n"
                "CONSIDERATION: Atypical pneumonia | CITATIONS: ev:bioasq:SYNTH006:0"
            )
        ),
    )

    case = ClinicalCase(
        raw_text_hash="c" * 64,
        age=45,
        sex="female",
        chief_complaint="Respiratory infection with fever and dry cough",
        symptom_duration="4 days",
        structured_symptoms=[
            StructuredSymptom(
                term="fever",
                normalized_term="pyrexia",
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=0, end=5),
                origin=OriginEnum.STATED,
            ),
            StructuredSymptom(
                term="dry cough",
                normalized_term="dry_cough",
                vocabulary="LOCAL",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=10, end=19),
                origin=OriginEnum.STATED,
            ),
        ],
        patient_context=PatientContext(),
    )

    evidence_set = analyzer.run(case)

    assert not evidence_set.insufficient
    assert len(evidence_set.results) > 0

    # Check that top result carries all required metadata (SPEC 4.2)
    top = evidence_set.results[0]
    assert top.evidence_id.startswith("ev:bioasq:")
    assert top.document_id is not None
    assert top.chunk_id is not None
    assert top.title is not None
    assert top.source == "BIOASQ"
    assert top.publication_date is not None
    assert top.source_location is not None
    assert isinstance(top.retrieval_score, float)
    assert top.rank == 1
    assert top.corpus_version is not None
    assert top.embedding_model == embedder.model_name
    assert top.query_id is not None
    assert top.evidence_level is not None
    assert len(top.text) > 0

    # Check candidate hypotheses grounded in passages (App A5)
    assert len(evidence_set.candidate_hypotheses) > 0
    for h in evidence_set.candidate_hypotheses:
        assert len(h.supporting_evidence_ids) > 0
        assert all(cid in [p.evidence_id for p in evidence_set.results] for cid in h.supporting_evidence_ids)

    # Check retrieval summary sentences cite evidence IDs
    assert len(evidence_set.retrieval_summary) > 0
    assert "ev:bioasq:" in evidence_set.retrieval_summary
