"""Real Dense Vector Retrieval Demo.

Demonstrates real vector retrieval over the isolated demo corpus using
BioBERT embeddings, Qdrant vector database, and PostgreSQL snippet metadata.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure isolated demo database and collection version
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["DATABASE_SYNC_URL"] = "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["QDRANT_COLLECTION_VERSION"] = "777"

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.persistence.models import EvidenceDocument, EvidencePassage
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "datasets", "processed", "bioasq", "retrieval_eval_candidates.csv")
DEMO_COLLECTION_VERSION = 777

HONESTY_BANNER = """========================================================================================
HONESTY DISCLOSURE:
REAL: embeddings, Qdrant retrieval, PostgreSQL, workflow orchestration, deterministic verifier checks, safety gate, provenance, audit.
SCRIPTED: language-model outputs and the verifier's entailment judgment (demo stand-in, no real LLM connected).
========================================================================================"""

BENCHMARK_LABELS = """[BENCHMARK MODE, NOT CLINICAL USE]
[CLOSED-WORLD SNIPPET CORPUS]
[Status: illustrative, subset corpus, unconfirmed labels, not an evaluation result]"""


def load_candidate_questions() -> Dict[str, Dict[str, str]]:
    """Load 30 smoke candidate questions indexed by eval_id."""
    candidates = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates[row["eval_id"].strip()] = row
    return candidates


def fetch_postgres_metadata(passage_evidence_ids: List[str], sync_session_factory) -> Dict[str, dict]:
    """Fetch text and document source_id (PMID) from PostgreSQL."""
    if not passage_evidence_ids:
        return {}
    with sync_session_factory() as session:
        stmt = (
            select(
                EvidencePassage.evidence_id,
                EvidencePassage.text,
                EvidencePassage.chunk_index,
                EvidenceDocument.source_id,
                EvidenceDocument.title,
            )
            .join(EvidenceDocument, EvidencePassage.document_id == EvidenceDocument.id)
            .where(EvidencePassage.evidence_id.in_(passage_evidence_ids))
        )
        rows = session.execute(stmt).fetchall()
        return {
            r.evidence_id: {
                "passage_text": r.text,
                "pmid": r.source_id,
                "title": r.title or "",
                "section": "abstract",
                "chunk_index": r.chunk_index,
            }
            for r in rows
        }


def run_retrieval_for_question(
    eval_id: str,
    question_text: str,
    question_type: str,
    embedder: EmbeddingService,
    qdrant: QdrantWrapper,
    sync_session_factory,
    top_k: int = 5,
):
    """Retrieve and display top passages for a given question using real BioBERT and Qdrant."""
    collection_name = format_collection_name(embedder.model_name, DEMO_COLLECTION_VERSION)

    print("\n" + "-" * 88)
    print(f"QUERY [{eval_id}] ({question_type.upper()}): {question_text}")
    print("-" * 88)

    # 1. Real BioBERT dense query embedding
    q_vec = embedder.embed_text(question_text)

    # 2. Real Qdrant vector similarity search
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=q_vec,
        limit=top_k,
    )

    if not hits:
        print(" -> No passages retrieved above score threshold.")
        return

    # Extract payloads and trace info
    first_payload = hits[0].payload or {}
    corpus_version = first_payload.get("corpus_version", "unknown")
    embedding_model = embedder.model_name

    # Print Trace Fields
    print(f"Retrieval Trace Fields:")
    print(f"  * Query Text:       \"{question_text}\"")
    print(f"  * Embedding Model:  {embedding_model}")
    print(f"  * Qdrant Target:    {collection_name}")
    print(f"  * Corpus Version:   {corpus_version}")
    print(f"  * Metric:           Cosine Distance (higher score = greater similarity)")
    print()

    # 3. Real PostgreSQL passage lookup
    hit_ids = [hit.payload.get("evidence_id") for hit in hits if hit.payload]
    pg_metadata = fetch_postgres_metadata(hit_ids, sync_session_factory)

    print(f"Top {len(hits)} Retrieved Passages (illustrative examples only):")
    for rank, hit in enumerate(hits, start=1):
        ev_id = hit.payload.get("evidence_id") if hit.payload else "unknown"
        meta = pg_metadata.get(ev_id, {})
        pmid = meta.get("pmid", "N/A")
        section = meta.get("section", "abstract")
        text = meta.get("passage_text", "").replace("\n", " ").strip()
        preview = text[:130] + ("..." if len(text) > 130 else "")

        print(f"  [{rank}] Score: {hit.score:.4f} | Evidence ID: {ev_id} | PMID: {pmid} | Section: {section}")
        print(f"      Text: \"{preview}\"")


def main():
    parser = argparse.ArgumentParser(description="Clinexus Real Dense Vector Retrieval Demo")
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Optional specific question ID from smoke30 (e.g. Q007, Q016, Q028)",
    )
    args = parser.parse_args()

    print(HONESTY_BANNER)
    print(BENCHMARK_LABELS)
    print()

    candidates = load_candidate_questions()

    embedder = EmbeddingService()
    qdrant = QdrantWrapper(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    engine = create_engine(settings.DATABASE_SYNC_URL, echo=False, pool_pre_ping=True)
    DemoSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    if args.question_id:
        qid = args.question_id.strip()
        if qid not in candidates:
            print(f"Error: Question ID '{qid}' not found in {CANDIDATES_PATH}.")
            print(f"Available IDs: {', '.join(sorted(candidates.keys()))}")
            sys.exit(1)
        item = candidates[qid]
        run_retrieval_for_question(
            eval_id=item["eval_id"],
            question_text=item["question"],
            question_type=item["question_type"],
            embedder=embedder,
            qdrant=qdrant,
            sync_session_factory=DemoSession,
            top_k=5,
        )
    else:
        # Default: 3 contrasting examples
        contrasting_ids = ["Q007", "Q016", "Q028"]
        print(f"Running retrieval on 3 contrasting candidate examples: {', '.join(contrasting_ids)}")
        for qid in contrasting_ids:
            item = candidates[qid]
            run_retrieval_for_question(
                eval_id=item["eval_id"],
                question_text=item["question"],
                question_type=item["question_type"],
                embedder=embedder,
                qdrant=qdrant,
                sync_session_factory=DemoSession,
                top_k=5,
            )

    print("\n" + "=" * 88)
    print("END OF RETRIEVAL DEMO")
    print("Note: Retrieval output is shown as examples only, labeled illustrative, subset corpus,")
    print("unconfirmed labels, not an evaluation result.")
    print("=" * 88)


if __name__ == "__main__":
    main()
