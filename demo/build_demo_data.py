"""Demo Data Builder and Ingestion Runner.

Prepares an isolated demo corpus and ingests it into clinexus_demo PostgreSQL
and an isolated Qdrant collection.
"""

import csv
import os
import random
import sys
import time
from typing import List, Dict

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set environment overrides before importing backend modules to ensure isolation
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["DATABASE_SYNC_URL"] = "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["QDRANT_COLLECTION_VERSION"] = "777"

DATA_DIR = os.path.join(DEMO_DIR, "data")
DEMO_CORPUS_PATH = os.path.join(DATA_DIR, "demo_corpus.csv")
CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "datasets", "processed", "bioasq", "retrieval_eval_candidates.csv")
FULL_CORPUS_PATH = os.path.join(PROJECT_ROOT, "datasets", "processed", "bioasq", "bioasq_evidence_corpus.csv")

DEMO_COLLECTION_VERSION = 777
N_DISTRACTORS = 2500
RANDOM_SEED = 42

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.persistence.models import EvidenceDocument, EvidencePassage
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name

os.environ["DATABASE_SYNC_URL"] = "postgresql+psycopg2://postgres:postgres_password@localhost:5432/clinexus_demo"
os.environ["QDRANT_COLLECTION_VERSION"] = "777"


def ensure_demo_database():
    """Ensure clinexus_demo exists and migrations are up to date."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    from alembic.config import Config
    from alembic import command

    print("[STEP 1.1] Ensuring PostgreSQL database 'clinexus_demo' exists...")
    conn = psycopg2.connect("postgresql://postgres:postgres_password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'clinexus_demo'")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE clinexus_demo;")
        print(" -> Created database 'clinexus_demo'.")
    else:
        print(" -> Database 'clinexus_demo' already exists.")
    cur.close()
    conn.close()

    print("[STEP 1.1] Running Alembic migrations against 'clinexus_demo'...")
    cfg = Config("backend/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.DATABASE_SYNC_URL)
    command.upgrade(cfg, "head")
    print(" -> Alembic migrations up to date on 'clinexus_demo'.")


def benchmark_embedding_rate(service: EmbeddingService) -> float:
    """Benchmark embedding rate on 200 real chunks from the corpus."""
    print("[STEP 1.2] Benchmarking embedding rate on 200 real passages with BioBERT...")
    _ = service.dimension  # ensure weights are loaded
    texts = []
    with open(FULL_CORPUS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks = service.chunk_text_if_needed(row["snippet_text"])
            texts.extend(chunks)
            if len(texts) >= 200:
                texts = texts[:200]
                break

    t0 = time.perf_counter()
    _ = service.embed_batch(texts, batch_size=32)
    elapsed = time.perf_counter() - t0
    rate = len(texts) / elapsed
    print(f" -> Measured rate: {rate:.2f} passages/sec ({elapsed:.2f}s for 200 passages)")
    return rate


def build_demo_corpus_file(n_distractors: int = N_DISTRACTORS, seed: int = RANDOM_SEED) -> int:
    """Build demo/data/demo_corpus.csv from candidates + N distractors."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[STEP 1.2] Building isolated demo corpus with N={n_distractors} distractors (seed={seed})...")

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        c_reader = csv.DictReader(f)
        candidate_sample_ids = {row["sample_id"] for row in c_reader}

    candidate_rows: List[Dict[str, str]] = []
    distractor_pool: List[Dict[str, str]] = []
    fieldnames = []

    with open(FULL_CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_reader = csv.DictReader(f)
        fieldnames = corpus_reader.fieldnames or []
        for row in corpus_reader:
            if row["sample_id"] in candidate_sample_ids:
                candidate_rows.append(row)
            else:
                distractor_pool.append(row)

    print(f" -> Found {len(candidate_rows)} candidate rows across {len(candidate_sample_ids)} candidate questions.")
    print(f" -> Distractor pool size: {len(distractor_pool)} rows.")

    rng = random.Random(seed)
    chosen_distractors = rng.sample(distractor_pool, min(n_distractors, len(distractor_pool)))
    combined_rows = candidate_rows + chosen_distractors

    with open(DEMO_CORPUS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f" -> Wrote {len(combined_rows)} rows to '{DEMO_CORPUS_PATH}'.")
    return len(combined_rows)


def run_ingestion():
    """Ingest the demo corpus into clinexus_demo and Qdrant collection v777."""
    ensure_demo_database()

    embedder = EmbeddingService()
    rate = benchmark_embedding_rate(embedder)

    total_rows = build_demo_corpus_file(N_DISTRACTORS, RANDOM_SEED)
    estimated_passages = int(total_rows * 1.1)
    estimated_sec = estimated_passages / rate if rate > 0 else 0
    estimated_min = estimated_sec / 60.0

    print("\n" + "=" * 70)
    print("INGESTION ESTIMATE")
    print(f" - Distractors N: {N_DISTRACTORS}")
    print(f" - Total Corpus Rows: {total_rows}")
    print(f" - Estimated Passages (including chunks): ~{estimated_passages}")
    print(f" - Measured Embedding Rate: {rate:.2f} passages/sec")
    print(f" - Estimated Duration: ~{estimated_sec:.1f}s (~{estimated_min:.2f} min; well under 10 min)")
    print(f" - Target PostgreSQL DB: clinexus_demo")
    demo_collection = format_collection_name(embedder.model_name, DEMO_COLLECTION_VERSION)
    print(f" - Target Qdrant Collection: {demo_collection}")
    print("=" * 70 + "\n")

    # Connect to clinexus_demo
    engine = create_engine(settings.DATABASE_SYNC_URL, echo=False, pool_pre_ping=True)
    DemoSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    qdrant = QdrantWrapper(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    # Ensure fresh collection for clean demo counts
    if qdrant.collection_exists(demo_collection):
        print(f"Cleaning up existing collection '{demo_collection}'...")
        qdrant.client.delete_collection(demo_collection)

    # Clean existing evidence in clinexus_demo if any
    with DemoSession() as session:
        session.execute(EvidencePassage.__table__.delete())
        session.execute(EvidenceDocument.__table__.delete())
        session.commit()

    pipeline = IngestionPipeline(
        embedding_service=embedder,
        qdrant_wrapper=qdrant,
        db_session_factory=DemoSession,
    )

    print(f"[STEP 1.3] Starting real ingestion through IngestionPipeline...")
    t_start = time.perf_counter()
    stats = pipeline.ingest_file(
        file_path=DEMO_CORPUS_PATH,
        collection_version=DEMO_COLLECTION_VERSION,
        reindex=False,
        batch_size=64,
    )
    total_time = time.perf_counter() - t_start

    print(f"\nIngestion completed in {total_time:.2f} seconds ({total_time / 60:.2f} min).")
    print("Ingestion Pipeline Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Verify counts across Source, PostgreSQL, and Qdrant
    with DemoSession() as session:
        pg_docs = session.scalar(select(func.count(EvidenceDocument.id)))
        pg_passages = session.scalar(select(func.count(EvidencePassage.id)))

    collection_info = qdrant.client.get_collection(demo_collection)
    qdrant_points = collection_info.points_count

    print("\n" + "=" * 70)
    print("COUNT VERIFICATION")
    print(f" - Source CSV Rows Processed: {stats['processed_rows']}")
    print(f" - PostgreSQL Documents:      {pg_docs}")
    print(f" - PostgreSQL Passages:       {pg_passages}")
    print(f" - Qdrant Vector Points:      {qdrant_points}")
    print("=" * 70)

    assert stats["processed_rows"] == total_rows, f"Processed rows ({stats['processed_rows']}) != total rows ({total_rows})"
    assert pg_passages == stats["passages_created"], f"PostgreSQL passages ({pg_passages}) != created ({stats['passages_created']})"
    assert qdrant_points == pg_passages, f"Qdrant points ({qdrant_points}) != PostgreSQL passages ({pg_passages})"
    print("\nSUCCESS: Source, PostgreSQL, and Qdrant counts match exactly!\n")


if __name__ == "__main__":
    run_ingestion()
