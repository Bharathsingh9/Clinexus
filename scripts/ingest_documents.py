#!/usr/bin/env python3
"""CLI script for ingesting documents into Clinexus.

Specification Section 9.3, Section 12 (Req ID: RET-1, DB-2).
"""

import argparse
import json
import logging
import sys
from backend.app.core.logging_config import setup_logging
from backend.app.ingestion.pipeline import IngestionPipeline

setup_logging()
logger = logging.getLogger("clinexus.ingestion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest biomedical corpus into Postgres and Qdrant.")
    parser.add_argument("--file", type=str, required=True, help="Path to BioASQ corpus CSV file.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing and embedding.")
    parser.add_argument("--reindex", action="store_true", help="Create a new collection version.")
    parser.add_argument("--version", type=int, default=1, help="Collection version number.")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum number of records to ingest.")
    parser.add_argument("--failed-records-path", type=str, default="failed_records.jsonl", help="Path to write failed records JSONL.")

    args = parser.parse_args()

    pipeline = IngestionPipeline()
    logger.info("Starting ingestion from %s...", args.file)

    try:
        stats = pipeline.ingest_file(
            file_path=args.file,
            collection_version=args.version,
            reindex=args.reindex,
            batch_size=args.batch_size,
            max_records=args.max_records,
            failed_records_path=args.failed_records_path,
        )
        print("\n=== Ingestion Summary ===")
        print(json.dumps(stats, indent=2))
        logger.info("Ingestion completed successfully.")
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
