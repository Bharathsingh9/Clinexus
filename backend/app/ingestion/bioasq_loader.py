r"""BioASQ Corpus Loader & Record Generator.

Specification Section 9.3, Section 6, Appendix A6 (Req ID: DAT-3, RET-1):
- Streaming and batched reading with strict CSV parsing rules (utf-8, utf-8-sig).
- PMID extraction regex: (?:^|/)(\d{1,9})/?$
- Platform evidence ID: ev:bioasq:<legacy_id>:<chunk_index>
- Chunker applied ONLY to rows > 100 tokens (DEC-020 verified limit).
- short_passage flag: True if len(text) < 50 characters.
"""

import csv
import hashlib
import io
import re
from typing import Dict, Generator, List, Optional, Tuple
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.schemas.ids import build_evidence_id

PMID_PATTERN = re.compile(r"(?:^|/)(\d{1,9})/?$")


def extract_pmid_from_url(url: str) -> Optional[str]:
    """Extract PubMed ID using SPEC Section 9.3 regex."""
    clean_url = url.strip()
    match = PMID_PATTERN.search(clean_url)
    if match:
        return match.group(1)
    return None


class BioASQCorpusRecord:
    """Internal normalized representation of a single corpus snippet."""

    def __init__(
        self,
        legacy_id: str,
        snippet_text: str,
        pubmed_url: str,
        pmid: str,
        section: str,
        sample_id: str,
        question_type: str,
    ) -> None:
        self.legacy_id = legacy_id
        self.snippet_text = snippet_text
        self.pubmed_url = pubmed_url
        self.pmid = pmid
        self.section = section
        self.sample_id = sample_id
        self.question_type = question_type


def stream_bioasq_csv(
    file_path: str,
    chunk_size: int = 1000,
) -> Generator[List[Dict[str, str]], None, None]:
    """Stream CSV rows using SPEC Section 9.4 parsing rules."""
    encodings = ["utf-8", "utf-8-sig"]
    file_obj = None

    for enc in encodings:
        try:
            file_obj = open(file_path, "r", encoding=enc, newline="")
            # Check first line to verify valid encoding
            file_obj.read(1024)
            file_obj.seek(0)
            break
        except (UnicodeDecodeError, OSError):
            if file_obj is not None:
                file_obj.close()
                file_obj = None

    if file_obj is None:
        raise ValueError(f"Unable to read {file_path} with utf-8 or utf-8-sig encodings.")

    try:
        reader = csv.DictReader(file_obj)
        batch: List[Dict[str, str]] = []
        for row in reader:
            # Preserve string types and strip whitespace from keys
            clean_row = {k.strip(): str(v) for k, v in row.items() if k is not None}
            batch.append(clean_row)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        file_obj.close()


def process_corpus_row(
    row: Dict[str, str],
    embedding_service: EmbeddingService,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> Tuple[Optional[BioASQCorpusRecord], List[Tuple[str, int, str, str, bool, int]], Optional[str]]:
    """Process a single BioASQ row into Document metadata and passage chunks.
    
    Returns:
    - record: BioASQCorpusRecord (if valid)
    - chunks: List of (evidence_id, chunk_index, chunk_text, text_hash, short_passage, token_count)
    - error: Optional error string if rejected
    """
    legacy_id = row.get("evidence_id", "").strip()
    snippet_text = row.get("snippet_text", "")
    pubmed_url = row.get("pubmed_url", "").strip()
    section = row.get("section", "abstract").strip()
    sample_id = row.get("sample_id", "").strip()
    question_type = row.get("question_type", "").strip()

    if not legacy_id:
        return None, [], "Missing evidence_id"

    if not snippet_text:
        return None, [], "Empty snippet_text"

    pmid = extract_pmid_from_url(pubmed_url)
    if not pmid:
        return None, [], f"PMID extraction failed for url: {pubmed_url}"

    record = BioASQCorpusRecord(
        legacy_id=legacy_id,
        snippet_text=snippet_text,
        pubmed_url=pubmed_url,
        pmid=pmid,
        section=section,
        sample_id=sample_id,
        question_type=question_type,
    )

    # Chunker applied ONLY when token count exceeds max_tokens (100)
    text_chunks = embedding_service.chunk_text_if_needed(
        snippet_text,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks: List[Tuple[str, int, str, str, bool, int]] = []
    for idx, c_text in enumerate(text_chunks):
        ev_id = build_evidence_id(source="bioasq", source_id=legacy_id, chunk_index=idx)
        c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
        is_short = len(c_text) < 50
        t_count = embedding_service.count_tokens(c_text)
        chunks.append((ev_id, idx, c_text, c_hash, is_short, t_count))

    return record, chunks, None
