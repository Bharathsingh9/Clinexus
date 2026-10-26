"""Platform Evidence ID Scheme.

Specification (SPEC 6, 9.3):
One evidence ID scheme: ev:<source>:<source_id>:<chunk_index>, built and parsed
only by schemas/ids.py. Examples: ev:pubmed:12345678:0, ev:bioasq:E000001:0.
"""

import re
from typing import Tuple

EVIDENCE_ID_REGEX = re.compile(r"^ev:([a-z0-9_-]+):([a-zA-Z0-9_.-]+):(\d+)$")


def build_evidence_id(source: str, source_id: str, chunk_index: int) -> str:
    """Build a validated platform evidence ID.
    
    Args:
        source: Source identifier (e.g., 'pubmed', 'bioasq').
        source_id: Unique record identifier in the source (e.g., PMID or sample_id).
        chunk_index: Non-negative 0-based chunk sequence index.
        
    Returns:
        Formatted evidence ID string: ev:<source>:<source_id>:<chunk_index>
        
    Raises:
        ValueError: If any parameter is empty, invalid, or chunk_index is negative.
    """
    if not source or not isinstance(source, str):
        raise ValueError("Source must be a non-empty string.")
    clean_source = source.strip().lower()
    if not re.match(r"^[a-z0-9_-]+$", clean_source):
        raise ValueError(f"Invalid characters in source: '{source}'. Allowed: a-z, 0-9, _, -")

    if not source_id or not isinstance(source_id, str):
        raise ValueError("Source ID must be a non-empty string.")
    clean_source_id = source_id.strip()
    if not re.match(r"^[a-zA-Z0-9_.-]+$", clean_source_id):
        raise ValueError(f"Invalid characters in source_id: '{source_id}'. Allowed: a-z, A-Z, 0-9, _, ., -")

    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError(f"chunk_index must be a non-negative integer, got {chunk_index}")

    evidence_id = f"ev:{clean_source}:{clean_source_id}:{chunk_index}"
    return evidence_id


def parse_evidence_id(evidence_id: str) -> Tuple[str, str, int]:
    """Parse and validate a platform evidence ID.
    
    Args:
        evidence_id: String formatted as ev:<source>:<source_id>:<chunk_index>
        
    Returns:
        Tuple of (source, source_id, chunk_index)
        
    Raises:
        ValueError: If evidence_id is malformed or invalid.
    """
    if not evidence_id or not isinstance(evidence_id, str):
        raise ValueError("evidence_id must be a non-empty string.")

    match = EVIDENCE_ID_REGEX.match(evidence_id.strip())
    if not match:
        raise ValueError(
            f"Malformed evidence ID '{evidence_id}'. Expected format 'ev:<source>:<source_id>:<chunk_index>'."
        )

    source, source_id, chunk_index_str = match.groups()
    return source, source_id, int(chunk_index_str)


def validate_evidence_id(evidence_id: str) -> bool:
    """Check whether a string matches the valid evidence ID format."""
    try:
        parse_evidence_id(evidence_id)
        return True
    except (ValueError, TypeError):
        return False
