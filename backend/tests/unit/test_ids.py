"""Unit Tests for Evidence ID Scheme.

Specification Section 6, 9.3 (Req IDs: DB-2, RET-1).
"""

import pytest
from backend.app.schemas.ids import (
    build_evidence_id,
    parse_evidence_id,
    validate_evidence_id,
)


def test_build_and_parse_roundtrip():
    """Verify standard build and parse roundtrip."""
    source = "pubmed"
    source_id = "12345678"
    chunk_index = 0

    ev_id = build_evidence_id(source, source_id, chunk_index)
    assert ev_id == "ev:pubmed:12345678:0"

    parsed_source, parsed_id, parsed_chunk = parse_evidence_id(ev_id)
    assert parsed_source == source
    assert parsed_id == source_id
    assert parsed_chunk == chunk_index


def test_bioasq_evidence_id():
    """Verify BioASQ corpus ID format."""
    ev_id = build_evidence_id("bioasq", "E000001", 2)
    assert ev_id == "ev:bioasq:E000001:2"

    source, source_id, chunk_index = parse_evidence_id(ev_id)
    assert source == "bioasq"
    assert source_id == "E000001"
    assert chunk_index == 2


def test_validate_evidence_id():
    """Verify boolean validator."""
    assert validate_evidence_id("ev:pubmed:34567890:0") is True
    assert validate_evidence_id("ev:bioasq:B123:5") is True
    assert validate_evidence_id("invalid_id") is False
    assert validate_evidence_id("ev:pubmed:123:-1") is False


@pytest.mark.parametrize(
    "malformed_id",
    [
        "",
        "pubmed:1234:0",
        "ev::1234:0",
        "ev:pubmed::0",
        "ev:pubmed:1234:",
        "ev:pubmed:1234:-1",
        "ev:pubmed:1234:abc",
        "ev:pub med:1234:0",
        "ev:pubmed:1234:0:extra",
    ],
)
def test_parse_malformed_id_raises_value_error(malformed_id: str):
    """Verify that all malformed IDs raise ValueError."""
    with pytest.raises(ValueError):
        parse_evidence_id(malformed_id)


def test_build_invalid_parameters():
    """Verify build_evidence_id rejects invalid inputs."""
    with pytest.raises(ValueError, match="Source must be a non-empty string"):
        build_evidence_id("", "123", 0)

    with pytest.raises(ValueError, match="Source ID must be a non-empty string"):
        build_evidence_id("pubmed", "", 0)

    with pytest.raises(ValueError, match="chunk_index must be a non-negative integer"):
        build_evidence_id("pubmed", "123", -1)
