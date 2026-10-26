"""Evidence Schemas.

Specification (SPEC 4.2, 4.3, 6, 9.3).
Models: RetrievalQuery, EvidenceDocument, EvidenceChunk, RetrievedEvidenceSet, ExternalEvidenceSet.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.schemas.ids import validate_evidence_id


class EvidenceSourceEnum(str, Enum):
    PUBMED = "PUBMED"
    BIOASQ = "BIOASQ"


class QueryStatusEnum(str, Enum):
    OK = "OK"
    EMPTY = "EMPTY"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    MALFORMED = "MALFORMED"
    UNAVAILABLE = "UNAVAILABLE"


class RetrievalQuery(BaseModel):
    """Query object for vector or literature retrieval."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = Field(..., min_length=1)
    query_type: str = Field(default="symptom_vector", description="symptom_vector | literature_search | gap_fill")
    target_finding_ids: List[str] = Field(default_factory=list)
    filters: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class EvidenceDocument(BaseModel):
    """Top-level document metadata for an article or record."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: EvidenceSourceEnum = Field(...)
    source_id: str = Field(..., min_length=1, description="PMID or BioASQ sample ID")
    title: Optional[str] = Field(None, description="Document title or UNKNOWN")
    authors: Optional[List[str]] = Field(None, description="Author list or UNKNOWN")
    journal: Optional[str] = Field(None, description="Journal name or UNKNOWN")
    publication_date: Optional[str] = Field(None, description="YYYY or YYYY-MM-DD or UNKNOWN")
    pmid: Optional[str] = Field(None)
    doi: Optional[str] = Field(None)
    url: Optional[str] = Field(None)
    abstract: Optional[str] = Field(None)
    abstract_missing: bool = Field(default=False)
    evidence_level: Optional[str] = Field(default="UNKNOWN", description="Publication type or UNKNOWN")
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class EvidenceChunk(BaseModel):
    """Passage chunk representation stored and embedded."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(...)
    evidence_id: str = Field(..., description="Platform evidence ID: ev:<source>:<source_id>:<chunk_index>")
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    text_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 of passage text")
    short_passage: bool = Field(default=False, description="Flagged true if under 50 characters")
    token_count: Optional[int] = Field(None, ge=0)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")

    @field_validator("evidence_id")
    @classmethod
    def validate_ev_id(cls, v: str) -> str:
        if not validate_evidence_id(v):
            raise ValueError(f"Invalid platform evidence ID format: {v}")
        return v


class RetrievedPassageResult(BaseModel):
    """Search result item returned from Qdrant and enriched with metadata."""
    evidence_id: str = Field(...)
    document_id: str = Field(...)
    chunk_id: str = Field(...)
    title: Optional[str] = Field(default="UNKNOWN")
    source: str = Field(...)
    publication_date: Optional[str] = Field(default="UNKNOWN")
    source_location: Optional[str] = Field(default="UNKNOWN")
    retrieval_score: float = Field(..., ge=-1.0, le=1.0)
    rank: int = Field(..., ge=1)
    corpus_version: str = Field(...)
    embedding_model: str = Field(...)
    query_id: str = Field(...)
    evidence_level: Optional[str] = Field(default="UNKNOWN")
    text: str = Field(..., min_length=1)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")

    @field_validator("evidence_id")
    @classmethod
    def validate_ev_id(cls, v: str) -> str:
        if not validate_evidence_id(v):
            raise ValueError(f"Invalid platform evidence ID: {v}")
        return v


class RetrievedEvidenceSet(BaseModel):
    """Output contract for Symptom RAG Analyzer (AGT-2)."""
    query_id: str = Field(...)
    results: List[RetrievedPassageResult] = Field(default_factory=list)
    insufficient: bool = Field(default=False)
    reason_code: Optional[str] = Field(None, description="VECTOR_DB_UNAVAILABLE | EMPTY_CORPUS | LOW_SIMILARITY")
    retrieval_summary: str = Field(default="", description="Summary where every sentence cites evidence IDs")
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason_code")
    @classmethod
    def validate_insufficient_reason(cls, v: Optional[str], info) -> Optional[str]:
        insufficient = info.data.get("insufficient")
        if insufficient and not v:
            raise ValueError("When insufficient is True, reason_code must be provided.")
        return v


class ExternalEvidenceSet(BaseModel):
    """Output contract for Evidence-Based Scanner (AGT-3)."""
    query_ids: List[str] = Field(default_factory=list)
    documents: List[EvidenceDocument] = Field(default_factory=list)
    passages: List[EvidenceChunk] = Field(default_factory=list)
    query_statuses: Dict[str, QueryStatusEnum] = Field(default_factory=dict)
    summary: str = Field(default="")
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")
