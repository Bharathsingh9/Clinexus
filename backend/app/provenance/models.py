"""Provenance Models and Enums.

Specification Section 5.3 (EXT-C, Req ID: EXT-C):
Relations:
- SUPPORTS
- PARTIALLY_SUPPORTS
- CONTRADICTS
- INSUFFICIENT
- DERIVED_FROM_PATIENT_INPUT
- RETRIEVED_BY
- GENERATED_IN

Node Kinds:
- Claims and summaries: INTERPRETATION
- Passages: SOURCE
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class ProvenanceRelationEnum(str, Enum):
    SUPPORTS = "SUPPORTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    INSUFFICIENT = "INSUFFICIENT"
    DERIVED_FROM_PATIENT_INPUT = "DERIVED_FROM_PATIENT_INPUT"
    RETRIEVED_BY = "RETRIEVED_BY"
    GENERATED_IN = "GENERATED_IN"


class ProvenanceNodeKindEnum(str, Enum):
    INTERPRETATION = "INTERPRETATION"
    SOURCE = "SOURCE"


class ProvenanceEdgeSchema(BaseModel):
    """Pydantic model representing a typed evidence-level provenance edge."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_version_id: str = Field(..., description="Report version this provenance edge belongs to")
    case_id: Optional[str] = Field(None, description="Clinical case ID")
    finding_id: Optional[str] = Field(None, description="CaseFinding ID if patient fact")
    query_id: Optional[str] = Field(None, description="Retrieval query ID")
    evidence_chunk_id: Optional[str] = Field(None, description="EvidencePassage ID")
    document_id: Optional[str] = Field(None, description="EvidenceDocument ID")
    claim_id: Optional[str] = Field(None, description="CandidateClaim ID")
    verification_id: Optional[str] = Field(None, description="ClaimVerification ID")
    report_section_id: Optional[str] = Field(None, description="Report section identifier")
    iteration_id: Optional[str] = Field(None, description="Optimization iteration ID")
    abstention_id: Optional[str] = Field(None, description="Abstention decision ID")
    review_event_id: Optional[str] = Field(None, description="Human review audit ID")
    relation: ProvenanceRelationEnum = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")


class ProvenanceNode(BaseModel):
    """Provenance graph node with distinct INTERPRETATION vs SOURCE classification."""
    node_id: str = Field(...)
    node_type: str = Field(..., description="claim | summary | passage | finding | query")
    kind: ProvenanceNodeKindEnum = Field(..., description="INTERPRETATION or SOURCE")
    label: str = Field(...)
    metadata: Dict[str, str] = Field(default_factory=dict)
    schema_version: str = Field(default="1.0")

    model_config = ConfigDict(extra="forbid")
