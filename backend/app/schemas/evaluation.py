"""Evaluation Schemas for BioASQ Questions and Gold Labels.

Specification Section 9.4, Section 9.7 (Req ID: DAT-4, EVL-5):
- WorkflowInput: STRICT isolation containing ONLY sample_id, question, question_type (extra='forbid').
  Guarantees zero gold answer strings, snippet texts, or gold PMIDs reach any workflow agent.
- BioASQGoldLabel: Isolated evaluation label contract containing exact_answer, ideal_answer,
  and extracted gold PMIDs for downstream offline evaluation scoring.
"""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class WorkflowInput(BaseModel):
    """Workflow entry contract for BioASQ question evaluation.
    
    Invariant (SPEC 9.7): Only sample_id, question, and question_type may enter any
    query, prompt, or agent. extra="forbid" ensures no gold metadata can leak.
    """
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., min_length=1, description="Raw clinical/biomedical question text")
    question_type: str = Field(..., description="BioASQ question type: factoid | yesno | list | summary")


class BioASQGoldLabel(BaseModel):
    """Gold ground-truth labels extracted from BioASQ normalized dataset.
    
    Stored strictly in PostgreSQL evaluation_labels table; never in evidence tables or Qdrant.
    """
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., description="Question sample_id matching WorkflowInput")
    question_type: str = Field(..., description="factoid | yesno | list | summary")
    exact_answer: Optional[Union[str, List[str], List[List[str]]]] = Field(
        default=None,
        description="Gold exact answer conforming to SPEC 9.4 accepted shapes",
    )
    ideal_answer: List[str] = Field(
        ...,
        min_length=1,
        description="Gold ideal answers (list of non-empty strings)",
    )
    gold_pmids: List[str] = Field(
        default_factory=list,
        description="Extracted PMIDs from source_reference URLs using SPEC 9.3 regex",
    )
    relevant_passages: List[str] = Field(
        default_factory=list,
        description="Platform passage IDs ev:bioasq:<legacy_id>:<chunk_index> carrying this sample_id",
    )
    derivation_source: str = Field(
        default="DERIVED_FROM_SAMPLE_ID",
        description="Provenance tag for relevance labels (SPEC 9.5)",
    )
    confirmed_by_human: bool = Field(
        default=False,
        description="Human confirmation status (SPEC 9.5)",
    )
