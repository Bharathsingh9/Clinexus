"""Retrieval Query Builder.

Specification Section 4.2 (Req ID: RET-4):
- Synthesize 1 to 3 RetrievalQuery objects from StructuredSymptom list and PatientContext.
- Assembled primarily by deterministic code, optionally refined by LLM expansion.
- Incorporates alternative_query_hints during optimization cycles.
"""

from typing import List, Optional
import uuid
from backend.app.schemas.clinical_case import ClinicalCase, PolarityEnum
from backend.app.schemas.evidence import RetrievalQuery


def build_retrieval_queries(
    case: ClinicalCase,
    alternative_hints: Optional[List[str]] = None,
) -> List[RetrievalQuery]:
    """Construct 1 to 3 focused RetrievalQuery objects from a clinical case."""
    queries: List[RetrievalQuery] = []

    # Filter present symptoms
    present_symptoms = [
        s for s in case.structured_symptoms if s.polarity == PolarityEnum.PRESENT
    ]
    present_terms = [s.term for s in present_symptoms]
    finding_ids = [s.finding_id for s in present_symptoms]

    # Query 1: Primary Presentation Query (Chief complaint + primary present symptoms)
    cc = case.chief_complaint.strip()
    primary_terms = present_terms[:4]
    if primary_terms:
        q1_text = f"{cc} presenting with {', '.join(primary_terms)}"
    else:
        q1_text = cc

    queries.append(
        RetrievalQuery(
            query_id=str(uuid.uuid4()),
            query_text=q1_text,
            query_type="symptom_vector",
            target_finding_ids=finding_ids[:4],
        )
    )

    # Query 2: Focused Symptom Cluster / Context Query (if additional findings exist)
    if len(present_terms) > 1 or case.patient_context.history:
        cluster_terms = present_terms[2:6] if len(present_terms) > 2 else present_terms
        history_terms = case.patient_context.history[:2]
        parts = []
        if cluster_terms:
            parts.append(" ".join(cluster_terms))
        if history_terms:
            parts.append("history of " + ", ".join(history_terms))
        if case.symptom_duration:
            parts.append(f"duration {case.symptom_duration}")

        q2_text = " - ".join(parts) if parts else f"{cc} clinical findings"
        queries.append(
            RetrievalQuery(
                query_id=str(uuid.uuid4()),
                query_text=q2_text,
                query_type="symptom_vector",
                target_finding_ids=finding_ids[2:6] if len(finding_ids) > 2 else finding_ids,
            )
        )

    # Query 3: Refinement / Alternative Query from hints or associated symptoms
    if alternative_hints and len(alternative_hints) > 0:
        hint_text = alternative_hints[0].strip()
        queries.append(
            RetrievalQuery(
                query_id=str(uuid.uuid4()),
                query_text=f"{cc} {hint_text}",
                query_type="symptom_vector",
                target_finding_ids=finding_ids,
            )
        )
    elif len(present_symptoms) > 0 and len(queries) < 3:
        # Construct differential-oriented query from associated symptoms if present
        all_associated = []
        for s in present_symptoms:
            all_associated.extend(s.associated_symptoms)
        if all_associated:
            queries.append(
                RetrievalQuery(
                    query_id=str(uuid.uuid4()),
                    query_text=f"{cc} associated with {', '.join(all_associated[:3])}",
                    query_type="symptom_vector",
                    target_finding_ids=finding_ids,
                )
            )

    # Enforce at most 3 queries
    return queries[:3]
