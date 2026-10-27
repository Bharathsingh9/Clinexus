"""Evidence-Based Scanner (Agent 3 - AGT-3, App A2, A5, A9).

Specification Section 4.3, Appendix A2, A5, A9:
- Input: ClinicalCase, RetrievedEvidenceSet, optional gap_hints.
- Output: ExternalEvidenceSet (or AgentExecutionRecord if FAILED with LLM_UNAVAILABLE).
- Per-Hypothesis Scanning: builds one focused PubMed query per CandidateHypothesis plus gap queries.
- In benchmark mode or when candidate hypotheses are empty, executes gap queries.
- Sequential execution on candidate hypotheses and gaps (App A2).
- Queries NCBI E-utilities client.
- Chunks abstracts using EmbeddingService.chunk_text_if_needed.
- Formats evidence IDs as ev:pubmed:<pmid>:<chunk_index>.
- Flags records without abstract (abstract_missing=True) and excludes them from passage chunks.
- Deduplicates against local evidence by PMID, DOI, and text hash.
- Per-query statuses: OK | EMPTY | TIMEOUT | RATE_LIMITED | MALFORMED | UNAVAILABLE | SKIPPED_OFFLINE.
- Offline mode: when EXTERNAL_MODE="offline", returns SKIPPED_OFFLINE and makes zero network calls.
- In online mode, if no LLM provider is configured, returns FAILED with error_code LLM_UNAVAILABLE.
"""

from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import uuid

from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.evidence_sources.ncbi_client import NCBIClient, NCBIClientProtocol
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.schemas.clinical_case import CandidateHypothesis, ClinicalCase
from backend.app.schemas.evidence import (
    EvidenceChunk,
    EvidenceDocument,
    ExternalEvidenceSet,
    QueryStatusEnum,
    RetrievedEvidenceSet,
)
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum

logger = logging.getLogger(__name__)


class EvidenceBasedScanner:
    """Agent 3: Evidence-Based Scanner querying external PubMed literature."""

    def __init__(
        self,
        ncbi_client: Optional[NCBIClientProtocol] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_service: Optional[EmbeddingService] = None,
        model_name: Optional[str] = None,
    ):
        self.ncbi_client = ncbi_client or NCBIClient()
        self.llm_provider = llm_provider
        self.embedding_service = embedding_service or EmbeddingService()
        self.model_name = model_name or settings.LLM_MODEL_SCANNER

    def _build_hypothesis_query(
        self,
        case: ClinicalCase,
        hypothesis: CandidateHypothesis,
    ) -> str:
        """Formulate a targeted PubMed query for a candidate differential consideration."""
        # Clean label and extract core clinical terms
        label_clean = re.sub(r"[^\w\s-]", "", hypothesis.label).strip()
        # Pair with primary findings
        symptoms_str = " ".join([
            s.normalized_term or s.term
            for s in case.structured_symptoms[:2]
        ])
        if symptoms_str:
            return f"({label_clean}) AND ({symptoms_str}) AND (diagnosis OR differential)"
        return f"({label_clean}) AND (differential diagnosis)"

    def _build_gap_queries(
        self,
        case: ClinicalCase,
        gap_hints: Optional[List[str]] = None,
    ) -> List[str]:
        """Formulate fallback / gap queries for presentations or benchmark mode."""
        queries: List[str] = []
        if gap_hints:
            for hint in gap_hints:
                queries.append(f"({hint}) AND (clinical presentation)")
            return queries

        # Default gap query from chief complaint and findings
        if case.chief_complaint:
            cc_clean = re.sub(r"[^\w\s-]", "", case.chief_complaint).strip()
            queries.append(f"({cc_clean}) AND (etiology OR diagnosis)")

        symptom_terms = [s.normalized_term or s.term for s in case.structured_symptoms[:3]]
        if len(symptom_terms) >= 2:
            s_query = " AND ".join([f"({t})" for t in symptom_terms])
            queries.append(f"{s_query} AND (clinical evaluation)")

        return queries if queries else ["clinical differential diagnosis"]

    def run(
        self,
        case: ClinicalCase,
        retrieved_evidence: Optional[RetrievedEvidenceSet] = None,
        gap_hints: Optional[List[str]] = None,
        benchmark_mode: bool = False,
    ) -> Union[ExternalEvidenceSet, AgentExecutionRecord]:
        """Execute Agent 3 scanning and return ExternalEvidenceSet or FAILED AgentExecutionRecord."""
        # 1. Offline Mode Check: Zero network calls, no LLM required to skip
        is_offline = (
            getattr(self.ncbi_client, "offline", False)
            or settings.EXTERNAL_MODE.lower() == "offline"
        )
        if is_offline:
            return ExternalEvidenceSet(
                query_ids=["q:external:offline"],
                documents=[],
                passages=[],
                query_statuses={"q:external:offline": QueryStatusEnum.SKIPPED_OFFLINE},
                summary="External scanning was skipped because EXTERNAL_MODE is set to offline.",
            )

        # 2. Invariant: If online, no silent fallbacks. Unconfigured LLM provider returns FAILED(LLM_UNAVAILABLE).
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="evidence_based_scanner",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured (LLM_PROVIDER unset or 'none'). Real LLM calls are blocked.",
                input_summary=f"Clinical case (age={case.age}, symptoms={len(case.structured_symptoms)})",
                output_summary=None,
            )

        # 3. Construct Queries: per-hypothesis queries + gap queries (App A5)
        queries_to_run: List[Tuple[str, str]] = []  # (query_id, query_string)

        hypotheses = retrieved_evidence.candidate_hypotheses if retrieved_evidence else []

        if not benchmark_mode and hypotheses:
            for idx, hyp in enumerate(hypotheses, 1):
                q_id = f"q:external:hyp:{idx}"
                q_str = self._build_hypothesis_query(case, hyp)
                queries_to_run.append((q_id, q_str))

        # Include gap queries if requested or in benchmark mode or no hypotheses
        if gap_hints or benchmark_mode or not hypotheses:
            gap_q_list = self._build_gap_queries(case, gap_hints)
            for idx, g_str in enumerate(gap_q_list, 1):
                q_id = f"q:external:gap:{idx}"
                queries_to_run.append((q_id, g_str))

        # 4. Collect local evidence identifiers for deduplication
        local_pmids: Set[str] = set()
        local_text_hashes: Set[str] = set()
        if retrieved_evidence:
            for r in retrieved_evidence.results:
                # Extract PMID from evidence_id or document_id
                m = re.search(r"ev:bioasq:(\d+):", r.evidence_id)
                if m:
                    local_pmids.add(m.group(1))
                if r.text:
                    h = hashlib.sha256(r.text.strip().lower().encode("utf-8")).hexdigest()
                    local_text_hashes.add(h)

        fetched_documents: List[EvidenceDocument] = []
        fetched_passages: List[EvidenceChunk] = []
        query_statuses: Dict[str, QueryStatusEnum] = {}
        query_ids: List[str] = []

        seen_external_pmids: Set[str] = set()

        # 5. Execute Queries Sequentially (App A2)
        for q_id, q_text in queries_to_run:
            query_ids.append(q_id)
            try:
                docs, status = self.ncbi_client.search_and_fetch(
                    query=q_text,
                    max_results=settings.EXTERNAL_MAX_RECORDS_PER_QUERY,
                )
            except Exception as exc:
                logger.error("NCBI client execution failed for query %s: %s", q_id, exc)
                query_statuses[q_id] = QueryStatusEnum.UNAVAILABLE
                continue

            query_statuses[q_id] = status

            for doc in docs:
                pmid = doc.pmid or doc.source_id
                # Deduplication against local evidence and already fetched external docs
                if pmid in local_pmids or pmid in seen_external_pmids:
                    logger.debug("Deduplicating external PMID %s against existing evidence", pmid)
                    continue

                seen_external_pmids.add(pmid)
                fetched_documents.append(doc)

                # Flag abstract_missing: records without abstract are excluded from passage chunks (SPEC 4.3)
                if doc.abstract_missing or not doc.abstract:
                    continue

                # Chunk abstract with effective chunker
                text_chunks = self.embedding_service.chunk_text_if_needed(doc.abstract)
                for c_idx, c_text in enumerate(text_chunks):
                    t_hash = hashlib.sha256(c_text.strip().lower().encode("utf-8")).hexdigest()
                    if t_hash in local_text_hashes:
                        continue
                    local_text_hashes.add(t_hash)

                    ev_id = f"ev:pubmed:{pmid}:{c_idx}"
                    short_flag = len(c_text.strip()) < 50
                    tokens = self.embedding_service.count_tokens(c_text)

                    chunk = EvidenceChunk(
                        document_id=doc.document_id,
                        evidence_id=ev_id,
                        chunk_index=c_idx,
                        text=c_text,
                        text_hash=t_hash,
                        short_passage=short_flag,
                        token_count=tokens,
                    )
                    fetched_passages.append(chunk)

        summary_parts = [
            f"Executed {len(queries_to_run)} external queries.",
            f"Retrieved {len(fetched_documents)} distinct PubMed articles and {len(fetched_passages)} passage chunks.",
        ]
        if any(st == QueryStatusEnum.RATE_LIMITED for st in query_statuses.values()):
            summary_parts.append("Note: rate limiting was encountered on some queries.")
        if any(st == QueryStatusEnum.UNAVAILABLE for st in query_statuses.values()):
            summary_parts.append("Warning: external source was temporarily unavailable for some queries.")

        return ExternalEvidenceSet(
            query_ids=query_ids,
            documents=fetched_documents,
            passages=fetched_passages,
            query_statuses=query_statuses,
            summary=" ".join(summary_parts),
        )
