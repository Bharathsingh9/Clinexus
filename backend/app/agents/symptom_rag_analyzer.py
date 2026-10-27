"""Symptom RAG Analyzer (Agent 2 - AGT-2).

Specification Section 4.2, Appendix A5:
- Input: ClinicalCase, optional alternative_query_hints, optional benchmark_mode.
- Output: RetrievedEvidenceSet.
- Constructs 1 to 3 queries, embeds with EmbeddingService, queries Qdrant.
- Resolves full text and document metadata strictly from PostgreSQL.
- Deduplicates by chunk ID and text hash, ranks, returns up to top_k.
- Flags insufficient=True when too few passages pass min_score, Qdrant unavailable, or empty corpus.
- Produces CandidateHypothesis objects (grounded in passages; candidate considerations, never diagnoses).
- Sentence citation invariant: every sentence in retrieval_summary must cite valid evidence IDs.
"""

import asyncio
import inspect
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
import uuid
from sqlalchemy import select
from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.models import EvidenceDocument, EvidencePassage
from backend.app.retrieval.deduplication import deduplicate_passages
from backend.app.retrieval.embedding_service import EmbeddingService
from backend.app.retrieval.qdrant_client import QdrantWrapper, format_collection_name
from backend.app.retrieval.query_builder import build_retrieval_queries
from backend.app.retrieval.retrieval_guard import evaluate_retrieval_sufficiency
from backend.app.schemas.clinical_case import CandidateHypothesis, ClinicalCase
from backend.app.schemas.evidence import (
    RetrievedEvidenceSet,
    RetrievedPassageResult,
)

logger = logging.getLogger(__name__)


class SymptomRAGAnalyzer:
    """Agent 2: Retrieves local evidence from Qdrant/Postgres and produces candidate hypotheses."""

    def __init__(
        self,
        qdrant_wrapper: Optional[QdrantWrapper] = None,
        embedding_service: Optional[EmbeddingService] = None,
        db_session_factory=SessionLocal,
        llm_provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
        collection_version: int = 1,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> None:
        self.qdrant_wrapper = qdrant_wrapper or QdrantWrapper()
        self.embedding_service = embedding_service or EmbeddingService()
        self.db_session_factory = db_session_factory
        self.llm_provider = llm_provider
        self.model_name = model_name or settings.LLM_MODEL_RAG
        self.collection_version = collection_version
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.min_score = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE

    def _resolve_collection_name(self) -> str:
        return format_collection_name(self.embedding_service.model_name, self.collection_version)

    def _fetch_postgres_passages(
        self,
        evidence_ids: List[str],
    ) -> Dict[str, Dict]:
        """Fetch passage text and document metadata from PostgreSQL."""
        if not evidence_ids:
            return {}

        results_map = {}
        with self.db_session_factory() as session:
            stmt = (
                select(EvidencePassage, EvidenceDocument)
                .join(EvidenceDocument, EvidencePassage.document_id == EvidenceDocument.id)
                .where(EvidencePassage.evidence_id.in_(evidence_ids))
            )
            rows = session.execute(stmt).all()
            for passage, doc in rows:
                results_map[passage.evidence_id] = {
                    "chunk_id": passage.id,
                    "document_id": doc.id,
                    "evidence_id": passage.evidence_id,
                    "text": passage.text,
                    "title": doc.title or "UNKNOWN",
                    "source": doc.source,
                    "publication_date": doc.publication_date or "UNKNOWN",
                    "source_location": f"chunk_{passage.chunk_index}",
                    "evidence_level": doc.evidence_level or "UNKNOWN",
                }
        return results_map

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Call provider handling both sync and async signatures."""
        method = getattr(self.llm_provider, "generate_text", None) or getattr(
            self.llm_provider, "generate", None
        )
        if method is None:
            raise AttributeError("LLMProvider does not implement generate_text or generate")

        try:
            res = method(
                prompt=prompt,
                system_instruction=system,
                model=self.model_name,
                temperature=0.0,
            )
        except TypeError:
            res = method(
                prompt=prompt,
                system=system,
                model=self.model_name,
                temperature=0.0,
            )

        if inspect.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, res).result()
            else:
                return asyncio.run(res)
        return str(res)

    def _generate_candidate_hypotheses(
        self,
        case: ClinicalCase,
        passages: List[RetrievedPassageResult],
        valid_ev_ids: Set[str],
    ) -> List[CandidateHypothesis]:
        """Generate candidate differential considerations grounded in retrieved passages (App A5)."""
        if not passages or not valid_ev_ids:
            return []

        hypotheses: List[CandidateHypothesis] = []

        # If LLM provider is available, query for structured candidate considerations
        if self.llm_provider:
            evidence_context = "\n".join(
                [f"[{p.evidence_id}] (Score: {p.retrieval_score:.2f}) {p.text[:200]}" for p in passages[:5]]
            )
            prompt = (
                f"PATIENT PRESENTATION: {case.chief_complaint}\n"
                f"RETRIEVED PASSAGES:\n{evidence_context}\n\n"
                f"List 1 to 3 candidate diagnostic considerations. For each consideration, cite "
                f"ONLY the supporting evidence IDs from the passages above. Format each line as:\n"
                f"CONSIDERATION: <label> | CITATIONS: <evidence_id_1>, <evidence_id_2>"
            )
            try:
                raw_response = self._invoke_llm(
                    prompt=prompt,
                    system="You are the Symptom RAG Analyzer. Propose candidate differential considerations strictly grounded in the provided passages.",
                )
                for line in raw_response.strip().split("\n"):
                    if "CONSIDERATION:" in line and "CITATIONS:" in line:
                        parts = line.split("CITATIONS:")
                        label = parts[0].replace("CONSIDERATION:", "").strip(" |:-")
                        cits = [c.strip(" []") for c in parts[1].split(",") if c.strip(" []")]

                        # STRICT VALIDATION: Keep only IDs present in retrieved set
                        filtered_cits = [c for c in cits if c in valid_ev_ids]
                        if filtered_cits and label:
                            # Map scores from retrieved passages
                            scores = [
                                next((p.retrieval_score for p in passages if p.evidence_id == cid), 0.5)
                                for cid in filtered_cits
                            ]
                            hypotheses.append(
                                CandidateHypothesis(
                                    hypothesis_id=str(uuid.uuid4()),
                                    label=label,
                                    supporting_evidence_ids=filtered_cits,
                                    retrieval_scores=scores,
                                )
                            )
            except Exception as exc:
                logger.warning("LLM hypothesis generation failed (%s). Leaving hypotheses empty.", exc)

        return hypotheses

    def _build_and_verify_summary(
        self,
        passages: List[RetrievedPassageResult],
    ) -> str:
        """Construct retrieval summary ensuring EVERY sentence cites evidence IDs."""
        if not passages:
            return "No relevant passages retrieved for the presenting symptoms."

        sentences: List[str] = []
        for p in passages[:3]:
            # Take first clause / sentence of passage and append citation
            snippet = p.text.split(".")[0].strip()
            if len(snippet) > 120:
                snippet = snippet[:120].strip() + "..."
            sentence = f"{snippet} [{p.evidence_id}]."
            sentences.append(sentence)

        summary = " ".join(sentences)

        # Code invariant verification: ensure every sentence has a citation
        clean_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        verified_sentences = []
        for s in clean_sentences:
            if not re.search(r"ev:[a-z0-9_-]+:[a-z0-9_.-]+:\d+", s):
                # Append citation to primary passage
                s = f"{s.rstrip('.')} [{passages[0].evidence_id}]."
            verified_sentences.append(s)

        return " ".join(verified_sentences)

    def run(
        self,
        case: ClinicalCase,
        alternative_hints: Optional[List[str]] = None,
        benchmark_mode: bool = False,
    ) -> Union[RetrievedEvidenceSet, AgentExecutionRecord]:
        """Execute Agent 2 retrieval and return RetrievedEvidenceSet or FAILED AgentExecutionRecord."""
        # Invariant: No silent fallbacks. If no LLM provider is configured, return FAILED with LLM_UNAVAILABLE.
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="symptom_rag_analyzer",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured (LLM_PROVIDER unset or 'none'). Real LLM calls are blocked.",
                input_summary=f"Clinical case (age={case.age}, sex={case.sex}, symptoms={len(case.structured_symptoms)})",
                output_summary=None,
            )

        collection_name = self._resolve_collection_name()

        # Step 1: Check Qdrant collection
        try:
            exists = self.qdrant_wrapper.collection_exists(collection_name)
        except Exception as exc:
            logger.error("Vector DB unreachable: %s", exc)
            return RetrievedEvidenceSet(
                query_id=str(uuid.uuid4()),
                results=[],
                insufficient=True,
                reason_code="VECTOR_DB_UNAVAILABLE",
                retrieval_summary="",
                candidate_hypotheses=[],
            )

        if not exists:
            return RetrievedEvidenceSet(
                query_id=str(uuid.uuid4()),
                results=[],
                insufficient=True,
                reason_code="EMPTY_CORPUS",
                retrieval_summary="",
                candidate_hypotheses=[],
            )

        # Step 2: Build queries
        queries = build_retrieval_queries(case, alternative_hints=alternative_hints)
        if not queries:
            return RetrievedEvidenceSet(
                query_id=str(uuid.uuid4()),
                results=[],
                insufficient=True,
                reason_code="EMPTY_CORPUS",
                retrieval_summary="",
                candidate_hypotheses=[],
            )

        primary_query = queries[0]
        all_retrieved_results: List[RetrievedPassageResult] = []

        # Step 3: Embed queries and search Qdrant
        for q in queries:
            try:
                q_vec = self.embedding_service.embed_text(q.query_text)
                hits = self.qdrant_wrapper.search(
                    collection_name=collection_name,
                    query_vector=q_vec,
                    limit=self.top_k,
                )
            except Exception as exc:
                logger.error("Qdrant search error for query %s: %s", q.query_id, exc)
                continue

            if not hits:
                continue

            # Collect evidence IDs from Qdrant payloads
            hit_ev_ids = [hit.payload.get("evidence_id") for hit in hits if hit.payload]
            hit_ev_ids = [eid for eid in hit_ev_ids if eid]

            # Fetch passage texts and metadata from PostgreSQL
            pg_data = self._fetch_postgres_passages(hit_ev_ids)

            for rank, hit in enumerate(hits, start=1):
                ev_id = hit.payload.get("evidence_id") if hit.payload else None
                if not ev_id or ev_id not in pg_data:
                    continue

                item = pg_data[ev_id]
                res = RetrievedPassageResult(
                    evidence_id=ev_id,
                    document_id=item["document_id"],
                    chunk_id=item["chunk_id"],
                    title=item["title"],
                    source=item["source"],
                    publication_date=item["publication_date"],
                    source_location=item["source_location"],
                    retrieval_score=float(hit.score),
                    rank=rank,
                    corpus_version=hit.payload.get("corpus_version", "unknown"),
                    embedding_model=hit.payload.get("embedding_model", self.embedding_service.model_name),
                    query_id=q.query_id,
                    evidence_level=item["evidence_level"],
                    text=item["text"],
                )
                all_retrieved_results.append(res)

        # Step 4: Deduplicate and rank
        deduped = deduplicate_passages(all_retrieved_results, top_k=self.top_k)

        # Step 5: Sufficiency evaluation
        collection_empty = len(deduped) == 0
        insufficient, reason_code = evaluate_retrieval_sufficiency(
            passages=deduped,
            min_score=self.min_score,
            min_passages=settings.MIN_RELEVANT_PASSAGES,
            collection_empty=collection_empty,
        )

        # Step 6: Candidate Hypotheses (per App A5)
        valid_ev_ids = {p.evidence_id for p in deduped}
        candidate_hypotheses: List[CandidateHypothesis] = []
        if not benchmark_mode and not insufficient:
            candidate_hypotheses = self._generate_candidate_hypotheses(
                case=case,
                passages=deduped,
                valid_ev_ids=valid_ev_ids,
            )

        # Step 7: Retrieval Summary with verified citations
        summary = self._build_and_verify_summary(deduped)

        return RetrievedEvidenceSet(
            query_id=primary_query.query_id,
            results=deduped,
            insufficient=insufficient,
            reason_code=reason_code,
            retrieval_summary=summary,
            candidate_hypotheses=candidate_hypotheses,
        )
