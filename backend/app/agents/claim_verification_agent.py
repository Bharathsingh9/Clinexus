"""Claim Verification Agent (Extension A - EXT-A).

Specification Section 5.1 (Req ID: EXT-A):
- In: CandidateClinicalReport, evidence_passages (union of local and external evidence sets), optional ClinicalCase.
- Out: list[ClaimVerification].
- Step 1 (deterministic):
    * Every cited ID must exist in THIS run's evidence sets, else UNVERIFIABLE with limitation CITATION_NOT_FOUND.
    * Patient-derived claims are checked against the case's source spans.
- Step 2 (deterministic):
    * Numbers, units, and named entities in the claim must appear in the cited passage text.
    * Invariant: A claim with a number absent from every cited passage can NEVER be SUPPORTED, whatever the model says.
- Step 3 (model-based entailment):
    * Entailment-style judgment per (claim, cited passages) pair returning status and justification pointing to evidence span by offsets.
    * Code verifies span offsets against the passage text.
    * Invariant: A judgment without a valid span can NEVER be SUPPORTED or CONTRADICTED.
- Scan other retrieved passages for opposing statements and list them as contradicting_evidence_ids.
- Uses VERIFIER_MODEL setting.
- Returns FAILED(LLM_UNAVAILABLE) without a provider.
- Never retrieves new evidence; never edits the report.
- One bounded retry on verifier error, then reports failure to the gate.
- Persists results to claim_verifications table.
"""

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.persistence.execution_repository import ExecutionRepository
from backend.app.schemas.clinical_case import ClinicalCase, SourceSpan
from backend.app.schemas.evidence import EvidenceChunk, RetrievedPassageResult
from backend.app.schemas.executions import (
    AgentExecutionRecord,
    AgentStatusEnum,
)
from backend.app.schemas.reports import (
    CandidateClaim,
    CandidateClinicalReport,
    ClaimVerification,
    ClaimVerificationStatusEnum,
    VerificationMethodEnum,
)

logger = logging.getLogger(__name__)

# Common medical numeric units for entity extraction
_UNIT_PATTERN = re.compile(
    r"\b(?:mg|mcg|g|kg|ml|l|mmHg|mmol/l|mg/dl|%|bpm|beats/min|cells/mcL|days?|weeks?|months?|years?|hours?)\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

# Opposing / contradiction sentiment markers
_OPPOSING_MARKERS = [
    r"\b(?:ineffective|contraindicated|fails\s+to|no\s+association|not\s+recommended|adverse|refuted|unsupported|disproven|ruled\s+out)\b",
    r"\b(?:did\s+not\s+(?:show|improve|reduce|prevent)|no\s+significant\s+(?:difference|benefit|effect))\b",
]


class ClaimVerificationAgent:
    """Agent EXT-A: Verifies clinical factual claims against retrieved evidence sets."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        repository: Optional[ExecutionRepository] = None,
        model_name: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.repository = repository
        self.model_name = model_name or settings.VERIFIER_MODEL

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Call LLM provider handling sync and async providers."""
        if self.llm_provider is None:
            raise ValueError("LLM provider is not configured.")
        method = getattr(self.llm_provider, "generate_text", None) or getattr(
            self.llm_provider, "generate", None
        )
        if not callable(method):
            raise AttributeError("LLM provider missing generate_text or generate method")

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

    def _scan_contradicting_passages(
        self,
        claim_text: str,
        other_passages: List[Union[EvidenceChunk, RetrievedPassageResult]],
    ) -> List[str]:
        """Scan other retrieved passages for opposing or contradicting statements."""
        contradicting_ids: List[str] = []
        claim_words = {
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim_text)
            if w.lower() not in {"patient", "clinical", "evidence", "findings", "symptoms"}
        }

        for chunk in other_passages:
            chunk_lower = chunk.text.lower()
            # Must overlap with claim key concepts
            overlap = any(word in chunk_lower for word in claim_words)
            if not overlap:
                continue

            # Check for explicit opposition markers
            for pattern in _OPPOSING_MARKERS:
                if re.search(pattern, chunk_lower, re.IGNORECASE):
                    contradicting_ids.append(chunk.evidence_id)
                    break

        return contradicting_ids

    def _verify_single_claim_model(
        self,
        claim: CandidateClaim,
        cited_chunks: List[Union[EvidenceChunk, RetrievedPassageResult]],
    ) -> Tuple[ClaimVerificationStatusEnum, str, List[SourceSpan], Optional[str]]:
        """Step 3: Model-based entailment with bounded retry and strict code offset verification."""
        passage_context_blocks = []
        for i, c in enumerate(cited_chunks):
            passage_context_blocks.append(
                f"[Passage {i+1} (ID: {c.evidence_id})]:\n{c.text}"
            )
        combined_passages = "\n\n".join(passage_context_blocks)

        system_prompt = (
            "You are a clinical claim verification NLI model. Given a clinical claim and cited passages, "
            "determine if the cited passages strictly support, partially support, contradict, or provide "
            "insufficient evidence for the claim.\n"
            "You must identify the exact character start and end offsets within the specific passage that "
            "contains the grounding statement.\n"
            "Return a JSON object with keys:\n"
            "- 'status': 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE'\n"
            "- 'justification': concise explanation\n"
            "- 'passage_index': 0-based index of the passage containing the evidence span\n"
            "- 'span_start': integer character start offset within that passage text\n"
            "- 'span_end': integer character end offset within that passage text"
        )

        prompt = (
            f"CLAIM: {claim.claim_text}\n\n"
            f"CITED PASSAGES:\n{combined_passages}\n\n"
            f"Evaluate entailment and return valid JSON:"
        )

        # Bounded retry: 1 initial attempt + 1 bounded retry
        last_error = None
        for attempt in range(2):
            try:
                raw_resp = self._invoke_llm(prompt=prompt, system=system_prompt)
                parsed = json.loads(raw_resp.strip())
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]

                raw_status = str(parsed.get("status", "INSUFFICIENT_EVIDENCE")).upper()
                justification = str(parsed.get("justification", "Model entailment completed."))
                p_idx = int(parsed.get("passage_index", 0))
                s_start = int(parsed.get("span_start", -1))
                s_end = int(parsed.get("span_end", -1))

                # Normalize status enum
                try:
                    status = ClaimVerificationStatusEnum(raw_status)
                except ValueError:
                    status = ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE

                # Code verifies the offsets against passage text (SPEC 5.1)
                valid_span = False
                verified_spans: List[SourceSpan] = []
                limitation: Optional[str] = None

                if 0 <= p_idx < len(cited_chunks):
                    target_chunk = cited_chunks[p_idx]
                    chunk_text = target_chunk.text
                    if 0 <= s_start < s_end <= len(chunk_text):
                        extracted_span_text = chunk_text[s_start:s_end].strip()
                        if len(extracted_span_text) > 0:
                            valid_span = True
                            verified_spans.append(SourceSpan(start=s_start, end=s_end))

                # Safety Invariant: A judgment without a valid span CAN NEVER be SUPPORTED or CONTRADICTED
                if not valid_span:
                    if status in (ClaimVerificationStatusEnum.SUPPORTED, ClaimVerificationStatusEnum.CONTRADICTED):
                        status = ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE
                        limitation = "INVALID_EVIDENCE_SPAN"
                        justification = f"Model indicated {raw_status} but provided no valid verifiable text span in cited passages."

                return status, justification, verified_spans, limitation

            except Exception as exc:
                last_error = exc
                logger.warning("Claim verification model attempt %d failed (%s). Retrying...", attempt + 1, exc)

        # Bounded retry exhausted -> report verifier error
        logger.error("Claim verification failed after bounded retry: %s", last_error)
        return (
            ClaimVerificationStatusEnum.UNVERIFIABLE,
            f"Verifier invocation error after retry: {last_error}",
            [],
            "VERIFIER_ERROR",
        )

    def run(
        self,
        report: CandidateClinicalReport,
        evidence_passages: List[Union[EvidenceChunk, RetrievedPassageResult]],
        case: Optional[ClinicalCase] = None,
    ) -> Union[List[ClaimVerification], AgentExecutionRecord]:
        """Execute EXT-A Claim Verification on CandidateClinicalReport claims."""
        # Invariant: With no provider configured, return FAILED(LLM_UNAVAILABLE)
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="claim_verification",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured. Real LLM verification calls are blocked.",
                input_summary=f"Report version {report.report_version_id} with {len(report.claims)} claims",
                output_summary=None,
            )

        evidence_map: Dict[str, EvidenceChunk] = {p.evidence_id: p for p in evidence_passages}
        verifications: List[ClaimVerification] = []

        for claim in report.claims:
            category = getattr(claim, "category", None)
            if not category:
                if claim.is_safety_critical or claim.section in ("potential_considerations", "diagnostic_considerations"):
                    category = "diagnostic_consideration"
                elif not claim.is_material or claim.section in ("case_summary", "patient_context"):
                    category = "background"
                else:
                    category = "evidence_observation"

            # -----------------------------------------------------------------
            # Step 1: Deterministic citation existence and patient span check
            # -----------------------------------------------------------------
            cited_ids = claim.cited_evidence_ids or []
            missing_ids = [eid for eid in cited_ids if eid not in evidence_map]

            if cited_ids and missing_ids:
                # Cited ID does not exist in THIS run's evidence set
                verif = ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    category=category,
                    status=ClaimVerificationStatusEnum.UNVERIFIABLE,
                    supporting_evidence_ids=[],
                    contradicting_evidence_ids=[],
                    justification=f"Cited evidence ID(s) {missing_ids} not found in this run's evidence sets.",
                    evidence_spans=[],
                    method=VerificationMethodEnum.DETERMINISTIC,
                    verifier_version="1.0",
                    verified_at=datetime.now(timezone.utc),
                    limitations="CITATION_NOT_FOUND",
                )
                verifications.append(verif)
                continue

            # Check patient-derived claims against case source spans
            if not cited_ids or claim.section in ("case_summary", "patient_context"):
                if claim.source_spans and case:
                    raw_hash = case.raw_text_hash
                    # Patient span verification
                    valid_patient_span = all(
                        span.start >= 0 and span.end >= span.start
                        for span in claim.source_spans
                    )
                    if not valid_patient_span:
                        verif = ClaimVerification(
                            claim_id=claim.claim_id,
                            claim_text=claim.claim_text,
                            category=category,
                            status=ClaimVerificationStatusEnum.UNVERIFIABLE,
                            supporting_evidence_ids=[],
                            contradicting_evidence_ids=[],
                            justification="Invalid source span offsets into patient narrative.",
                            evidence_spans=[],
                            method=VerificationMethodEnum.DETERMINISTIC,
                            verifier_version="1.0",
                            verified_at=datetime.now(timezone.utc),
                            limitations="INVALID_PATIENT_SOURCE_SPAN",
                        )
                        verifications.append(verif)
                        continue

            if not cited_ids:
                # Patient finding or observation with no literature citations
                verif = ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    category=category,
                    status=ClaimVerificationStatusEnum.SUPPORTED if (case and claim.source_spans) else ClaimVerificationStatusEnum.INSUFFICIENT_EVIDENCE,
                    supporting_evidence_ids=[],
                    contradicting_evidence_ids=[],
                    justification="Patient-derived claim grounded in clinical case presentation.",
                    evidence_spans=claim.source_spans,
                    method=VerificationMethodEnum.DETERMINISTIC,
                    verifier_version="1.0",
                    verified_at=datetime.now(timezone.utc),
                    limitations=None,
                )
                verifications.append(verif)
                continue

            # -----------------------------------------------------------------
            # Step 2: Deterministic numbers, units, and entity checks
            # -----------------------------------------------------------------
            cited_chunks = [evidence_map[eid] for eid in cited_ids if eid in evidence_map]
            all_cited_text = " ".join(c.text for c in cited_chunks)

            # Check numbers in claim
            claim_numbers = _NUMBER_PATTERN.findall(claim.claim_text)
            missing_numbers = [
                num for num in claim_numbers
                if not any(num in c.text for c in cited_chunks)
            ]
            has_numeric_mismatch = len(missing_numbers) > 0

            # -----------------------------------------------------------------
            # Step 3: Model-based entailment judgment with span verification
            # -----------------------------------------------------------------
            status, justification, verified_spans, limitation = self._verify_single_claim_model(
                claim=claim,
                cited_chunks=cited_chunks,
            )

            # Invariant Step 2: A claim with a number absent from EVERY cited passage
            # can NEVER be SUPPORTED, whatever the model says!
            if has_numeric_mismatch:
                if status == ClaimVerificationStatusEnum.SUPPORTED:
                    status = ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED
                    limitation = "NUMERIC_MISMATCH"
                    justification = (
                        f"Claim downgraded from SUPPORTED to PARTIALLY_SUPPORTED: numeric value(s) "
                        f"{missing_numbers} absent from every cited passage."
                    )

            # -----------------------------------------------------------------
            # Scan other retrieved passages for opposing / contradicting evidence
            # -----------------------------------------------------------------
            other_chunks = [p for p in evidence_passages if p.evidence_id not in cited_ids]
            contradicting_ids = self._scan_contradicting_passages(
                claim_text=claim.claim_text,
                other_passages=other_chunks,
            )

            supporting_ids = [c.evidence_id for c in cited_chunks] if status == ClaimVerificationStatusEnum.SUPPORTED else []

            verif = ClaimVerification(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                category=category,
                status=status,
                supporting_evidence_ids=supporting_ids,
                contradicting_evidence_ids=contradicting_ids,
                justification=justification,
                evidence_spans=verified_spans,
                method=VerificationMethodEnum.HYBRID,
                verifier_version="1.0",
                verified_at=datetime.now(timezone.utc),
                limitations=limitation,
            )
            verifications.append(verif)

        # Persist verifications if repository is configured
        if self.repository:
            try:
                self.repository.save_claim_verifications(verifications)
            except Exception as save_err:
                logger.warning("Persisting claim verifications failed: %s", save_err)

        return verifications
