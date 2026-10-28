"""Clinical Data Fusion (Agent 4 - AGT-4, App A1, A8).

Specification Section 4.4, Section 8, Appendix A1, A8:
- Input: ClinicalCase, RetrievedEvidenceSet, ExternalEvidenceSet.
- Output: CandidateClinicalReport (or AgentExecutionRecord if FAILED with LLM_UNAVAILABLE).
- Structured 10 sections:
    1. case_summary
    2. relevant_findings
    3. evidence_supported_observations
    4. potential_considerations
    5. alternative_explanations
    6. contradictory_or_limiting_evidence
    7. missing_information
    8. uncertainty_and_limitations
    9. recommended_evidence_review_steps
    10. references
- Patient facts and evidence placed in separate, labeled prompt blocks.
- Evidence text wrapped in untrusted data blocks: <UNTRUSTED_EVIDENCE data-only="true">.
- Every factual statement is a CandidateClaim (claim_id, text, section, cited_evidence_ids).
- Citation validation: code verifies every cited ID exists in the evidence sets.
  Re-asks LLM once on violation; if still invalid, drops the claim with a recorded warning.
- Forbidden language lint: safety/forbidden_phrases.py checks for imperative treatment or dosing language.
- STRICT SAFETY: No treatment guidance, no prescription, no dosing orders (App A8).
- Pure confidence calculation using baseline weighting (App A1).
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.safety.confidence import compute_confidence
from backend.app.safety.forbidden_phrases import lint_forbidden_phrases
from backend.app.schemas.clinical_case import ClinicalCase
from backend.app.schemas.evidence import ExternalEvidenceSet, QueryStatusEnum, RetrievedEvidenceSet
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum
from backend.app.schemas.reports import (
    CandidateClaim,
    CandidateClinicalReport,
    ReportStatusEnum,
)

logger = logging.getLogger(__name__)

REPORT_SECTIONS = [
    "case_summary",
    "relevant_findings",
    "evidence_supported_observations",
    "potential_considerations",
    "alternative_explanations",
    "contradictory_or_limiting_evidence",
    "missing_information",
    "uncertainty_and_limitations",
    "recommended_evidence_review_steps",
    "references",
]


class ClinicalDataFusion:
    """Agent 4: Multi-source evidence synthesis and clinical report generation."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.model_name = model_name or settings.LLM_MODEL_FUSION

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Call provider handling both sync and async signatures."""
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

    def _build_untrusted_prompt(
        self,
        case: ClinicalCase,
        local_evidence: Optional[RetrievedEvidenceSet],
        external_evidence: Optional[ExternalEvidenceSet],
    ) -> str:
        """Construct prompt with strict data delimitation and untrusted evidence tags (SPEC 8)."""
        prompt_parts: List[str] = []

        # 1. Patient Input Block
        prompt_parts.append("<PATIENT_INPUT>")
        prompt_parts.append(f"Age: {case.age if case.age is not None else 'UNKNOWN'}")
        prompt_parts.append(f"Sex: {case.sex}")
        prompt_parts.append(f"Chief Complaint: {case.chief_complaint}")
        prompt_parts.append(f"Duration: {case.symptom_duration or 'UNKNOWN'}")
        findings_str = "; ".join([
            f"{s.term} ({s.polarity.value})" for s in case.structured_symptoms
        ])
        prompt_parts.append(f"Documented Findings: {findings_str}")
        prompt_parts.append("</PATIENT_INPUT>\n")

        # 2. Untrusted Evidence Block (Local RAG)
        prompt_parts.append('<UNTRUSTED_EVIDENCE source="LOCAL_RAG" data-only="true">')
        if local_evidence and local_evidence.results:
            for p in local_evidence.results[:5]:
                prompt_parts.append(f"[{p.evidence_id}] {p.title}: {p.text}")
        else:
            prompt_parts.append("No local passages available.")
        prompt_parts.append("</UNTRUSTED_EVIDENCE>\n")

        # 3. Untrusted Evidence Block (External PubMed)
        prompt_parts.append('<UNTRUSTED_EVIDENCE source="EXTERNAL_PUBMED" data-only="true">')
        if external_evidence and external_evidence.passages:
            for ep in external_evidence.passages[:5]:
                prompt_parts.append(f"[{ep.evidence_id}] {ep.text}")
        else:
            prompt_parts.append("No external passages available.")
        prompt_parts.append("</UNTRUSTED_EVIDENCE>\n")

        prompt_parts.append(
            "Instructions:\n"
            "Synthesize a differential clinical report into the 10 standard sections.\n"
            "DO NOT include any treatment, prescription, medication orders, or dosing commands.\n"
            "Every factual statement must cite valid evidence IDs from the evidence blocks above.\n"
            "Format output as JSON with keys: 'sections' (mapping section names to text) and 'claims' "
            "(list of objects with 'claim_text', 'section', 'cited_evidence_ids')."
        )

        return "\n".join(prompt_parts)

    def run(
        self,
        case: ClinicalCase,
        local_evidence: Optional[RetrievedEvidenceSet] = None,
        external_evidence: Optional[ExternalEvidenceSet] = None,
        version_number: int = 1,
        case_id: Optional[str] = None,
    ) -> Union[CandidateClinicalReport, AgentExecutionRecord]:
        """Execute Agent 4 synthesis pipeline and return CandidateClinicalReport or FAILED record."""
        # Step 1: Invariant - No silent fallbacks. Unconfigured provider returns FAILED(LLM_UNAVAILABLE).
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="clinical_data_fusion",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured (LLM_PROVIDER unset or 'none'). Real LLM calls are blocked.",
                input_summary=f"Clinical case (age={case.age}, symptoms={len(case.structured_symptoms)})",
                output_summary=None,
            )

        cid = case_id or str(uuid.uuid4())
        report_id = str(uuid.uuid4())
        report_version_id = str(uuid.uuid4())

        # Collect all valid evidence IDs across local and external sets
        valid_evidence_ids: Set[str] = set()
        retrieval_scores: List[float] = []

        if local_evidence and local_evidence.results:
            for p in local_evidence.results:
                valid_evidence_ids.add(p.evidence_id)
                if p.duplicate_evidence_ids:
                    valid_evidence_ids.update(p.duplicate_evidence_ids)
                retrieval_scores.append(p.retrieval_score)

        if external_evidence and external_evidence.passages:
            for ep in external_evidence.passages:
                valid_evidence_ids.add(ep.evidence_id)

        # Step 2: Build untrusted prompt and invoke LLM
        prompt = self._build_untrusted_prompt(case, local_evidence, external_evidence)
        system_prompt = (
            "You are the Clinical Data Fusion agent in Clinexus. Synthesize evidence-based clinical differential "
            "reports adhering to strict safety constraints: NO treatment guidance, NO dosing, NO prescriptions."
        )

        try:
            raw_response = self._invoke_llm(prompt=prompt, system=system_prompt)
            data = json.loads(raw_response.strip())
        except Exception as exc:
            logger.warning("LLM fusion extraction attempt 1 failed (%s). Retrying repair...", exc)
            repair_prompt = f"Previous output was invalid JSON. Reformat into valid JSON with 'sections' and 'claims':\n{prompt}"
            try:
                raw_response = self._invoke_llm(prompt=repair_prompt, system=system_prompt)
                data = json.loads(raw_response.strip())
            except Exception as exc2:
                logger.error("LLM fusion repair failed: %s", exc2)
                return AgentExecutionRecord(
                    agent_name="clinical_data_fusion",
                    status=AgentStatusEnum.FAILED,
                    error_code="INVALID_LLM_OUTPUT",
                    error_message=f"Failed to obtain valid JSON from LLM: {exc2}",
                    input_summary=f"Clinical case (case_id={cid})",
                )

        raw_sections: Dict[str, str] = data.get("sections", {})
        raw_claims: List[Dict[str, Any]] = data.get("claims", [])

        # Step 3: Populate 10 sections ensuring completeness
        sections: Dict[str, str] = {}
        for sec in REPORT_SECTIONS:
            sections[sec] = raw_sections.get(sec, f"No specific findings documented for {sec.replace('_', ' ')}.")

        # Step 4: Forbidden language linting across all sections (SAF-5, App A8)
        combined_text = " ".join(sections.values())
        forbidden_violations = lint_forbidden_phrases(combined_text)
        if forbidden_violations:
            logger.error("Safety violation: forbidden language detected in generated report: %s", forbidden_violations)
            return AgentExecutionRecord(
                agent_name="clinical_data_fusion",
                status=AgentStatusEnum.FAILED,
                error_code="FORBIDDEN_LANGUAGE_DETECTED",
                error_message=f"Report violated safety invariant: {forbidden_violations[0][0]} - {forbidden_violations[0][1]}",
                input_summary=f"Clinical case (case_id={cid})",
            )

        # Step 5: Citation validation & repair retry
        validated_claims: List[CandidateClaim] = []
        invalid_claims_to_repair: List[Dict[str, Any]] = []

        for c in raw_claims:
            c_text = c.get("claim_text", "").strip()
            c_sec = c.get("section", "evidence_supported_observations")
            c_eids = [eid.strip() for eid in c.get("cited_evidence_ids", []) if eid.strip()]

            if not c_text:
                continue

            # Verify every cited ID exists in evidence sets
            unretrieved = [eid for eid in c_eids if eid not in valid_evidence_ids]
            if unretrieved:
                invalid_claims_to_repair.append(c)
            else:
                validated_claims.append(
                    CandidateClaim(
                        claim_id=str(uuid.uuid4()),
                        report_version_id=report_version_id,
                        claim_text=c_text,
                        section=c_sec,
                        cited_evidence_ids=c_eids,
                    )
                )

        # If invalid citations exist, re-ask LLM once to repair citations (SPEC 4.4)
        if invalid_claims_to_repair:
            logger.warning("Detected %d claims with fabricated/unretrieved citations. Re-asking LLM once...", len(invalid_claims_to_repair))
            repair_prompt = (
                f"The following claims cited unretrieved evidence IDs not in the evidence sets: {invalid_claims_to_repair}\n"
                f"Valid evidence IDs available: {sorted(list(valid_evidence_ids))}\n"
                f"Reformat these claims citing ONLY valid evidence IDs, or remove the ungrounded citations.\n"
                f"Output valid JSON list of claims with keys 'claim_text', 'section', 'cited_evidence_ids':"
            )
            try:
                rep_resp = self._invoke_llm(prompt=repair_prompt, system=system_prompt)
                repaired_list = json.loads(rep_resp.strip())
                if isinstance(repaired_list, dict):
                    repaired_list = repaired_list.get("claims", [])
                for rc in repaired_list:
                    rc_text = rc.get("claim_text", "").strip()
                    rc_sec = rc.get("section", "evidence_supported_observations")
                    rc_eids = [eid.strip() for eid in rc.get("cited_evidence_ids", []) if eid.strip()]
                    # Strict second-pass check: drop if still invalid
                    if any(eid not in valid_evidence_ids for eid in rc_eids):
                        logger.warning("Dropping claim citing unretrieved ID on second violation: %s", rc_text)
                    elif rc_text:
                        validated_claims.append(
                            CandidateClaim(
                                claim_id=str(uuid.uuid4()),
                                report_version_id=report_version_id,
                                claim_text=rc_text,
                                section=rc_sec,
                                cited_evidence_ids=rc_eids,
                            )
                        )
            except Exception as repair_err:
                logger.warning("Citation repair invocation failed (%s). Dropping invalid claims.", repair_err)

        # Step 6: Pure confidence computation (SPEC App A1)
        unavailable_sources = 0
        if external_evidence:
            if any(st == QueryStatusEnum.UNAVAILABLE for st in external_evidence.query_statuses.values()):
                unavailable_sources = 1

        missing_fields = 0
        if case.age is None:
            missing_fields += 1
        if case.sex == "UNKNOWN":
            missing_fields += 1
        if not case.chief_complaint:
            missing_fields += 1
        if not case.symptom_duration:
            missing_fields += 1

        conf_res = compute_confidence(
            retrieval_scores=retrieval_scores,
            claim_support_score=None,  # Baseline variant (App A1)
            unavailable_sources=unavailable_sources,
            configured_sources=2,
            missing_case_fields=missing_fields,
            total_case_fields=4,
            is_baseline=True,
        )

        summary_text = sections.get("case_summary", f"Differential evaluation for presenting symptoms of {case.chief_complaint}.")

        return CandidateClinicalReport(
            report_id=report_id,
            report_version_id=report_version_id,
            version_number=version_number,
            case_id=cid,
            summary=summary_text,
            claims=validated_claims,
            sections=sections,
            confidence_score=conf_res.confidence_score,
            status=ReportStatusEnum.DRAFT,
            created_at=datetime.now(timezone.utc),
        )
