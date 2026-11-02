"""AnswerExtractor Agent (Phase 5 Stage C).

Specification Section 9.9, Appendix A7 (Req ID: EVL-6 / AGT-6):
- Pure, tested module that extracts exact answers from verified report claims.
- Receives ONLY question_type, question text, and claims whose verification status
  is SUPPORTED or PARTIALLY_SUPPORTED (claim text and verification status).
- INVARIANT: Never receives gold answers, sample_ids, raw snippet text outside claims,
  or any evaluation-label object.
- Output by question type (JSON validated in code):
    * yesno: "yes", "no", or "ABSTAIN"
    * factoid: up to 5 ranked strings, or "ABSTAIN"
    * list: a list of entity strings, or "ABSTAIN"
    * summary: short text built only from supplied claims, or "ABSTAIN"
  Any other shape is invalid (retried once, then LLM_INVALID_RESPONSE).
- If the workflow abstained, or no SUPPORTED/PARTIALLY_SUPPORTED claim exists,
  the extractor is NOT called: the record is a workflow-level abstention.
- Typed LLM errors give a FAILED execution record with the error code and no fabricated answer.
- Records extractor_version and model_name in every output.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import settings
from backend.app.core.llm_exceptions import (
    LLMInvalidResponse,
    LLMProviderError,
    LLMUnavailable,
)
from backend.app.core.llm_provider import LLMProvider
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum
from backend.app.schemas.reports import ClaimVerificationStatusEnum

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "1.0.0"

_ALLOWED_STATUSES = {
    ClaimVerificationStatusEnum.SUPPORTED,
    ClaimVerificationStatusEnum.PARTIALLY_SUPPORTED,
}


class ExtractedReportClaim(BaseModel):
    """Input claim representation containing ONLY claim text and verification status.

    extra='forbid' prevents sample_id, gold data, or raw snippets from entering.
    """
    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(..., min_length=1, description="Factual claim text")
    status: ClaimVerificationStatusEnum = Field(
        ...,
        description="Verification status: must be SUPPORTED or PARTIALLY_SUPPORTED",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: ClaimVerificationStatusEnum) -> ClaimVerificationStatusEnum:
        if v not in _ALLOWED_STATUSES:
            raise ValueError(
                f"AnswerExtractor claims must be SUPPORTED or PARTIALLY_SUPPORTED; received status '{v}'"
            )
        return v


class AnswerExtractorInput(BaseModel):
    """Clean input payload for answer extraction without sample_id or gold leakage."""
    model_config = ConfigDict(extra="forbid")

    question_type: str = Field(..., description="BioASQ question type: yesno | factoid | list | summary")
    question: str = Field(..., min_length=1, description="Clinical or biomedical question text")
    claims: List[ExtractedReportClaim] = Field(
        default_factory=list,
        description="Verified claims with status SUPPORTED or PARTIALLY_SUPPORTED",
    )
    workflow_abstained: bool = Field(
        default=False,
        description="Whether the upstream workflow finalized with an abstention",
    )

    @field_validator("question_type")
    @classmethod
    def validate_qtype(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("yesno", "factoid", "list", "summary"):
            raise ValueError(
                f"Invalid question_type '{v}'. Allowed types are: yesno, factoid, list, summary"
            )
        return clean


class AnswerExtractionResult(BaseModel):
    """Strictly typed extraction result carrying provenance, version, and model name."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question_type: str
    extracted_answer: Union[str, List[str]]
    workflow_abstained: bool = False
    extractor_version: str = EXTRACTOR_VERSION
    model_name: str
    execution_record: Optional[AgentExecutionRecord] = None


def validate_answer_shape(question_type: str, raw_answer: Any) -> Union[str, List[str]]:
    """Strictly validate and normalize the extracted answer shape according to question_type.

    yesno: 'yes', 'no', or 'ABSTAIN'
    factoid: up to 5 ranked strings, or 'ABSTAIN'
    list: a list of entity strings, or 'ABSTAIN'
    summary: short text built only from supplied claims, or 'ABSTAIN'
    Any other shape raises ValueError.
    """
    clean_qtype = question_type.strip().lower()

    if isinstance(raw_answer, str) and raw_answer.strip().upper() == "ABSTAIN":
        return "ABSTAIN"

    if clean_qtype == "yesno":
        if not isinstance(raw_answer, str):
            raise ValueError(f"yesno answer must be a string; got {type(raw_answer).__name__}")
        clean = raw_answer.strip().lower()
        if clean in ("yes", "no"):
            return clean
        if clean == "abstain":
            return "ABSTAIN"
        raise ValueError(f"yesno answer must be 'yes', 'no', or 'ABSTAIN'; got '{raw_answer}'")

    if clean_qtype == "factoid":
        if isinstance(raw_answer, str):
            clean = raw_answer.strip()
            if clean.upper() == "ABSTAIN":
                return "ABSTAIN"
            if clean:
                return [clean]
            raise ValueError("factoid answer cannot be an empty string")
        if isinstance(raw_answer, list):
            if len(raw_answer) == 0:
                raise ValueError("factoid answer list cannot be empty (must contain entities or 'ABSTAIN')")
            raw_answer = raw_answer[:5]
            cleaned_list: List[str] = []
            for item in raw_answer:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"factoid elements must be non-empty strings; got {item!r}")
                cleaned_list.append(item.strip())
            return cleaned_list
        raise ValueError(
            f"factoid answer must be list of strings or 'ABSTAIN'; got {type(raw_answer).__name__}"
        )

    if clean_qtype == "list":
        if isinstance(raw_answer, str):
            clean = raw_answer.strip()
            if clean.upper() == "ABSTAIN":
                return "ABSTAIN"
            if clean:
                return [clean]
            raise ValueError("list answer cannot be an empty string")
        if isinstance(raw_answer, list):
            cleaned_list = []
            for item in raw_answer:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"list elements must be non-empty strings; got {item!r}")
                cleaned_list.append(item.strip())
            if not cleaned_list:
                raise ValueError("list answer cannot be empty (must contain entities or 'ABSTAIN')")
            return cleaned_list
        raise ValueError(
            f"list answer must be list of strings or 'ABSTAIN'; got {type(raw_answer).__name__}"
        )

    if clean_qtype == "summary":
        if not isinstance(raw_answer, str):
            raise ValueError(f"summary answer must be a string; got {type(raw_answer).__name__}")
        clean = raw_answer.strip()
        if clean.upper() == "ABSTAIN":
            return "ABSTAIN"
        if not clean:
            raise ValueError("summary answer cannot be an empty string")
        return clean

    raise ValueError(f"Unknown question_type '{question_type}'")


class AnswerExtractor:
    """AnswerExtractor Agent for benchmark answer extraction from verified report claims.

    Uses dedicated prompt, versioning, per-agent model setting LLM_MODEL_EXTRACTOR,
    and strict output validation with one retry.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.model_name = (
            model_name
            or getattr(settings, "LLM_MODEL_EXTRACTOR", None)
            or getattr(settings, "LLM_MODEL", "gemini-2.0-flash")
        )
        self.extractor_version = EXTRACTOR_VERSION

    def _build_system_prompt(self, question_type: str) -> str:
        return (
            "You are the AnswerExtractor in the Clinexus Clinical Decision Support platform.\n"
            "Your task is to extract the exact answer to a biomedical question strictly and exclusively "
            "from a set of verified, evidence-supported claims.\n"
            "RULES:\n"
            "1. Rely ONLY on the provided claims. Never extrapolate, hallucinate, or use external knowledge.\n"
            "2. If the provided claims do not contain sufficient evidence to answer the question, output 'ABSTAIN'.\n"
            "3. Format your response strictly as valid JSON with a single key 'answer':\n"
            "   - For yesno questions: {\"answer\": \"yes\"} or {\"answer\": \"no\"} or {\"answer\": \"ABSTAIN\"}\n"
            "   - For factoid questions: {\"answer\": [\"entity1\", \"entity2\"]} (up to 5 ranked strings) or {\"answer\": \"ABSTAIN\"}\n"
            "   - For list questions: {\"answer\": [\"entity1\", \"entity2\", ...]} (list of entity strings) or {\"answer\": \"ABSTAIN\"}\n"
            "   - For summary questions: {\"answer\": \"concise paragraph synthesized solely from the claims\"} or {\"answer\": \"ABSTAIN\"}\n"
            "4. Return JSON only with no extra commentary or markdown formatting outside the JSON."
        )

    def _build_user_prompt(self, question: str, question_type: str, claims: List[ExtractedReportClaim]) -> str:
        claims_formatted = "\n".join(
            f"- [{claim.status.value}] {claim.claim_text}" for claim in claims
        )
        return (
            f"QUESTION TYPE: {question_type}\n"
            f"QUESTION: {question}\n\n"
            f"VERIFIED REPORT CLAIMS (SUPPORTED / PARTIALLY_SUPPORTED):\n"
            f"{claims_formatted}\n\n"
            f"Extract the answer adhering strictly to the required schema for {question_type} questions.\n"
            f"JSON:"
        )

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Invoke LLM synchronously or asynchronously preserving event loop context."""
        if self.llm_provider is None:
            raise LLMUnavailable("LLM provider is not configured.")

        method = getattr(self.llm_provider, "generate_text", None)
        if method is None:
            raise LLMUnavailable("Configured provider does not implement generate_text.")

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

    def _parse_raw_llm_output(self, raw_text: str, question_type: str) -> Union[str, List[str]]:
        """Parse raw LLM response text into JSON and validate the answer shape."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"LLM output must be a JSON object; got {type(data).__name__}")

        if "answer" not in data:
            raise ValueError("LLM JSON object missing required key 'answer'")

        return validate_answer_shape(question_type, data["answer"])

    def _extract_with_retry(
        self,
        question: str,
        question_type: str,
        claims: List[ExtractedReportClaim],
    ) -> Union[str, List[str]]:
        """Attempt extraction, retrying once on JSON/schema invalidity before raising LLMInvalidResponse."""
        system_prompt = self._build_system_prompt(question_type)
        user_prompt = self._build_user_prompt(question, question_type, claims)

        # Attempt 1
        try:
            raw_response = self._invoke_llm(prompt=user_prompt, system=system_prompt)
            return self._parse_raw_llm_output(raw_response, question_type)
        except (json.JSONDecodeError, ValueError) as err1:
            logger.warning(
                "AnswerExtractor attempt 1 produced invalid shape for '%s' (%s). Retrying once...",
                question_type,
                err1,
            )
        except LLMProviderError:
            raise

        # Attempt 2 (Repair retry)
        repair_prompt = (
            f"Your previous response was invalid for question_type '{question_type}'.\n"
            f"QUESTION: {question}\n"
            f"CLAIMS:\n" + "\n".join(f"- {c.claim_text}" for c in claims) + "\n\n"
            f"You MUST return valid JSON with a single key 'answer'.\n"
            f"Format required for {question_type}:\n"
            f"- yesno: 'yes', 'no', or 'ABSTAIN'\n"
            f"- factoid: list of up to 5 entity strings, or 'ABSTAIN'\n"
            f"- list: list of entity strings, or 'ABSTAIN'\n"
            f"- summary: short paragraph text, or 'ABSTAIN'\n"
            f"JSON:"
        )
        try:
            raw_response2 = self._invoke_llm(prompt=repair_prompt, system=system_prompt)
            return self._parse_raw_llm_output(raw_response2, question_type)
        except (json.JSONDecodeError, ValueError) as err2:
            logger.error(
                "AnswerExtractor attempt 2 failed to produce valid %s answer (%s).",
                question_type,
                err2,
            )
            raise LLMInvalidResponse(
                f"AnswerExtractor produced invalid answer shape for {question_type} after repair retry: {err2}"
            ) from err2

    def extract(self, input_data: AnswerExtractorInput) -> AnswerExtractionResult:
        """Extract exact answer from verified report claims.

        Adheres strictly to workflow-level abstention: if the workflow abstained,
        or if no SUPPORTED/PARTIALLY_SUPPORTED claims exist, the extractor is NOT called.
        """
        start_time = datetime.now(timezone.utc)
        input_hash = hashlib.sha256(
            f"{input_data.question_type}:{input_data.question}:{len(input_data.claims)}".encode("utf-8")
        ).hexdigest()

        # Invariant 3: If workflow abstained or no supported claims exist, the extractor is NOT called
        if input_data.workflow_abstained or len(input_data.claims) == 0:
            return AnswerExtractionResult(
                question_type=input_data.question_type,
                extracted_answer="ABSTAIN",
                workflow_abstained=True,
                extractor_version=self.extractor_version,
                model_name=self.model_name,
                execution_record=None,
            )

        # Invariant: No provider configured -> FAILED(LLM_UNAVAILABLE)
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            end_time = datetime.now(timezone.utc)
            record = AgentExecutionRecord(
                execution_id=str(uuid.uuid4()),
                agent_name="answer_extractor",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured. Real LLM calls are blocked.",
                started_at=start_time,
                completed_at=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
                input_hash=input_hash,
                output_hash=None,
                input_summary=f"Question type: {input_data.question_type}, {len(input_data.claims)} verified claims",
                output_summary=None,
            )
            return AnswerExtractionResult(
                question_type=input_data.question_type,
                extracted_answer="ABSTAIN",
                workflow_abstained=False,
                extractor_version=self.extractor_version,
                model_name=self.model_name,
                execution_record=record,
            )

        # Execute extraction with bounded repair retry
        try:
            answer = self._extract_with_retry(
                question=input_data.question,
                question_type=input_data.question_type,
                claims=input_data.claims,
            )
            end_time = datetime.now(timezone.utc)
            output_hash = hashlib.sha256(str(answer).encode("utf-8")).hexdigest()

            record = AgentExecutionRecord(
                execution_id=str(uuid.uuid4()),
                agent_name="answer_extractor",
                status=AgentStatusEnum.COMPLETED,
                error_code=None,
                error_message=None,
                started_at=start_time,
                completed_at=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
                input_hash=input_hash,
                output_hash=output_hash,
                input_summary=f"Question type: {input_data.question_type}, {len(input_data.claims)} verified claims",
                output_summary=f"Extracted answer: {answer}",
            )
            return AnswerExtractionResult(
                question_type=input_data.question_type,
                extracted_answer=answer,
                workflow_abstained=False,
                extractor_version=self.extractor_version,
                model_name=self.model_name,
                execution_record=record,
            )
        except LLMProviderError as exc:
            end_time = datetime.now(timezone.utc)
            record = AgentExecutionRecord(
                execution_id=str(uuid.uuid4()),
                agent_name="answer_extractor",
                status=AgentStatusEnum.FAILED,
                error_code=exc.error_code,
                error_message=f"Answer extraction failed: {type(exc).__name__}",
                started_at=start_time,
                completed_at=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
                input_hash=input_hash,
                output_hash=None,
                input_summary=f"Question type: {input_data.question_type}, {len(input_data.claims)} verified claims",
                output_summary=None,
            )
            return AnswerExtractionResult(
                question_type=input_data.question_type,
                extracted_answer="ABSTAIN",
                workflow_abstained=False,
                extractor_version=self.extractor_version,
                model_name=self.model_name,
                execution_record=record,
            )
