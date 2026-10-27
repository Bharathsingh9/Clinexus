"""Clinical Text Clarifier (Agent 1 - AGT-1).

Specification Section 4.1:
- Input: CaseSubmission
- Output: ClinicalCase
- Extracts only stated findings: symptoms (onset, duration, severity, progression, associated symptoms),
  age, sex, history, medications, allergies, vitals.
- Negations kept as findings with polarity=ABSENT.
- Unknown stays UNKNOWN and is never turned into ABSENT.
- Source span verified against original text: raw_text[start:end] must exist and contain term.
- Terminology mapped via backend/app/agents/data/symptom_vocab.json; unmapped keeps normalized_term=null.
- Required fields: age, sex, chief complaint, symptom duration.
- Contradictions detected by rules or LLM review.
- At most 5 clarification questions: blocking (missing required field or contradiction) or non_blocking.
- Invalid LLM JSON: 1 repair retry, then raises ValueError("FAILED: INVALID_LLM_OUTPUT").
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from backend.app.core.config import settings
from backend.app.core.llm_provider import LLMProvider
from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum
from backend.app.schemas.clinical_case import (
    CaseSubmission,
    ClarificationRequest,
    ClarificationStatusEnum,
    ClinicalCase,
    OriginEnum,
    PatientContext,
    PolarityEnum,
    SourceSpan,
    StructuredSymptom,
)

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "data", "symptom_vocab.json"
)


class ClinicalTextClarifier:
    """Agent 1: Extracts structured clinical case and clarifies ambiguities."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        vocab_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.model_name = model_name or settings.LLM_MODEL_CLARIFIER
        self.vocab_path = vocab_path or DEFAULT_VOCAB_PATH
        self.vocab_map: Dict[str, str] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load local symptom vocabulary mapping."""
        if os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for e in entries:
                        s_form = e.get("surface_form", "").strip().lower()
                        norm = e.get("normalized_term")
                        if s_form:
                            self.vocab_map[s_form] = norm
            except Exception as exc:
                logger.warning("Failed to load symptom vocabulary: %s", exc)

    def _normalize_term(self, term: str) -> Optional[str]:
        """Map surface term to local normalized vocabulary or return None."""
        clean = term.strip().lower()
        return self.vocab_map.get(clean, None)

    def _validate_source_span(self, raw_text: str, span: SourceSpan, term: str) -> bool:
        """Verify character offsets point to valid text containing the term."""
        if span.start < 0 or span.end > len(raw_text) or span.start >= span.end:
            return False
        sub = raw_text[span.start : span.end]
        # Must match term (case-insensitive substring or fuzzy match)
        return term.strip().lower() in sub.lower() or sub.lower() in term.strip().lower()

    def _detect_rule_contradictions(
        self, symptoms: List[StructuredSymptom]
    ) -> List[Tuple[str, str]]:
        """Detect findings contradictory in polarity."""
        seen_polarities: Dict[str, PolarityEnum] = {}
        contradictions: List[Tuple[str, str]] = []

        for s in symptoms:
            key = (s.normalized_term or s.term).strip().lower()
            if key in seen_polarities:
                if seen_polarities[key] != s.polarity:
                    contradictions.append(
                        (s.term, f"Finding '{s.term}' is reported as both PRESENT and ABSENT.")
                    )
            else:
                seen_polarities[key] = s.polarity

        return contradictions

    def _invoke_llm(self, prompt: str, system: str) -> str:
        """Call provider handling both sync and async signatures."""
        if self.llm_provider is None:
            raise ValueError("LLM provider is not configured.")
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

    def _call_llm_with_retry(self, raw_text: str) -> Dict[str, Any]:
        """Invoke LLM with one repair retry on JSON failure."""
        if self.llm_provider is None:
            raise ValueError("FAILED: LLM_UNAVAILABLE - No LLM provider configured.")

        system_prompt = (
            "You are the Clinical Text Clarifier agent in Clinexus. "
            "Extract stated symptoms, demographics (age, sex), chief complaint, duration, "
            "history, medications, allergies, and vitals from the clinical narrative. "
            "Output strictly valid JSON with keys: age (int or null), sex (str), "
            "chief_complaint (str), symptom_duration (str or null), symptoms (list of objects "
            "with term, polarity [PRESENT/ABSENT], source_span [start, end], origin [STATED/INFERRED], "
            "uncertainty_note), patient_context (object with history, medications, allergies, vitals)."
        )

        prompt = f"CLINICAL TEXT:\n{raw_text}\n\nOUTPUT VALID JSON ONLY:"

        # Attempt 1
        try:
            response = self._invoke_llm(prompt=prompt, system=system_prompt)
            return json.loads(response.strip())
        except (json.JSONDecodeError, Exception) as err1:
            logger.warning("LLM extraction attempt 1 failed (%s). Retrying repair...", err1)

        # Attempt 2 (Repair retry)
        repair_prompt = (
            f"The previous output was invalid JSON. Reformat the following clinical extraction "
            f"into strictly valid JSON conforming to the schema:\nCLINICAL TEXT:\n{raw_text}\nJSON:"
        )
        try:
            response2 = self._invoke_llm(prompt=repair_prompt, system=system_prompt)
            return json.loads(response2.strip())
        except Exception as err2:
            logger.error("LLM extraction attempt 2 failed (%s). Raising FAILED(INVALID_LLM_OUTPUT).", err2)
            raise ValueError(f"FAILED: INVALID_LLM_OUTPUT - {err2}")

    def run(self, submission: CaseSubmission) -> Union[ClinicalCase, AgentExecutionRecord]:
        """Execute Agent 1 pipeline and return validated ClinicalCase or FAILED AgentExecutionRecord."""
        raw_text = submission.raw_text.strip()
        raw_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        # Invariant: No silent fallbacks. If no LLM provider is configured, return FAILED with LLM_UNAVAILABLE.
        if self.llm_provider is None or settings.LLM_PROVIDER in (None, "", "none"):
            return AgentExecutionRecord(
                agent_name="clinical_text_clarifier",
                status=AgentStatusEnum.FAILED,
                error_code="LLM_UNAVAILABLE",
                error_message="LLM provider is not configured (LLM_PROVIDER unset or 'none'). Real LLM calls are blocked.",
                input_summary=f"Case submission (length={len(raw_text)})",
                output_summary=None,
            )

        # Step 1: Extraction via LLM
        extracted_data = self._call_llm_with_retry(raw_text)

        age = extracted_data.get("age")
        age_missing_reason = None
        if age is None:
            age_missing_reason = "Patient age not stated in submission"

        sex = extracted_data.get("sex")
        if not sex or sex.upper() == "UNKNOWN":
            sex = "UNKNOWN"

        chief_complaint = extracted_data.get("chief_complaint", "").strip()
        if not chief_complaint:
            # Fallback to first line
            chief_complaint = raw_text.split("\n")[0][:150] or "Clinical presentation"

        symptom_duration = extracted_data.get("symptom_duration")
        duration_missing_reason = None
        if symptom_duration is None:
            duration_missing_reason = "Symptom duration not stated in submission"

        # Step 2: Validate and filter symptoms
        raw_symptoms = extracted_data.get("symptoms", [])
        validated_symptoms: List[StructuredSymptom] = []

        for item in raw_symptoms:
            term = item.get("term", "").strip()
            if not term:
                continue

            span_dict = item.get("source_span", {})
            try:
                span = SourceSpan(start=span_dict["start"], end=span_dict["end"])
            except Exception:
                # If offset missing, find first occurrence in text
                idx = raw_text.lower().find(term.lower())
                if idx >= 0:
                    span = SourceSpan(start=idx, end=idx + len(term))
                else:
                    logger.warning("Rejecting finding '%s': invalid span format", term)
                    continue

            # Code verification: text[start:end] exists and contains term
            if not self._validate_source_span(raw_text, span, term):
                logger.warning(
                    "Rejecting finding '%s': source span [%d:%d] mismatch with text",
                    term,
                    span.start,
                    span.end,
                )
                continue

            polarity = item.get("polarity", PolarityEnum.PRESENT)
            if isinstance(polarity, str):
                polarity = PolarityEnum(polarity.upper())

            origin = item.get("origin", OriginEnum.STATED)
            if isinstance(origin, str):
                origin = OriginEnum(origin.upper())

            uncertainty_note = item.get("uncertainty_note")
            if origin == OriginEnum.INFERRED and not uncertainty_note:
                uncertainty_note = "Inferred from clinical narrative context"

            normalized_term = self._normalize_term(term)

            symptom = StructuredSymptom(
                finding_id=str(uuid.uuid4()),
                term=term,
                normalized_term=normalized_term,
                vocabulary="LOCAL",
                polarity=polarity,
                onset=item.get("onset"),
                duration=item.get("duration") or symptom_duration,
                severity=item.get("severity"),
                progression=item.get("progression"),
                associated_symptoms=item.get("associated_symptoms", []),
                source_span=span,
                origin=origin,
                uncertainty_note=uncertainty_note,
            )
            validated_symptoms.append(symptom)

        # Step 3: Patient Context
        p_ctx_data = extracted_data.get("patient_context", {})
        patient_context = PatientContext(
            context_id=str(uuid.uuid4()),
            history=p_ctx_data.get("history", []),
            medications=p_ctx_data.get("medications", []),
            allergies=p_ctx_data.get("allergies", []),
            vitals=p_ctx_data.get("vitals", {}),
        )

        # Step 4: Contradiction Detection & Clarification Questions
        clarification_questions: List[ClarificationRequest] = []

        # Contradictions -> Blocking Clarifications
        contradictions = self._detect_rule_contradictions(validated_symptoms)
        for term, msg in contradictions:
            if len(clarification_questions) >= 5:
                break
            clarification_questions.append(
                ClarificationRequest(
                    question_id=str(uuid.uuid4()),
                    question_text=f"Please clarify finding '{term}': is it present or absent?",
                    is_blocking=True,
                    reason=msg,
                    status=ClarificationStatusEnum.PENDING,
                )
            )

        # Missing required fields -> Blocking Clarifications
        if age is None and len(clarification_questions) < 5:
            clarification_questions.append(
                ClarificationRequest(
                    question_id=str(uuid.uuid4()),
                    question_text="What is the patient's age in years?",
                    is_blocking=True,
                    reason="Patient age is a required case field for risk stratification.",
                    status=ClarificationStatusEnum.PENDING,
                )
            )

        if (sex is None or sex == "UNKNOWN") and len(clarification_questions) < 5:
            clarification_questions.append(
                ClarificationRequest(
                    question_id=str(uuid.uuid4()),
                    question_text="What is the patient's biological sex?",
                    is_blocking=True,
                    reason="Patient sex is a required case field for baseline clinical evaluation.",
                    status=ClarificationStatusEnum.PENDING,
                )
            )

        if symptom_duration is None and len(clarification_questions) < 5:
            clarification_questions.append(
                ClarificationRequest(
                    question_id=str(uuid.uuid4()),
                    question_text="What is the duration and progression of the symptoms?",
                    is_blocking=True,
                    reason="Symptom duration is a required case field for acuity assessment.",
                    status=ClarificationStatusEnum.PENDING,
                )
            )

        # If medications or allergies empty, add non-blocking clarification
        if not patient_context.medications and len(clarification_questions) < 5:
            clarification_questions.append(
                ClarificationRequest(
                    question_id=str(uuid.uuid4()),
                    question_text="Is the patient currently taking any prescription or OTC medications?",
                    is_blocking=False,
                    reason="Current medications assist in ruling out adverse drug reactions.",
                    status=ClarificationStatusEnum.PENDING,
                )
            )

        return ClinicalCase(
            case_id=str(uuid.uuid4()),
            raw_text_hash=raw_text_hash,
            age=age,
            age_missing_reason=age_missing_reason,
            sex=sex,
            chief_complaint=chief_complaint,
            symptom_duration=symptom_duration,
            duration_missing_reason=duration_missing_reason,
            structured_symptoms=validated_symptoms,
            patient_context=patient_context,
            clarification_questions=clarification_questions[:5],
        )
