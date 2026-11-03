"""Google Gemini LLM Provider (Google AI Studio API).

Implements the LLMProvider protocol using the Gemini REST API
(generativelanguage.googleapis.com).  Selected when LLM_PROVIDER=gemini.

SECRETS RULES (non-negotiable):
- The API key travels ONLY in the ``x-goog-api-key`` HTTP header.
- The key, request body, and response body never appear in logs,
  exception messages, test output, docs, or the decision log.
- Logs record only: model name, HTTP status, latency, token counts,
  and a call counter.

Decision log references:
- DEC-042: Gemini REST adapter implementation.
- API docs consulted:
  * https://ai.google.dev/api/generate-content
  * https://ai.google.dev/gemini-api/docs/structured-output
  * https://ai.google.dev/gemini-api/docs/rate-limits
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.llm_exceptions import (
    LLMAuthError,
    LLMCallCapReached,
    LLMInvalidResponse,
    LLMQuotaExhausted,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)

logger = logging.getLogger("clinexus.llm.gemini")

# ---------------------------------------------------------------------------
# Pydantic-to-Gemini schema conversion
# ---------------------------------------------------------------------------

_PY_TO_GEMINI_TYPE = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _pydantic_to_gemini_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic v2 model's JSON Schema to a Gemini responseSchema."""
    raw = model_cls.model_json_schema()
    return _convert_json_schema(raw, raw.get("$defs", {}))


def _convert_json_schema(
    schema: Dict[str, Any], defs: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively convert a JSON Schema node to Gemini schema format."""
    # Resolve $ref
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        return _convert_json_schema(defs[ref_name], defs)

    # Handle anyOf / oneOf (nullable patterns)
    for combo_key in ("anyOf", "oneOf"):
        if combo_key in schema:
            non_null = [s for s in schema[combo_key] if s.get("type") != "null"]
            if non_null:
                result = _convert_json_schema(non_null[0], defs)
                result["nullable"] = True
                return result

    result: Dict[str, Any] = {}
    json_type = schema.get("type", "object")
    result["type"] = _PY_TO_GEMINI_TYPE.get(json_type, "STRING")

    if "description" in schema:
        result["description"] = schema["description"]

    if "enum" in schema:
        result["enum"] = schema["enum"]

    if json_type == "object":
        props = schema.get("properties", {})
        if props:
            result["properties"] = {
                k: _convert_json_schema(v, defs) for k, v in props.items()
            }
        req = schema.get("required")
        if req:
            result["required"] = req

    elif json_type == "array":
        items = schema.get("items", {})
        if items:
            result["items"] = _convert_json_schema(items, defs)

    return result


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _TokenBucketRateLimiter:
    """Simple async token-bucket rate limiter.

    Configured by ``requests_per_minute`` (> 0).  The bucket refills
    continuously; ``acquire()`` awaits until a token is available.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError(
                f"LLM_REQUESTS_PER_MINUTE must be > 0, got {requests_per_minute}"
            )
        self._interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_allowed: float = 0.0

    async def acquire(self) -> float:
        """Wait until the next call is allowed.  Returns the wait time in seconds."""
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval
            return wait


# ---------------------------------------------------------------------------
# Token usage accumulator
# ---------------------------------------------------------------------------

class TokenUsage:
    """Per-run accumulator for LLM token counts."""

    def __init__(self) -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_thinking_tokens: int = 0
        self.call_count: int = 0

    def record(
        self, input_tokens: int, output_tokens: int, thinking_tokens: int = 0
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_thinking_tokens += thinking_tokens
        self.call_count += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_thinking_tokens": self.total_thinking_tokens,
            "call_count": self.call_count,
        }


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# finishReason values that indicate the output is NOT usable.
_BAD_FINISH_REASONS = {"SAFETY", "RECITATION", "OTHER", "REPEATED_CONTENT"}


class GeminiProvider:
    """LLM provider backed by the Google Gemini REST API.

    Implements the ``LLMProvider`` protocol (``generate_text``,
    ``generate_json``).

    Security: the API key is stored in ``_api_key`` but is NEVER
    logged, included in exception messages, repr, or str output.
    It travels only in the ``x-goog-api-key`` HTTP header.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str,
        timeout_s: int = 15,
        max_retries: int = 3,
        requests_per_minute: int = 10,
        max_output_tokens: Optional[int] = None,
        eval_max_llm_calls: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_level: Optional[str] = None,
        max_retry_wait_s: Optional[float] = None,
        max_call_seconds: Optional[float] = None,
    ) -> None:
        if not api_key:
            raise LLMAuthError(
                "LLM_API_KEY is required when LLM_PROVIDER=gemini"
            )
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._eval_max_llm_calls = eval_max_llm_calls

        # Temperature setting (default unset)
        self._temperature = (
            temperature
            if temperature is not None
            else getattr(settings, "LLM_TEMPERATURE", None)
        )

        # Thinking level setting (LOW, MEDIUM, HIGH only, default unset)
        resolved_thinking = (
            thinking_level
            if thinking_level is not None
            else getattr(settings, "LLM_THINKING_LEVEL", None)
        )
        if resolved_thinking is not None:
            clean_tl = resolved_thinking.strip().upper()
            if clean_tl not in {"LOW", "MEDIUM", "HIGH"}:
                raise ValueError(
                    f"LLM_THINKING_LEVEL must be one of ['HIGH', 'LOW', 'MEDIUM'], got {resolved_thinking!r}"
                )
            self._thinking_level: Optional[str] = clean_tl
        else:
            self._thinking_level = None

        # Retry wait cap setting (default 30, must be > 0)
        self._max_retry_wait_s: float = (
            float(max_retry_wait_s)
            if max_retry_wait_s is not None
            else float(getattr(settings, "LLM_MAX_RETRY_WAIT_S", 30.0))
        )
        if self._max_retry_wait_s <= 0:
            raise ValueError(
                f"LLM_MAX_RETRY_WAIT_S must be > 0, got {self._max_retry_wait_s}"
            )

        # Call total time bound (default 120, must be > 0)
        self._max_call_seconds: float = (
            float(max_call_seconds)
            if max_call_seconds is not None
            else float(getattr(settings, "LLM_MAX_CALL_SECONDS", 120.0))
        )
        if self._max_call_seconds <= 0:
            raise ValueError(
                f"LLM_MAX_CALL_SECONDS must be > 0, got {self._max_call_seconds}"
            )

        self._rate_limiter = _TokenBucketRateLimiter(requests_per_minute)
        self._usage = TokenUsage()
        self._call_counter: int = 0  # calls actually sent

        # Persistent async client – created lazily.
        self._client: Optional[httpx.AsyncClient] = None

    # -- public helpers ------------------------------------------------

    def get_usage(self) -> Dict[str, Any]:
        """Return a snapshot of accumulated token usage including settings metadata."""
        snap = self._usage.snapshot()
        snap["temperature_sent"] = self._temperature is not None
        snap["thinking_level"] = self._thinking_level
        snap["model_name"] = self._default_model
        return snap

    def reset_usage(self) -> None:
        """Reset the per-run token accumulator and call counter."""
        self._usage = TokenUsage()
        self._call_counter = 0

    # -- protocol methods ----------------------------------------------

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a plain-text completion."""
        effective_temp = temperature if temperature is not None else self._temperature
        return await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model or self._default_model,
            temperature=effective_temp,
            max_tokens=max_tokens,
            response_schema=None,
        )

    async def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseModel:
        """Generate structured JSON validated against *schema*."""
        effective_temp = temperature if temperature is not None else self._temperature
        raw = await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model or self._default_model,
            temperature=effective_temp,
            max_tokens=None,
            response_schema=schema,
        )
        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            raise LLMInvalidResponse(
                f"Gemini returned text that does not match the requested schema: {type(exc).__name__}"
            ) from exc

    # -- security: never reveal the key --------------------------------

    def __repr__(self) -> str:
        return (
            f"GeminiProvider(model={self._default_model!r}, "
            f"timeout={self._timeout_s}s, retries={self._max_retries})"
        )

    __str__ = __repr__

    # -- internals -----------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_s),
                headers={
                    "Content-Type": "application/json",
                    # KEY ONLY IN HEADER – never in URL/query.
                    "x-goog-api-key": self._api_key,
                },
            )
        return self._client

    def _build_request_body(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_schema: Optional[Type[BaseModel]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }

        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        gen_config: Dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature

        if self._thinking_level:
            gen_config["thinkingConfig"] = {
                "thinkingLevel": self._thinking_level,
            }

        effective_max = max_tokens or self._max_output_tokens
        if effective_max:
            gen_config["maxOutputTokens"] = effective_max

        if response_schema is not None:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = _pydantic_to_gemini_schema(
                response_schema
            )

        if gen_config:
            body["generationConfig"] = gen_config
        return body

    async def _call(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_schema: Optional[Type[BaseModel]],
    ) -> str:
        """Core method: rate-limit, build request, retry loop, extract text."""

        # -- call cap guard ---
        if (
            self._eval_max_llm_calls is not None
            and self._call_counter >= self._eval_max_llm_calls
        ):
            raise LLMCallCapReached(
                f"LLM call cap reached ({self._eval_max_llm_calls} calls)"
            )

        url = f"{_GEMINI_API_BASE}/models/{model}:generateContent"
        body = self._build_request_body(
            prompt, system_instruction, temperature, max_tokens, response_schema
        )
        client = self._get_client()

        call_start = time.monotonic()
        last_exception: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 2):  # attempt 1 .. max+1
            elapsed = time.monotonic() - call_start
            remaining = self._max_call_seconds - elapsed
            if remaining <= 0:
                raise LLMTimeout(
                    f"Call exceeded maximum call time limit of {self._max_call_seconds}s"
                )

            # rate-limit
            await self._rate_limiter.acquire()

            # count the call (before sending, so cap works correctly)
            self._call_counter += 1

            t0 = time.monotonic()
            try:
                req_timeout = min(float(self._timeout_s), max(0.1, remaining))
                resp = await client.post(url, json=body, timeout=req_timeout)
            except httpx.TimeoutException:
                latency = time.monotonic() - t0
                logger.warning(
                    "Gemini request timed out",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "latency_s": round(latency, 3),
                    },
                )
                last_exception = LLMTimeout(
                    f"Request timed out (attempt {attempt})"
                )
                if attempt <= self._max_retries:
                    elapsed_after = time.monotonic() - call_start
                    remaining_after = self._max_call_seconds - elapsed_after
                    if remaining_after <= 0:
                        raise LLMTimeout(
                            f"Call exceeded maximum call time limit of {self._max_call_seconds}s"
                        )
                    await self._backoff(attempt, retry_after=None, max_wait=remaining_after)
                    continue
                raise last_exception from None
            except httpx.HTTPError:
                latency = time.monotonic() - t0
                logger.warning(
                    "Gemini connection error",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "latency_s": round(latency, 3),
                    },
                )
                last_exception = LLMUnavailable(
                    f"Connection error (attempt {attempt})"
                )
                if attempt <= self._max_retries:
                    elapsed_after = time.monotonic() - call_start
                    remaining_after = self._max_call_seconds - elapsed_after
                    if remaining_after <= 0:
                        raise LLMTimeout(
                            f"Call exceeded maximum call time limit of {self._max_call_seconds}s"
                        )
                    await self._backoff(attempt, retry_after=None, max_wait=remaining_after)
                    continue
                raise last_exception from None

            latency = time.monotonic() - t0
            status = resp.status_code

            # -- NO RETRY on 400, 401, 403, 404 -----------------------
            if status == 400:
                logger.warning(
                    "Gemini 400 Bad Request",
                    extra={"model": model, "http_status": 400, "latency_s": round(latency, 3)},
                )
                raise LLMInvalidResponse("Bad request (HTTP 400)")
            if status in (401, 403):
                logger.warning(
                    "Gemini auth error",
                    extra={"model": model, "http_status": status, "latency_s": round(latency, 3)},
                )
                raise LLMAuthError(f"Authentication/authorization failed (HTTP {status})")
            if status == 404:
                logger.warning(
                    "Gemini 404 Not Found",
                    extra={"model": model, "http_status": 404, "latency_s": round(latency, 3)},
                )
                raise LLMInvalidResponse("Model or resource not found (HTTP 404)")

            # -- 429 and 5xx ------------------------------------------
            if status == 429 or status >= 500:
                if status == 429:
                    # 1. Quota exhaustion check: per-day quota raises LLM_QUOTA_EXHAUSTED immediately (NO retry)
                    if self._is_daily_quota_exhausted(resp):
                        logger.warning(
                            "Gemini per-day quota exhausted",
                            extra={
                                "model": model,
                                "http_status": 429,
                                "attempt": attempt,
                                "latency_s": round(latency, 3),
                            },
                        )
                        raise LLMQuotaExhausted(
                            "Google Gemini per-day quota exhausted. Please check your API plan and billing at "
                            "https://ai.google.dev/pricing or https://console.cloud.google.com/billing."
                        )

                    retry_after = self._parse_retry_after(resp)

                    # 2. Retry wait cap: if delay exceeds cap, raise LLMRateLimited immediately (NO sleep)
                    if retry_after is not None and retry_after > self._max_retry_wait_s:
                        logger.warning(
                            "Gemini retry delay exceeds cap",
                            extra={
                                "model": model,
                                "http_status": 429,
                                "retry_delay": retry_after,
                                "cap": self._max_retry_wait_s,
                            },
                        )
                        raise LLMRateLimited(
                            f"Rate limit retry delay of {retry_after}s exceeds maximum wait of {self._max_retry_wait_s}s"
                        )

                    logger.warning(
                        "Gemini rate limited",
                        extra={
                            "model": model,
                            "http_status": 429,
                            "attempt": attempt,
                            "latency_s": round(latency, 3),
                        },
                    )
                    last_exception = LLMRateLimited(
                        f"Rate limited (HTTP 429, attempt {attempt})"
                    )
                else:
                    retry_after = self._parse_retry_after(resp)
                    logger.warning(
                        "Gemini server error",
                        extra={
                            "model": model,
                            "http_status": status,
                            "attempt": attempt,
                            "latency_s": round(latency, 3),
                        },
                    )
                    last_exception = LLMUnavailable(
                        f"Server error (HTTP {status}, attempt {attempt})"
                    )

                if attempt <= self._max_retries:
                    elapsed_after = time.monotonic() - call_start
                    remaining_after = self._max_call_seconds - elapsed_after
                    if remaining_after <= 0:
                        raise LLMTimeout(
                            f"Call exceeded maximum call time limit of {self._max_call_seconds}s"
                        )
                    await self._backoff(
                        attempt, retry_after=retry_after, max_wait=remaining_after
                    )
                    continue
                raise last_exception

            # -- 200 OK -----------------------------------------------
            if status != 200:
                raise LLMInvalidResponse(
                    f"Unexpected HTTP status {status}"
                )

            data = resp.json()

            # Record token usage (candidates + thinking tokens for output total)
            usage_meta = data.get("usageMetadata", {})
            input_tokens = usage_meta.get("promptTokenCount", 0)
            candidates_tokens = usage_meta.get("candidatesTokenCount", 0)
            thinking_tokens = (
                usage_meta.get("thoughtsTokenCount", 0)
                or usage_meta.get("reasoningTokenCount", 0)
            )
            output_tokens = candidates_tokens + thinking_tokens
            self._usage.record(input_tokens, output_tokens, thinking_tokens)

            logger.info(
                "Gemini call completed",
                extra={
                    "model": model,
                    "http_status": 200,
                    "latency_s": round(latency, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "call_counter": self._call_counter,
                },
            )

            # Extract text from candidates
            text = self._extract_text(data, expecting_json=response_schema is not None)
            return text

        # Should not reach here, but safety net
        raise LLMUnavailable("All retries exhausted")  # pragma: no cover

    def _extract_text(
        self, data: Dict[str, Any], expecting_json: bool
    ) -> str:
        """Extract text from Gemini response, raising on unusable output."""
        candidates = data.get("candidates")
        if not candidates:
            # Check promptFeedback for blocked prompts
            feedback = data.get("promptFeedback", {})
            block_reason = feedback.get("blockReason", "UNKNOWN")
            raise LLMInvalidResponse(
                f"No candidates in response (prompt blocked: {block_reason})"
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "")

        if finish_reason in _BAD_FINISH_REASONS:
            raise LLMInvalidResponse(
                f"Response blocked or unusable (finishReason={finish_reason})"
            )

        if finish_reason == "MAX_TOKENS":
            raise LLMInvalidResponse(
                "Output truncated (finishReason=MAX_TOKENS)"
            )

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise LLMInvalidResponse("Empty content parts in response")

        # Join text of response parts that are not marked as thoughts
        non_thought_parts = [
            p.get("text", "")
            for p in parts
            if not p.get("thought", False) and p.get("text")
        ]
        text = "".join(non_thought_parts).strip()
        if not text:
            raise LLMInvalidResponse("No non-thought text in response candidate")

        if expecting_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                raise LLMInvalidResponse(
                    "Response is not valid JSON when JSON was requested"
                )

        return text

    @staticmethod
    def _is_daily_quota_exhausted(resp: httpx.Response) -> bool:
        """Inspect error details to detect per-day quota exhaustion."""
        try:
            body = resp.json()
        except Exception:
            return False

        error = body.get("error", {})
        if not isinstance(error, dict):
            return False

        details = error.get("details", [])
        if not isinstance(details, list):
            details = []

        day_terms = ("day", "daily", "per_day", "per-day", "perday")

        for detail in details:
            if not isinstance(detail, dict):
                continue

            # Check metadata fields (e.g. ErrorInfo)
            metadata = detail.get("metadata", {})
            if isinstance(metadata, dict):
                for k, v in metadata.items():
                    val_str = str(v).lower()
                    key_str = str(k).lower()
                    if "quota" in key_str or "metric" in key_str or "limit" in key_str:
                        if any(term in val_str for term in day_terms):
                            return True
                    if any(term in val_str for term in ("requestsperday", "tokensperday", "perday", "per_day", "per-day", "daily")):
                        return True

            # Check QuotaFailure violations
            violations = detail.get("violations", [])
            if isinstance(violations, list):
                for v in violations:
                    if isinstance(v, dict):
                        v_str = (
                            str(v.get("description", ""))
                            + " "
                            + str(v.get("subject", ""))
                            + " "
                            + str(v.get("metric", ""))
                        ).lower()
                        if any(term in v_str for term in day_terms):
                            return True

            # Inspect all string values in the detail
            for k, v in detail.items():
                if isinstance(v, str):
                    v_lower = v.lower()
                    if any(term in v_lower for term in ("requestsperday", "tokensperday", "per_day", "per-day", "perday")) and "quota" in v_lower:
                        return True

        # Check top-level error message
        msg = str(error.get("message", "")).lower()
        if "quota" in msg and any(term in msg for term in ("per day", "daily", "per-day", "per_day")):
            return True

        return False

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> Optional[float]:
        """Extract retry delay from Retry-After header or Gemini error body."""
        # Standard Retry-After header
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass

        # Gemini-specific: RetryInfo in error body details
        try:
            body = resp.json()
            details = body.get("error", {}).get("details", [])
            for detail in details:
                if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
                    delay_val = detail.get("retryDelay") or detail.get("retry_delay")
                    if isinstance(delay_val, (int, float)):
                        return float(delay_val)
                    if isinstance(delay_val, str):
                        match = re.match(r"^([\d.]+)s?$", delay_val.strip())
                        if match:
                            return float(match.group(1))
        except Exception:
            return None

        return None

    @staticmethod
    async def _backoff(
        attempt: int,
        retry_after: Optional[float],
        max_wait: Optional[float] = None,
    ) -> None:
        """Exponential backoff with jitter, honoring Retry-After, bounded by max_wait."""
        import random

        if retry_after is not None and retry_after > 0:
            wait = retry_after
        else:
            base = min(2 ** (attempt - 1), 32)  # 1, 2, 4, 8, ...
            jitter = random.uniform(0, base * 0.5)
            wait = base + jitter

        if max_wait is not None and wait > max_wait:
            wait = max_wait

        if wait > 0:
            await asyncio.sleep(wait)
