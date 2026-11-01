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

from backend.app.core.llm_exceptions import (
    LLMAuthError,
    LLMCallCapReached,
    LLMInvalidResponse,
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
        self.call_count: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

    def snapshot(self) -> Dict[str, int]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
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
        self._rate_limiter = _TokenBucketRateLimiter(requests_per_minute)
        self._usage = TokenUsage()
        self._call_counter: int = 0  # calls actually sent

        # Persistent async client – created lazily.
        self._client: Optional[httpx.AsyncClient] = None

    # -- public helpers ------------------------------------------------

    def get_usage(self) -> Dict[str, int]:
        """Return a snapshot of accumulated token usage."""
        return self._usage.snapshot()

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
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a plain-text completion."""
        return await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=None,
        )

    async def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate structured JSON validated against *schema*."""
        raw = await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model or self._default_model,
            temperature=temperature,
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
        temperature: float,
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

        gen_config: Dict[str, Any] = {"temperature": temperature}
        effective_max = max_tokens or self._max_output_tokens
        if effective_max:
            gen_config["maxOutputTokens"] = effective_max

        if response_schema is not None:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = _pydantic_to_gemini_schema(
                response_schema
            )

        body["generationConfig"] = gen_config
        return body

    async def _call(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        temperature: float,
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

        last_exception: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):  # attempt 1 .. max+1
            # rate-limit
            await self._rate_limiter.acquire()

            # count the call (before sending, so cap works correctly)
            self._call_counter += 1

            t0 = time.monotonic()
            try:
                resp = await client.post(url, json=body)
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
                    await self._backoff(attempt, retry_after=None)
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
                    await self._backoff(attempt, retry_after=None)
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
                raise LLMInvalidResponse(f"Model or resource not found (HTTP 404)")

            # -- RETRY on 429 and 5xx ---------------------------------
            if status == 429 or status >= 500:
                retry_after = self._parse_retry_after(resp)
                if status == 429:
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
                    await self._backoff(attempt, retry_after=retry_after)
                    continue
                raise last_exception

            # -- 200 OK -----------------------------------------------
            if status != 200:
                raise LLMInvalidResponse(
                    f"Unexpected HTTP status {status}"
                )

            data = resp.json()

            # Record token usage
            usage_meta = data.get("usageMetadata", {})
            input_tokens = usage_meta.get("promptTokenCount", 0)
            output_tokens = usage_meta.get("candidatesTokenCount", 0)
            self._usage.record(input_tokens, output_tokens)

            logger.info(
                "Gemini call completed",
                extra={
                    "model": model,
                    "http_status": 200,
                    "latency_s": round(latency, 3),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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

        text = parts[0].get("text", "")
        if not text.strip():
            raise LLMInvalidResponse("Empty text in response candidate")

        if expecting_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                raise LLMInvalidResponse(
                    "Response is not valid JSON when JSON was requested"
                )

        return text

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> Optional[float]:
        """Extract retry delay from Retry-After header or Gemini error body."""
        # Standard Retry-After header
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                return None

        # Gemini-specific: RetryInfo in error body details
        try:
            body = resp.json()
            details = body.get("error", {}).get("details", [])
            for detail in details:
                if detail.get("@type", "").endswith("RetryInfo"):
                    delay_str = detail.get("retryDelay", "")
                    match = re.match(r"([\d.]+)s", delay_str)
                    if match:
                        return float(match.group(1))
        except Exception:
            return None

        return None

    @staticmethod
    async def _backoff(attempt: int, retry_after: Optional[float]) -> None:
        """Exponential backoff with jitter, honoring Retry-After."""
        import random

        if retry_after is not None and retry_after > 0:
            wait = retry_after
        else:
            base = min(2 ** (attempt - 1), 32)  # 1, 2, 4, 8, ...
            jitter = random.uniform(0, base * 0.5)
            wait = base + jitter

        await asyncio.sleep(wait)
