"""Unit tests for the Gemini LLM provider.

Tests use httpx.MockTransport at the HTTP boundary – no real network calls.
The API key, request body, and response body must NEVER appear in logs,
exception messages, test output, docs, or the decision log.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from backend.app.core.gemini_provider import GeminiProvider, _pydantic_to_gemini_schema
from backend.app.core.llm_exceptions import (
    LLMAuthError,
    LLMCallCapReached,
    LLMInvalidResponse,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

# A FIXED fake key used ONLY in tests – never a real key.
_TEST_KEY = "test-key-not-real-abc123"
_TEST_MODEL = "gemini-2.0-flash"


class SimpleSchema(BaseModel):
    """A simple schema for structured output tests."""
    ok: bool
    message: str


def _gemini_ok_response(
    text: str = "Hello world",
    finish_reason: str = "STOP",
    input_tokens: int = 5,
    output_tokens: int = 10,
) -> Dict[str, Any]:
    """Build a minimal successful Gemini response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": input_tokens + output_tokens,
        },
        "modelVersion": _TEST_MODEL,
    }


def _gemini_error_response(
    code: int, message: str = "error", status: str = "ERROR",
    retry_delay: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Gemini error body."""
    details: List[Dict[str, Any]] = []
    if retry_delay:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": retry_delay,
        })
    return {
        "error": {
            "code": code,
            "message": message,
            "status": status,
            "details": details,
        }
    }


def _make_transport(responses: List[httpx.Response]) -> httpx.MockTransport:
    """Build a MockTransport that returns the given responses in sequence."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if call_count < len(responses):
            resp = responses[call_count]
            call_count += 1
            return resp
        # Default to 500 if out of responses
        return httpx.Response(500, json=_gemini_error_response(500))

    return httpx.MockTransport(handler)


def _make_provider(
    transport: Optional[httpx.MockTransport] = None,
    max_retries: int = 3,
    requests_per_minute: int = 6000,  # effectively no rate limit in tests
    max_output_tokens: Optional[int] = None,
    eval_max_llm_calls: Optional[int] = None,
) -> GeminiProvider:
    """Create a GeminiProvider with a mocked transport."""
    provider = GeminiProvider(
        api_key=_TEST_KEY,
        default_model=_TEST_MODEL,
        timeout_s=5,
        max_retries=max_retries,
        requests_per_minute=requests_per_minute,
        max_output_tokens=max_output_tokens,
        eval_max_llm_calls=eval_max_llm_calls,
    )
    if transport is not None:
        # Use the provider's _get_client which sets the API key header,
        # then replace the transport on the underlying client.
        provider._client = httpx.AsyncClient(
            transport=transport,
            timeout=5,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": _TEST_KEY,
            },
        )
    return provider


# ---------------------------------------------------------------------------
# Test: success path returns text and records token usage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_success_path_returns_text_and_records_usage():
    """Success path returns text and records token usage."""
    resp = httpx.Response(200, json=_gemini_ok_response(
        text="Generated answer", input_tokens=12, output_tokens=8,
    ))
    provider = _make_provider(transport=_make_transport([resp]))

    result = await provider.generate_text("Hello")
    assert result == "Generated answer"

    usage = provider.get_usage()
    assert usage["total_input_tokens"] == 12
    assert usage["total_output_tokens"] == 8
    assert usage["call_count"] == 1


# ---------------------------------------------------------------------------
# Test: structured mode sends schema and parses valid JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_structured_mode_sends_schema_and_parses_json():
    """Structured mode sends the schema and parses valid JSON."""
    json_text = json.dumps({"ok": True, "message": "works"})
    resp = httpx.Response(200, json=_gemini_ok_response(text=json_text))

    request_bodies: List[Dict[str, Any]] = []

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return resp

    transport = httpx.MockTransport(capturing_handler)
    provider = _make_provider(transport=transport)

    result = await provider.generate_json("Give me JSON", schema=SimpleSchema)
    assert isinstance(result, SimpleSchema)
    assert result.ok is True
    assert result.message == "works"

    # Verify the schema was sent
    body = request_bodies[0]
    gen_config = body["generationConfig"]
    assert gen_config["responseMimeType"] == "application/json"
    assert "responseSchema" in gen_config
    assert gen_config["responseSchema"]["type"] == "OBJECT"


# ---------------------------------------------------------------------------
# Test: 429 then success – waits per backoff and honors Retry-After
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_429_then_success_honors_retry_after():
    """429 then success: waits per backoff and honors Retry-After."""
    rate_limited = httpx.Response(
        429,
        json=_gemini_error_response(429, retry_delay="2s"),
    )
    success = httpx.Response(200, json=_gemini_ok_response())

    transport = _make_transport([rate_limited, success])
    provider = _make_provider(transport=transport, max_retries=3)

    sleep_calls: List[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Don't actually sleep

    with patch("backend.app.core.gemini_provider.asyncio.sleep", mock_sleep):
        result = await provider.generate_text("Test")

    assert result == "Hello world"
    # The backoff should have been called with the Retry-After value of 2.0
    assert any(abs(s - 2.0) < 0.01 for s in sleep_calls), \
        f"Expected a ~2.0s backoff from Retry-After, got sleeps: {sleep_calls}"


# ---------------------------------------------------------------------------
# Test: retries exhausted gives LLM_RATE_LIMITED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retries_exhausted_gives_rate_limited():
    """Retries exhausted on 429 gives LLM_RATE_LIMITED."""
    responses = [
        httpx.Response(429, json=_gemini_error_response(429))
        for _ in range(5)  # more than max_retries+1
    ]
    transport = _make_transport(responses)
    provider = _make_provider(transport=transport, max_retries=2)

    with patch("backend.app.core.gemini_provider.asyncio.sleep", return_value=None):
        with pytest.raises(LLMRateLimited) as exc_info:
            await provider.generate_text("Test")
    assert exc_info.value.error_code == "LLM_RATE_LIMITED"


# ---------------------------------------------------------------------------
# Test: repeated 5xx gives LLM_UNAVAILABLE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_5xx_gives_unavailable():
    """Repeated 5xx gives LLM_UNAVAILABLE."""
    responses = [
        httpx.Response(500, json=_gemini_error_response(500))
        for _ in range(5)
    ]
    transport = _make_transport(responses)
    provider = _make_provider(transport=transport, max_retries=2)

    with patch("backend.app.core.gemini_provider.asyncio.sleep", return_value=None):
        with pytest.raises(LLMUnavailable) as exc_info:
            await provider.generate_text("Test")
    assert exc_info.value.error_code == "LLM_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Test: timeout gives LLM_TIMEOUT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_gives_llm_timeout():
    """Timeout gives LLM_TIMEOUT."""
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out")

    transport = httpx.MockTransport(timeout_handler)
    provider = _make_provider(transport=transport, max_retries=1)

    with patch("backend.app.core.gemini_provider.asyncio.sleep", return_value=None):
        with pytest.raises(LLMTimeout) as exc_info:
            await provider.generate_text("Test")
    assert exc_info.value.error_code == "LLM_TIMEOUT"


# ---------------------------------------------------------------------------
# Test: 401 and 403 give LLM_AUTH_ERROR with no retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_401_gives_auth_error_no_retry():
    """401 gives LLM_AUTH_ERROR with no retry."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(401, json=_gemini_error_response(401))

    transport = httpx.MockTransport(handler)
    provider = _make_provider(transport=transport, max_retries=3)

    with pytest.raises(LLMAuthError) as exc_info:
        await provider.generate_text("Test")
    assert exc_info.value.error_code == "LLM_AUTH_ERROR"
    assert call_count == 1  # No retry


@pytest.mark.asyncio
async def test_403_gives_auth_error_no_retry():
    """403 gives LLM_AUTH_ERROR with no retry."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(403, json=_gemini_error_response(403))

    transport = httpx.MockTransport(handler)
    provider = _make_provider(transport=transport, max_retries=3)

    with pytest.raises(LLMAuthError):
        await provider.generate_text("Test")
    assert call_count == 1


# ---------------------------------------------------------------------------
# Test: 400 gives LLM_INVALID_RESPONSE with no retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_400_gives_invalid_response_no_retry():
    """400 gives LLM_INVALID_RESPONSE with no retry."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json=_gemini_error_response(400))

    transport = httpx.MockTransport(handler)
    provider = _make_provider(transport=transport, max_retries=3)

    with pytest.raises(LLMInvalidResponse) as exc_info:
        await provider.generate_text("Test")
    assert exc_info.value.error_code == "LLM_INVALID_RESPONSE"
    assert call_count == 1


# ---------------------------------------------------------------------------
# Test: empty candidates gives LLM_INVALID_RESPONSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_candidates_gives_invalid_response():
    """Empty candidates give LLM_INVALID_RESPONSE."""
    body = {
        "candidates": [],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0, "totalTokenCount": 5},
    }
    resp = httpx.Response(200, json=body)
    provider = _make_provider(transport=_make_transport([resp]))

    with pytest.raises(LLMInvalidResponse, match="No candidates"):
        await provider.generate_text("Test")


# ---------------------------------------------------------------------------
# Test: safety block gives LLM_INVALID_RESPONSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_block_gives_invalid_response():
    """Safety block gives LLM_INVALID_RESPONSE."""
    body = _gemini_ok_response(finish_reason="SAFETY")
    resp = httpx.Response(200, json=body)
    provider = _make_provider(transport=_make_transport([resp]))

    with pytest.raises(LLMInvalidResponse, match="SAFETY"):
        await provider.generate_text("Test")


# ---------------------------------------------------------------------------
# Test: truncated output gives LLM_INVALID_RESPONSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_truncated_output_gives_invalid_response():
    """Truncated output (MAX_TOKENS) gives LLM_INVALID_RESPONSE."""
    body = _gemini_ok_response(finish_reason="MAX_TOKENS")
    resp = httpx.Response(200, json=body)
    provider = _make_provider(transport=_make_transport([resp]))

    with pytest.raises(LLMInvalidResponse, match="truncated"):
        await provider.generate_text("Test")


# ---------------------------------------------------------------------------
# Test: non-JSON when JSON requested gives LLM_INVALID_RESPONSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_json_when_json_requested_gives_invalid_response():
    """Non-JSON when JSON was requested gives LLM_INVALID_RESPONSE."""
    body = _gemini_ok_response(text="This is not JSON at all")
    resp = httpx.Response(200, json=body)
    provider = _make_provider(transport=_make_transport([resp]))

    with pytest.raises(LLMInvalidResponse, match="not valid JSON"):
        await provider.generate_json("Give JSON", schema=SimpleSchema)


# ---------------------------------------------------------------------------
# Test: call cap stops the next call with LLM_CALL_CAP_REACHED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_cap_stops_next_call():
    """The call cap stops the next call with LLM_CALL_CAP_REACHED
    and counts only sent calls."""
    responses = [
        httpx.Response(200, json=_gemini_ok_response())
        for _ in range(5)
    ]
    transport = _make_transport(responses)
    provider = _make_provider(transport=transport, eval_max_llm_calls=2)

    # First two calls succeed
    await provider.generate_text("call 1")
    await provider.generate_text("call 2")

    # Third call should be capped
    with pytest.raises(LLMCallCapReached) as exc_info:
        await provider.generate_text("call 3")
    assert exc_info.value.error_code == "LLM_CALL_CAP_REACHED"

    # Counter counts only sent calls
    assert provider._call_counter == 2


# ---------------------------------------------------------------------------
# Test: rate limiter spaces calls (patch the clock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limiter_spaces_calls():
    """Rate limiter spaces calls."""
    responses = [
        httpx.Response(200, json=_gemini_ok_response())
        for _ in range(3)
    ]
    transport = _make_transport(responses)
    # 60 requests per minute = 1 per second
    provider = _make_provider(transport=transport, requests_per_minute=60)

    sleep_calls: List[float] = []
    original_sleep = asyncio.sleep

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # Don't actually sleep

    with patch("backend.app.core.gemini_provider.asyncio.sleep", mock_sleep):
        await provider.generate_text("call 1")
        await provider.generate_text("call 2")
        await provider.generate_text("call 3")

    # The rate limiter should have invoked sleep for spacing
    # (may be 0.0 for first call, then ~1s for subsequent)
    # Just verify at least one sleep was for spacing (> 0)
    positive_sleeps = [s for s in sleep_calls if s > 0]
    assert len(positive_sleeps) >= 1, f"Expected rate limiting sleeps, got {sleep_calls}"


# ---------------------------------------------------------------------------
# Test: key never appears in logs, exception text, repr, or URL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_key_never_in_logs_or_exceptions_or_repr(caplog):
    """The key never appears in any captured log record, exception text,
    or repr of the provider or its request, and the key travels only in
    the header (absent from the URL)."""
    requested_urls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        # Verify key is in header
        assert request.headers.get("x-goog-api-key") == _TEST_KEY
        return httpx.Response(200, json=_gemini_ok_response())

    transport = httpx.MockTransport(handler)
    provider = _make_provider(transport=transport)

    with caplog.at_level(logging.DEBUG, logger="clinexus"):
        result = await provider.generate_text("Test")

    # Key must not appear in repr/str
    assert _TEST_KEY not in repr(provider)
    assert _TEST_KEY not in str(provider)

    # Key must not appear in logs
    for record in caplog.records:
        assert _TEST_KEY not in record.getMessage(), \
            f"Key leaked in log: {record.getMessage()}"

    # Key must not appear in the URL
    for url in requested_urls:
        assert _TEST_KEY not in url, f"Key leaked in URL: {url}"


@pytest.mark.asyncio
async def test_key_absent_from_request_url():
    """Explicitly verify that the API key is completely absent from the request URL."""
    captured_urls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        assert request.headers.get("x-goog-api-key") == _TEST_KEY
        return httpx.Response(200, json=_gemini_ok_response())

    transport = httpx.MockTransport(handler)
    provider = _make_provider(transport=transport)
    await provider.generate_text("Clinical test prompt")

    assert len(captured_urls) == 1
    assert _TEST_KEY not in captured_urls[0]
    assert "key=" not in captured_urls[0]



@pytest.mark.asyncio
async def test_key_not_in_error_exception_text():
    """Key does not leak in exception messages on errors."""
    resp = httpx.Response(401, json=_gemini_error_response(401))
    provider = _make_provider(transport=_make_transport([resp]))

    with pytest.raises(LLMAuthError) as exc_info:
        await provider.generate_text("Test")

    assert _TEST_KEY not in str(exc_info.value)
    assert _TEST_KEY not in repr(exc_info.value)


# ---------------------------------------------------------------------------
# Test: patient-like text in the request body never appears in logs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patient_text_not_in_logs(caplog):
    """Patient-like text in the request body never appears in captured logs."""
    patient_text = "Patient John Smith has chest pain and shortness of breath since 3 days"

    resp = httpx.Response(200, json=_gemini_ok_response())
    provider = _make_provider(transport=_make_transport([resp]))

    with caplog.at_level(logging.DEBUG, logger="clinexus"):
        await provider.generate_text(patient_text)

    for record in caplog.records:
        msg = record.getMessage()
        assert patient_text not in msg, \
            f"Patient text leaked in log: {msg}"


# ---------------------------------------------------------------------------
# Test: agents and verifier return FAILED with matching error code
# ---------------------------------------------------------------------------

class _FakeProviderThatRaises:
    """Test double that raises a typed LLM error on any call."""

    def __init__(self, error: LLMProviderError) -> None:
        self._error = error

    async def generate_text(self, *args: Any, **kwargs: Any) -> str:
        raise self._error

    async def generate_json(self, *args: Any, **kwargs: Any) -> BaseModel:
        raise self._error


def _make_fake_settings():
    """Return a dict simulating necessary settings attributes."""
    class FakeSettings:
        LLM_PROVIDER = "gemini"
        LLM_MODEL = "gemini-2.0-flash"
        LLM_MODEL_CLARIFIER = "gemini-2.0-flash"
        LLM_MODEL_RAG = "gemini-2.0-flash"
        LLM_MODEL_SCANNER = "gemini-2.0-flash"
        LLM_MODEL_FUSION = "gemini-2.0-flash"
        LLM_MODEL_OPTIMIZER = "gemini-2.0-flash"
        VERIFIER_MODEL = "gemini-2.0-flash"
    return FakeSettings()


def _get_agent_error_code(agent_name: str, result: Any) -> Optional[str]:
    """Extract error_code from an AgentExecutionRecord result."""
    from backend.app.schemas.executions import AgentExecutionRecord, AgentStatusEnum
    if isinstance(result, AgentExecutionRecord) and result.status == AgentStatusEnum.FAILED:
        return result.error_code
    return None


@pytest.mark.parametrize("error_cls,error_code", [
    (LLMAuthError, "LLM_AUTH_ERROR"),
    (LLMRateLimited, "LLM_RATE_LIMITED"),
    (LLMTimeout, "LLM_TIMEOUT"),
    (LLMInvalidResponse, "LLM_INVALID_RESPONSE"),
    (LLMCallCapReached, "LLM_CALL_CAP_REACHED"),
    (LLMUnavailable, "LLM_UNAVAILABLE"),
])
def test_clarifier_returns_failed_on_provider_error(error_cls, error_code):
    """ClinicalTextClarifier returns FAILED with matching error code when
    the provider raises each typed error."""
    from backend.app.agents.clinical_text_clarifier import ClinicalTextClarifier
    from backend.app.schemas.clinical_case import CaseSubmission
    from backend.app.schemas.executions import AgentStatusEnum, AgentExecutionRecord

    fake_provider = _FakeProviderThatRaises(error_cls(f"Test {error_code}"))
    agent = ClinicalTextClarifier.__new__(ClinicalTextClarifier)
    agent.llm_provider = fake_provider
    agent.model_name = "gemini-2.0-flash"
    agent.symptom_vocab = {}

    submission = CaseSubmission(
        raw_text="A 45-year-old male presents with chest pain for 3 days.",
    )

    with patch("backend.app.agents.clinical_text_clarifier.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.LLM_MODEL_CLARIFIER = "gemini-2.0-flash"
        result = agent.run(submission)

    assert isinstance(result, AgentExecutionRecord)
    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == error_code
    assert result.output_summary is None


@pytest.mark.parametrize("error_cls,error_code", [
    (LLMAuthError, "LLM_AUTH_ERROR"),
    (LLMRateLimited, "LLM_RATE_LIMITED"),
    (LLMTimeout, "LLM_TIMEOUT"),
    (LLMInvalidResponse, "LLM_INVALID_RESPONSE"),
    (LLMCallCapReached, "LLM_CALL_CAP_REACHED"),
    (LLMUnavailable, "LLM_UNAVAILABLE"),
])
def test_fusion_returns_failed_on_provider_error(error_cls, error_code):
    """ClinicalDataFusion returns FAILED with matching error code when
    the provider raises each typed error."""
    from backend.app.agents.clinical_data_fusion import ClinicalDataFusion
    from backend.app.schemas.executions import AgentStatusEnum, AgentExecutionRecord
    from backend.app.schemas.clinical_case import ClinicalCase, PatientContext
    from backend.app.schemas.evidence import RetrievedEvidenceSet, ExternalEvidenceSet

    fake_provider = _FakeProviderThatRaises(error_cls(f"Test {error_code}"))
    agent = ClinicalDataFusion.__new__(ClinicalDataFusion)
    agent.llm_provider = fake_provider
    agent.model_name = "gemini-2.0-flash"

    clinical_case = ClinicalCase(
        case_id="test-case-001",
        raw_text_hash="c" * 64,
        age=45,
        sex="MALE",
        chief_complaint="Chest pain",
        structured_symptoms=[],
        patient_context=PatientContext(),
    )
    local_evidence = RetrievedEvidenceSet(
        query_id="q-local-1",
        results=[],
    )
    external_evidence = ExternalEvidenceSet(
        query_ids=[],
        documents=[],
        passages=[],
    )

    with patch("backend.app.agents.clinical_data_fusion.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.LLM_MODEL_FUSION = "gemini-2.0-flash"
        result = agent.run(clinical_case, local_evidence, external_evidence)

    assert isinstance(result, AgentExecutionRecord)
    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == error_code
    assert result.output_summary is None


@pytest.mark.parametrize("error_cls,error_code", [
    (LLMAuthError, "LLM_AUTH_ERROR"),
    (LLMRateLimited, "LLM_RATE_LIMITED"),
    (LLMTimeout, "LLM_TIMEOUT"),
    (LLMInvalidResponse, "LLM_INVALID_RESPONSE"),
    (LLMCallCapReached, "LLM_CALL_CAP_REACHED"),
    (LLMUnavailable, "LLM_UNAVAILABLE"),
])
def test_rag_analyzer_returns_failed_on_provider_error(error_cls, error_code):
    """SymptomRAGAnalyzer returns FAILED with matching error code when
    the provider raises each typed error during hypothesis generation."""
    from backend.app.agents.symptom_rag_analyzer import SymptomRAGAnalyzer
    from backend.app.schemas.clinical_case import ClinicalCase, PatientContext, StructuredSymptom
    from backend.app.schemas.executions import AgentStatusEnum, AgentExecutionRecord
    from unittest.mock import MagicMock

    fake_provider = _FakeProviderThatRaises(error_cls(f"Test {error_code}"))
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True

    mock_hit = MagicMock()
    mock_hit.score = 0.85
    mock_hit.payload = {
        "evidence_id": "ev:bioasq:1001:0",
        "corpus_version": "v1",
        "embedding_model": "test-model",
    }
    mock_qdrant.search.return_value = [mock_hit]

    mock_embed = MagicMock()
    mock_embed.model_name = "test-embed"
    mock_embed.embed_query.return_value = [0.1] * 768

    agent = SymptomRAGAnalyzer.__new__(SymptomRAGAnalyzer)
    agent.llm_provider = fake_provider
    agent.model_name = "gemini-2.0-flash"
    agent.qdrant_wrapper = mock_qdrant
    agent.embedding_service = mock_embed
    agent.collection_version = 1
    agent.top_k = 10
    agent.min_score = 0.30

    agent._fetch_postgres_passages = MagicMock(
        return_value={
            "ev:bioasq:1001:0": {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "evidence_id": "ev:bioasq:1001:0",
                "text": "Synthetic text about angina.",
                "title": "[SYNTHETIC] Title",
                "source": "BIOASQ",
                "publication_date": "2021",
                "source_location": "abstract",
                "evidence_level": "UNKNOWN",
            }
        }
    )

    from backend.app.schemas.clinical_case import PolarityEnum, SourceSpan, OriginEnum
    clinical_case = ClinicalCase(
        case_id="test-case-001",
        raw_text_hash="c" * 64,
        age=45,
        sex="MALE",
        chief_complaint="Chest pain",
        structured_symptoms=[
            StructuredSymptom(
                term="chest pain",
                normalized_term="chest pain",
                polarity=PolarityEnum.PRESENT,
                source_span=SourceSpan(start=0, end=10),
                origin=OriginEnum.STATED,
            )
        ],
        patient_context=PatientContext(),
    )

    with patch("backend.app.agents.symptom_rag_analyzer.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.LLM_MODEL_RAG = "gemini-2.0-flash"
        mock_settings.MIN_RELEVANT_PASSAGES = 1
        result = agent.run(clinical_case)

    assert isinstance(result, AgentExecutionRecord)
    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == error_code
    assert result.output_summary is None


@pytest.mark.parametrize("error_cls,error_code", [
    (LLMAuthError, "LLM_AUTH_ERROR"),
    (LLMRateLimited, "LLM_RATE_LIMITED"),
    (LLMTimeout, "LLM_TIMEOUT"),
    (LLMInvalidResponse, "LLM_INVALID_RESPONSE"),
    (LLMCallCapReached, "LLM_CALL_CAP_REACHED"),
    (LLMUnavailable, "LLM_UNAVAILABLE"),
])
def test_optimizer_returns_failed_on_provider_error(error_cls, error_code):
    """AdaptiveOptimizer returns FAILED with matching error code when
    the provider raises each typed error during instruction generation."""
    from backend.app.agents.adaptive_optimizer import AdaptiveOptimizer
    from backend.app.schemas.reports import CandidateClinicalReport, ReportStatusEnum
    from backend.app.schemas.executions import AgentStatusEnum, AgentExecutionRecord

    fake_provider = _FakeProviderThatRaises(error_cls(f"Test {error_code}"))
    agent = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
    agent.llm_provider = fake_provider
    agent.model_name = "gemini-2.0-flash"

    report = CandidateClinicalReport(
        case_id="case-1",
        version_number=1,
        summary="Test report summary",
        claims=[],
        sections={},
        confidence_score=0.45,
        status=ReportStatusEnum.DRAFT,
    )

    with patch("backend.app.agents.adaptive_optimizer.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.LLM_MODEL_OPTIMIZER = "gemini-2.0-flash"
        mock_settings.MAX_OPTIMIZER_ITERATIONS = 3
        mock_settings.CONFIDENCE_THRESHOLD = 0.65
        result = agent.run(report=report)

    assert isinstance(result, AgentExecutionRecord)
    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == error_code
    assert result.output_summary is None


@pytest.mark.parametrize("error_cls,error_code", [
    (LLMAuthError, "LLM_AUTH_ERROR"),
    (LLMRateLimited, "LLM_RATE_LIMITED"),
    (LLMTimeout, "LLM_TIMEOUT"),
    (LLMInvalidResponse, "LLM_INVALID_RESPONSE"),
    (LLMCallCapReached, "LLM_CALL_CAP_REACHED"),
    (LLMUnavailable, "LLM_UNAVAILABLE"),
])
def test_verifier_returns_failed_on_provider_error(error_cls, error_code):
    """ClaimVerificationAgent returns FAILED with matching error code when
    the provider raises each typed error."""
    from backend.app.agents.claim_verification_agent import ClaimVerificationAgent
    from backend.app.schemas.reports import CandidateClinicalReport, CandidateClaim, ReportStatusEnum
    from backend.app.schemas.evidence import EvidenceChunk
    from backend.app.schemas.executions import AgentStatusEnum, AgentExecutionRecord

    fake_provider = _FakeProviderThatRaises(error_cls(f"Test {error_code}"))
    agent = ClaimVerificationAgent.__new__(ClaimVerificationAgent)
    agent.llm_provider = fake_provider
    agent.model_name = "gemini-2.0-flash"
    agent.repository = None

    chunk = EvidenceChunk(
        chunk_id="chk-1",
        evidence_id="ev:bioasq:1001:0",
        document_id="doc-1",
        chunk_index=0,
        text="Patient presents with acute chest pain.",
        text_hash="a" * 64,
    )

    claim = CandidateClaim(
        report_version_id="rv-1",
        claim_text="Patient has chest pain.",
        section="evidence_supported_observations",
        is_material=True,
        is_safety_critical=False,
        cited_evidence_ids=["ev:bioasq:1001:0"],
    )

    report = CandidateClinicalReport(
        case_id="case-1",
        report_version_id="rv-1",
        version_number=1,
        summary="Summary",
        claims=[claim],
        sections={},
        confidence_score=0.8,
        status=ReportStatusEnum.DRAFT,
    )

    with patch("backend.app.agents.claim_verification_agent.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.VERIFIER_MODEL = "gemini-2.0-flash"
        result = agent.run(report=report, evidence_passages=[chunk])

    assert isinstance(result, AgentExecutionRecord)
    assert result.status == AgentStatusEnum.FAILED
    assert result.error_code == error_code
    assert result.output_summary is None


def test_scanner_does_not_depend_on_llm_provider():
    """EvidenceBasedScanner is deterministic (NCBI E-utilities) and succeeds without LLM provider."""
    from backend.app.agents.evidence_based_scanner import EvidenceBasedScanner
    from backend.app.schemas.clinical_case import ClinicalCase, PatientContext
    from backend.app.schemas.evidence import ExternalEvidenceSet
    from backend.tests.fixtures.fake_ncbi_client import FakeNCBIClient

    fake_ncbi = FakeNCBIClient(offline=False)
    scanner = EvidenceBasedScanner(ncbi_client=fake_ncbi, llm_provider=None)

    case = ClinicalCase(
        case_id="test-case-001",
        raw_text_hash="c" * 64,
        age=45,
        sex="MALE",
        chief_complaint="Chest pain",
        structured_symptoms=[],
        patient_context=PatientContext(),
    )
    result = scanner.run(case)
    assert isinstance(result, ExternalEvidenceSet)


# ---------------------------------------------------------------------------
# Test: settings validation for LLM_PROVIDER
# ---------------------------------------------------------------------------

def test_settings_rejects_invalid_provider():
    """Settings reject LLM_PROVIDER values other than fake, none, gemini."""
    from pydantic import ValidationError
    from backend.app.core.config import Settings

    with pytest.raises(ValidationError, match="LLM_PROVIDER must be one of"):
        Settings(LLM_PROVIDER="openai")

    with pytest.raises(ValidationError, match="LLM_PROVIDER must be one of"):
        Settings(LLM_PROVIDER="anthropic")


def test_settings_accepts_valid_providers():
    """Settings accept fake, none, gemini."""
    from backend.app.core.config import Settings

    for provider in ("fake", "none", "gemini"):
        cfg = Settings(LLM_PROVIDER=provider)
        assert cfg.LLM_PROVIDER == provider


def test_settings_gemini_without_key_is_allowed_at_settings_level():
    """Settings creation with LLM_PROVIDER=gemini and no LLM_API_KEY succeeds
    (the auth error happens when the provider is instantiated, not at settings level)."""
    from backend.app.core.config import Settings
    cfg = Settings(LLM_PROVIDER="gemini", LLM_API_KEY=None)
    assert cfg.LLM_PROVIDER == "gemini"


def test_gemini_provider_constructor_fails_without_key():
    """GeminiProvider constructor raises LLM_AUTH_ERROR when key is empty."""
    with pytest.raises(LLMAuthError):
        GeminiProvider(api_key="", default_model="gemini-2.0-flash")

    with pytest.raises(LLMAuthError):
        GeminiProvider(api_key=None, default_model="gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Test: pydantic-to-gemini schema conversion
# ---------------------------------------------------------------------------

def test_pydantic_to_gemini_schema():
    """Schema conversion produces valid Gemini responseSchema."""
    schema = _pydantic_to_gemini_schema(SimpleSchema)
    assert schema["type"] == "OBJECT"
    assert "ok" in schema["properties"]
    assert schema["properties"]["ok"]["type"] == "BOOLEAN"
    assert schema["properties"]["message"]["type"] == "STRING"


# ---------------------------------------------------------------------------
# Test: reset_usage clears counters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_usage_clears_counters():
    """reset_usage clears the token accumulator and call counter."""
    resp = httpx.Response(200, json=_gemini_ok_response())
    provider = _make_provider(transport=_make_transport([resp]))

    await provider.generate_text("Test")
    assert provider.get_usage()["call_count"] == 1

    provider.reset_usage()
    assert provider.get_usage()["call_count"] == 0
    assert provider.get_usage()["total_input_tokens"] == 0
    assert provider._call_counter == 0


def test_create_llm_provider_factory():
    """create_llm_provider returns GeminiProvider when configured, fails without key, or returns None."""
    from backend.app.core.llm_provider import create_llm_provider
    from backend.app.core.config import Settings

    # fake -> None
    cfg_fake = Settings(LLM_PROVIDER="fake")
    assert create_llm_provider(cfg_fake) is None

    # none -> None
    cfg_none = Settings(LLM_PROVIDER="none")
    assert create_llm_provider(cfg_none) is None

    # gemini without key -> LLMAuthError
    cfg_gemini_no_key = Settings(LLM_PROVIDER="gemini", LLM_API_KEY=None)
    with pytest.raises(LLMAuthError, match="LLM_API_KEY is required"):
        create_llm_provider(cfg_gemini_no_key)

    # gemini with key -> GeminiProvider instance
    cfg_gemini_with_key = Settings(LLM_PROVIDER="gemini", LLM_API_KEY="valid-secret-key")
    provider = create_llm_provider(cfg_gemini_with_key)
    assert isinstance(provider, GeminiProvider)
    assert provider._default_model == cfg_gemini_with_key.LLM_MODEL


# ---------------------------------------------------------------------------
# Live smoke test (requires network and real key)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_gemini():
    """Live smoke test: sends a harmless non-clinical prompt to Gemini.

    Requires: LLM_PROVIDER=gemini and a valid LLM_API_KEY in the environment.
    Prints only: model name, latency, token counts, JSON parse result.
    NEVER prints: the key, the request body, or the full response.
    """
    import os
    provider_name = os.environ.get("LLM_PROVIDER", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    if provider_name != "gemini" or not api_key:
        pytest.skip("LLM_PROVIDER=gemini and LLM_API_KEY required for live test")

    model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
    provider = GeminiProvider(
        api_key=api_key,
        default_model=model,
        timeout_s=30,
        max_retries=2,
        requests_per_minute=10,
    )

    t0 = time.monotonic()
    result = await provider.generate_text(
        'Return a JSON object with the key "ok" set to true',
        temperature=0.0,
    )
    latency = time.monotonic() - t0

    usage = provider.get_usage()
    json_ok = False
    try:
        parsed = json.loads(result)
        json_ok = parsed.get("ok") is True
    except json.JSONDecodeError:
        pass

    print(f"\n--- Live Smoke Test ---")
    print(f"Model: {model}")
    print(f"Latency: {latency:.3f}s")
    print(f"Input tokens: {usage['total_input_tokens']}")
    print(f"Output tokens: {usage['total_output_tokens']}")
    print(f"JSON parsed OK: {json_ok}")
    print(f"-----------------------")

    assert json_ok, "Expected JSON with ok=true"
