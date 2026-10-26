"""Unit Tests for Settings and Startup Validation.

Specification Section 2, Appendix A6, A7.
"""

import pytest
from pydantic import ValidationError
from backend.app.core.config import (
    AUTHORITATIVE_EMBEDDING_MODEL,
    DOCUMENTED_FALLBACK_EMBEDDING_MODEL,
    Settings,
)


def test_default_settings():
    """Verify default setting values comply with SPEC Section 2 and DEC-012."""
    cfg = Settings()
    assert cfg.MAX_OPTIMIZER_ITERATIONS == 3
    assert cfg.CONFIDENCE_THRESHOLD == 0.65
    assert cfg.RETRIEVAL_TOP_K == 10
    assert cfg.RETRIEVAL_MIN_SCORE == 0.30
    assert cfg.EMBEDDING_MODEL == AUTHORITATIVE_EMBEDDING_MODEL
    assert cfg.EXTERNAL_MODE == "online"


def test_per_agent_model_defaults():
    """Verify per-agent model settings default to LLM_MODEL (SPEC App A7)."""
    cfg = Settings(LLM_MODEL="custom-base-model")
    assert cfg.LLM_MODEL_CLARIFIER == "custom-base-model"
    assert cfg.LLM_MODEL_RAG == "custom-base-model"
    assert cfg.LLM_MODEL_SCANNER == "custom-base-model"
    assert cfg.LLM_MODEL_FUSION == "custom-base-model"
    assert cfg.LLM_MODEL_OPTIMIZER == "custom-base-model"
    assert cfg.VERIFIER_MODEL == "custom-base-model"


def test_per_agent_model_override():
    """Verify individual agents can have independent model settings."""
    cfg = Settings(
        LLM_MODEL="base-model",
        LLM_MODEL_CLARIFIER="clarifier-model",
        VERIFIER_MODEL="verifier-model",
    )
    assert cfg.LLM_MODEL_CLARIFIER == "clarifier-model"
    assert cfg.LLM_MODEL_RAG == "base-model"
    assert cfg.VERIFIER_MODEL == "verifier-model"


def test_max_optimizer_iterations_hard_cap():
    """Verify MAX_OPTIMIZER_ITERATIONS cannot exceed 3."""
    # <= 3 is allowed
    cfg = Settings(MAX_OPTIMIZER_ITERATIONS=2)
    assert cfg.MAX_OPTIMIZER_ITERATIONS == 2

    # > 3 must fail
    with pytest.raises(ValidationError, match="MAX_OPTIMIZER_ITERATIONS cannot exceed 3"):
        Settings(MAX_OPTIMIZER_ITERATIONS=4)


def test_invalid_external_mode():
    """Verify EXTERNAL_MODE must be 'online' or 'offline'."""
    cfg = Settings(EXTERNAL_MODE="offline")
    assert cfg.EXTERNAL_MODE == "offline"

    with pytest.raises(ValidationError, match="EXTERNAL_MODE must be 'online' or 'offline'"):
        Settings(EXTERNAL_MODE="hybrid")


def test_embedding_model_validation():
    """Verify embedding model defaults to DEC-012, allows NeuML fallback, and rejects unknown."""
    # Default is DEC-012
    cfg_default = Settings()
    assert cfg_default.EMBEDDING_MODEL == AUTHORITATIVE_EMBEDDING_MODEL

    # Explicit fallback NeuML is permitted
    cfg_fallback = Settings(EMBEDDING_MODEL=DOCUMENTED_FALLBACK_EMBEDDING_MODEL)
    assert cfg_fallback.EMBEDDING_MODEL == DOCUMENTED_FALLBACK_EMBEDDING_MODEL

    # Unknown model is strictly rejected
    with pytest.raises(ValidationError, match="Invalid EMBEDDING_MODEL"):
        Settings(EMBEDDING_MODEL="unauthorized/random-model")


def test_secrets_have_no_defaults_and_fail_when_unset(monkeypatch):
    """Task 1: Verify SECRET_KEY and ADMIN_PASSWORD have NO defaults and fail when unset."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    # Calling Settings without SECRET_KEY or ADMIN_PASSWORD must fail
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    errors = excinfo.value.errors()
    missing_fields = {e["loc"][0] for e in errors if e.get("type") == "missing"}
    assert "SECRET_KEY" in missing_fields, "SECRET_KEY must be a required field with no default"
    assert "ADMIN_PASSWORD" in missing_fields, "ADMIN_PASSWORD must be a required field with no default"
