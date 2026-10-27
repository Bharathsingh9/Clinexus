"""Application Configuration and Settings with Startup Validation.

Specification Section 2, Appendix A6, A7.
"""

from typing import List, Optional
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

AUTHORITATIVE_EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DOCUMENTED_FALLBACK_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
ALLOWED_EMBEDDING_MODELS = {AUTHORITATIVE_EMBEDDING_MODEL, DOCUMENTED_FALLBACK_EMBEDDING_MODEL}


class Settings(BaseSettings):
    """Platform configuration settings loaded from environment or .env file."""

    # Project metadata
    PROJECT_NAME: str = "Clinexus"
    ENVIRONMENT: str = "development"

    # Database URLs
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/clinexus",
        description="Async connection string for SQLAlchemy engine"
    )
    DATABASE_SYNC_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/clinexus",
        description="Synchronous connection string for Alembic migrations"
    )

    # Qdrant Vector Database
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_VERSION: int = 1

    # Embedding Model Configuration (SPEC App A6, DEC-012)
    # Default is pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb; NeuML is explicit fallback only.
    EMBEDDING_MODEL: str = Field(
        default=AUTHORITATIVE_EMBEDDING_MODEL,
        description="Default resolved embedding model ID"
    )
    EMBED_BATCH_SIZE: int = 32

    # LLM Provider & Per-Agent Models (SPEC Section 2, Appendix A7)
    LLM_PROVIDER: str = Field(default="fake", description="fake | openai | anthropic | gemini")
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: str = Field(default="fake-model", description="Base default model")
    LLM_MODEL_CLARIFIER: Optional[str] = None
    LLM_MODEL_RAG: Optional[str] = None
    LLM_MODEL_SCANNER: Optional[str] = None
    LLM_MODEL_FUSION: Optional[str] = None
    LLM_MODEL_OPTIMIZER: Optional[str] = None
    VERIFIER_MODEL: Optional[str] = None

    # NCBI E-utilities Configuration
    NCBI_EMAIL: Optional[str] = None
    NCBI_API_KEY: Optional[str] = None

    # Algorithmic & Workflow Defaults (SPEC Section 2)
    MAX_OPTIMIZER_ITERATIONS: int = Field(default=3, description="Hard cap in code, max 3")
    CONFIDENCE_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
    RETRIEVAL_TOP_K: int = Field(default=10, ge=1)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.30, ge=-1.0, le=1.0)
    RETRIEVAL_SCORE_FLOOR: float = Field(
        default=0.30,
        ge=-1.0,
        le=1.0,
        description="[ENGINEERING-DECISION: UNCALIBRATED] Initial uncalibrated lower bound for retrieval strength normalization, to be recalibrated on dev split in Phase 5"
    )
    RETRIEVAL_SCORE_CEILING: float = Field(
        default=0.90,
        ge=-1.0,
        le=1.0,
        description="[ENGINEERING-DECISION: UNCALIBRATED] Initial uncalibrated upper bound for retrieval strength normalization, to be recalibrated on dev split in Phase 5"
    )
    CHUNK_TOKENS: int = Field(default=300, ge=50)
    CHUNK_OVERLAP_TOKENS: int = Field(default=50, ge=0)
    EXTERNAL_MAX_RECORDS_PER_QUERY: int = Field(default=10, ge=1)
    HTTP_TIMEOUT_S: int = Field(default=15, ge=1)
    HTTP_MAX_RETRIES: int = Field(default=3, ge=0)
    MIN_RELEVANT_PASSAGES: int = Field(default=2, ge=1)
    STALE_YEARS: int = Field(default=10, ge=1)
    EXTERNAL_MODE: str = Field(default="online", description="online | offline")

    # Confidence Weights (SPEC Section 2, App A1)
    CONF_WEIGHT_CLAIM_SUPPORT: float = Field(default=0.50, ge=0.0, le=1.0)
    CONF_WEIGHT_RETRIEVAL_STRENGTH: float = Field(default=0.20, ge=0.0, le=1.0)
    CONF_WEIGHT_SOURCE_AVAILABILITY: float = Field(default=0.15, ge=0.0, le=1.0)
    CONF_WEIGHT_INFO_COMPLETENESS: float = Field(default=0.15, ge=0.0, le=1.0)

    # Auth & Admin Credentials (SPEC Section 2, Section 7)
    SECRET_KEY: str = Field(
        ...,
        min_length=32,
        description="Cryptographic secret key for signing JWT tokens; must be set in environment or .env."
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_EMAIL: str = "admin@clinexus.local"
    ADMIN_PASSWORD: str = Field(
        ...,
        min_length=8,
        description="Initial administrator password; must be set in environment or .env."
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=True,
    )

    @field_validator("MAX_OPTIMIZER_ITERATIONS")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        if v > 3:
            raise ValueError(
                f"MAX_OPTIMIZER_ITERATIONS cannot exceed 3 (hard ceiling in code; got {v})"
            )
        if v < 1:
            raise ValueError(f"MAX_OPTIMIZER_ITERATIONS must be >= 1, got {v}")
        return v

    @field_validator("EXTERNAL_MODE")
    @classmethod
    def validate_external_mode(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ("online", "offline"):
            raise ValueError(f"EXTERNAL_MODE must be 'online' or 'offline', got '{v}'")
        return clean_v

    @field_validator("EMBEDDING_MODEL")
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        if v not in ALLOWED_EMBEDDING_MODELS:
            raise ValueError(
                f"Invalid EMBEDDING_MODEL '{v}'. Allowed values are: "
                f"default '{AUTHORITATIVE_EMBEDDING_MODEL}' or explicit fallback '{DOCUMENTED_FALLBACK_EMBEDDING_MODEL}'."
            )
        return v

    @model_validator(mode="after")
    def populate_agent_models(self) -> "Settings":
        """Resolve per-agent model settings defaulting to LLM_MODEL (SPEC App A7)."""
        base = self.LLM_MODEL
        if not self.LLM_MODEL_CLARIFIER:
            self.LLM_MODEL_CLARIFIER = base
        if not self.LLM_MODEL_RAG:
            self.LLM_MODEL_RAG = base
        if not self.LLM_MODEL_SCANNER:
            self.LLM_MODEL_SCANNER = base
        if not self.LLM_MODEL_FUSION:
            self.LLM_MODEL_FUSION = base
        if not self.LLM_MODEL_OPTIMIZER:
            self.LLM_MODEL_OPTIMIZER = base
        if not self.VERIFIER_MODEL:
            self.VERIFIER_MODEL = base

        if self.RETRIEVAL_SCORE_FLOOR >= self.RETRIEVAL_SCORE_CEILING:
            raise ValueError(
                f"RETRIEVAL_SCORE_FLOOR ({self.RETRIEVAL_SCORE_FLOOR}) must be strictly less than "
                f"RETRIEVAL_SCORE_CEILING ({self.RETRIEVAL_SCORE_CEILING})"
            )
        return self


def load_settings() -> Settings:
    """Load settings from environment or .env, failing with a clear message if mandatory secrets are unset."""
    try:
        return Settings()
    except Exception as e:
        err_msg = str(e)
        if "SECRET_KEY" in err_msg or "ADMIN_PASSWORD" in err_msg:
            raise RuntimeError(
                "Startup failed: Mandatory security settings 'SECRET_KEY' and 'ADMIN_PASSWORD' "
                "are unset. They must be explicitly provided via environment variables or .env file "
                "with no default values in code."
            ) from e
        raise


# Global singleton settings instance
settings = load_settings()
