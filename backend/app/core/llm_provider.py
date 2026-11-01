"""LLM Provider Protocol Definition.

Specification Section 2, Section 6, Rule R6:
- LLMProvider protocol (no concrete network adapter yet in Phase 1).
- Concrete network adapters will be implemented in subsequent phases.
- Test doubles live strictly under backend/tests/.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM completions and structured outputs."""

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a text completion.
        
        Args:
            prompt: User/context prompt text.
            system_instruction: Optional system instruction.
            model: Specific model name override (e.g. per-agent setting).
            temperature: Sampling temperature (default 0.0 for deterministic output).
            max_tokens: Optional maximum tokens in completion.
            
        Returns:
            Generated text string.
        """
        ...

    async def generate_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseModel:
        """Generate and validate structured output against a Pydantic schema.
        
        Args:
            prompt: User/context prompt text.
            schema: Pydantic v2 model class to parse into.
            system_instruction: Optional system instruction.
            model: Specific model name override.
            temperature: Sampling temperature (default 0.0).
            
        Returns:
            Instance of the specified schema model.
        """
        ...


def create_llm_provider(settings_obj=None) -> Optional[LLMProvider]:
    """Factory creating configured LLMProvider based on LLM_PROVIDER setting.

    Selected when LLM_PROVIDER=gemini.
    Raises LLMAuthError if LLM_PROVIDER=gemini and LLM_API_KEY is unset.
    Returns None if LLM_PROVIDER is 'none' or 'fake'.
    """
    if settings_obj is None:
        from backend.app.core.config import settings as default_settings
        settings_obj = default_settings

    provider_name = (settings_obj.LLM_PROVIDER or "").strip().lower()
    if provider_name == "gemini":
        from backend.app.core.gemini_provider import GeminiProvider
        from backend.app.core.llm_exceptions import LLMAuthError

        if not settings_obj.LLM_API_KEY:
            raise LLMAuthError("LLM_API_KEY is required when LLM_PROVIDER=gemini")
        return GeminiProvider(
            api_key=settings_obj.LLM_API_KEY,
            default_model=settings_obj.LLM_MODEL,
            timeout_s=settings_obj.HTTP_TIMEOUT_S,
            max_retries=settings_obj.HTTP_MAX_RETRIES,
            requests_per_minute=settings_obj.LLM_REQUESTS_PER_MINUTE,
            max_output_tokens=settings_obj.LLM_MAX_OUTPUT_TOKENS,
            eval_max_llm_calls=settings_obj.EVAL_MAX_LLM_CALLS,
        )
    return None
