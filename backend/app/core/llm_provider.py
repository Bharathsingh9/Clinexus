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
