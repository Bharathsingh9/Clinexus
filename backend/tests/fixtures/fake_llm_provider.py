"""Fake LLM Provider for Unit and Integration Testing.

Satisfies LLMProvider protocol under backend/tests/ (Rule R6).
"""

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel
from backend.app.core.llm_provider import LLMProvider


class FakeLLMProvider:
    """Deterministic, mock LLM provider for tests (Rule R6)."""

    def __init__(
        self,
        default_text: str = "[SYNTHETIC] Test completion response",
        canned_json_responses: Optional[List[BaseModel]] = None,
        prompt_mapping: Optional[Dict[str, str]] = None,
        text_handler: Optional[Callable[[str, Optional[str]], str]] = None,
    ) -> None:
        self.default_text = default_text
        self.canned_json_responses: List[BaseModel] = canned_json_responses or []
        self.prompt_mapping: Dict[str, str] = prompt_mapping or {}
        self.text_handler = text_handler
        self.calls: List[Dict[str, Any]] = []

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.calls.append({
            "method": "generate_text",
            "prompt": prompt,
            "system_instruction": system_instruction,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.text_handler is not None:
            return self.text_handler(prompt, system_instruction)

        combined = f"{system_instruction or ''}\n{prompt}"
        for key, val in self.prompt_mapping.items():
            if key in combined:
                return val

        return self.default_text

    async def generate_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseModel:
        self.calls.append({
            "method": "generate_json",
            "prompt": prompt,
            "schema": schema.__name__,
            "system_instruction": system_instruction,
            "model": model,
            "temperature": temperature,
        })
        if self.canned_json_responses:
            resp = self.canned_json_responses.pop(0)
            if isinstance(resp, schema):
                return resp
            # Try validating dict or object
            return schema.model_validate(resp)
        raise RuntimeError("FakeLLMProvider has no configured canned JSON responses remaining.")
