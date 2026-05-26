"""Deterministic model client for tests and examples."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import ModelResponse, ToolCall
from .base import ModelClient


class DummyModelClient(ModelClient):
    name = "dummy"

    def __init__(self, text: str = "", tool_calls: Optional[List[ToolCall]] = None):
        self.text = text
        self.tool_calls = tool_calls or []

    def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        return ModelResponse(text=self.text or prompt, tool_calls=list(self.tool_calls), metadata=metadata or {})
