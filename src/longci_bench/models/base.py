"""Unified model client interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import ModelResponse


class ModelClient:
    name = "model"

    def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        raise NotImplementedError
