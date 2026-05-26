"""HTTP model client for local model endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request

from ..schemas import ModelResponse
from .base import ModelClient


class HTTPModelClient(ModelClient):
    def __init__(self, endpoint: str, name: str = "http-model", timeout: float = 120.0):
        self.endpoint = endpoint
        self.name = name
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        payload = {
            "prompt": prompt,
            "tools": tools or [],
            "metadata": metadata or {},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return ModelResponse.from_mapping(data)
