"""Adapters for future tool implementations.

These adapters keep the rest of the benchmark independent from how a tool is
implemented. The real toolset can be wired in here without changing evaluation,
model orchestration, or HTTP serving code.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional
from urllib import request as urllib_request

from ..schemas import ToolRequest, ToolResponse
from .base import Tool


ToolHandler = Callable[[ToolRequest], ToolResponse]


class CallableTool(Tool):
    def __init__(
        self,
        name: str,
        handler: ToolHandler,
        description: str = "",
        input_schema: Optional[Mapping[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self._handler = handler
        self._input_schema = dict(input_schema or {})

    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema or super().input_schema()

    def run(self, request: ToolRequest) -> ToolResponse:
        response = self._handler(request)
        if isinstance(response, ToolResponse):
            return response
        raise TypeError("CallableTool handler must return ToolResponse")


class ExternalHTTPTool(Tool):
    def __init__(
        self,
        name: str,
        endpoint: str,
        description: str = "",
        timeout: float = 30.0,
        input_schema: Optional[Mapping[str, Any]] = None,
    ):
        self.name = name
        self.endpoint = endpoint
        self.description = description
        self.timeout = timeout
        self._input_schema = dict(input_schema or {})

    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema or super().input_schema()

    def run(self, request: ToolRequest) -> ToolResponse:
        payload = {
            "tool_name": self.name,
            "text": request.text,
            "cipai": request.cipai,
            "metadata": request.metadata,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(http_request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        parsed = ToolResponse.from_mapping(data)
        if not parsed.tool_name:
            parsed.tool_name = self.name
        return parsed


def external_http_tool_from_config(config: Mapping[str, Any]) -> ExternalHTTPTool:
    name = str(config["name"])
    endpoint = str(config["endpoint"])
    return ExternalHTTPTool(
        name=name,
        endpoint=endpoint,
        description=str(config.get("description", "")),
        timeout=float(config.get("timeout", 30.0)),
        input_schema=config.get("input_schema"),
    )
