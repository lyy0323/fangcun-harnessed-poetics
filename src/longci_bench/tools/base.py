"""Base classes for benchmark tools."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..schemas import ToolRequest, ToolResponse


class Tool:
    name = "tool"
    description = ""
    tool_group = "general"
    usage_stage = "runtime"
    model_exposure = "extra"
    tags = ()

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Generated ci text to inspect."},
                "cipai": {"type": "string", "description": "Cipai name when required."},
                "metadata": {"type": "object"},
            },
            "required": ["text"],
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the tool that produced this response."},
                "passed": {"type": "boolean", "description": "Whether the checked input passed this tool."},
                "issues": {
                    "type": "array",
                    "description": "Machine-readable issues, preferably with exact spans or positions.",
                    "items": {"type": "object"},
                },
                "metrics": {"type": "object", "description": "Numeric or aggregate diagnostics."},
                "metadata": {"type": "object", "description": "Tool-specific structured details."},
            },
            "required": ["tool_name", "passed", "issues", "metrics", "metadata"],
        }

    def contract(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tool_group": self.tool_group,
            "usage_stage": self.usage_stage,
            "model_exposure": self.model_exposure,
            "tags": list(self.tags),
            "input_schema": self.input_schema(),
            "output_schema": self.output_schema(),
            "model_tool_schema": self.model_tool_schema(),
        }

    def model_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }

    def request_from_input(self, payload: Mapping[str, Any]) -> ToolRequest:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("Tool input metadata must be a JSON object when provided")
        return ToolRequest(
            tool_name=str(payload.get("tool_name") or self.name),
            text=str(payload.get("text", "")),
            cipai=payload.get("cipai"),
            metadata=dict(metadata),
        )

    def call(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        response = self.run(self.request_from_input(payload))
        if not response.tool_name:
            response.tool_name = self.name
        return response.to_dict()

    def run(self, request: ToolRequest) -> ToolResponse:
        raise NotImplementedError
