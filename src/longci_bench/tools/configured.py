"""Editable toolset views for zero-shot experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from ..schemas import ToolRequest
from .base import Tool

if TYPE_CHECKING:
    from .registry import ToolRegistry


class ConfiguredTool(Tool):
    """Expose a tool with experiment-specific name, description, and defaults."""

    def __init__(
        self,
        wrapped: Tool,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata_defaults: Optional[Mapping[str, Any]] = None,
    ):
        self.wrapped = wrapped
        self.name = name or wrapped.name
        self.description = description if description is not None else wrapped.description
        self.metadata_defaults = dict(metadata_defaults or {})
        self.tool_group = getattr(wrapped, "tool_group", self.tool_group)
        self.usage_stage = getattr(wrapped, "usage_stage", self.usage_stage)
        self.model_exposure = getattr(wrapped, "model_exposure", self.model_exposure)
        self.tags = tuple(getattr(wrapped, "tags", ()))

    def input_schema(self) -> Dict[str, Any]:
        return self.wrapped.input_schema()

    def run(self, request: ToolRequest):
        metadata = dict(self.metadata_defaults)
        metadata.update(request.metadata or {})
        response = self.wrapped.run(
            ToolRequest(
                tool_name=self.wrapped.name,
                text=request.text,
                cipai=request.cipai,
                metadata=metadata,
            )
        )
        response.tool_name = self.name
        return response


def apply_toolset_config(registry: "ToolRegistry", config_path: str) -> "ToolRegistry":
    """Return a registry view shaped by a JSON toolset config."""

    from .registry import ToolRegistry

    config = _load_config(config_path)
    include = _optional_string_list(config.get("include"))
    exclude = set(_optional_string_list(config.get("exclude")) or [])
    tool_configs = dict(config.get("tools") or {})
    source_tools = {name: tool for name, tool in registry.items()}
    selected = ToolRegistry()

    if include is None:
        exposed_names = list(source_tools)
        for exposed_name in tool_configs:
            if exposed_name not in exposed_names:
                exposed_names.append(exposed_name)
    else:
        exposed_names = include

    for exposed_name in exposed_names:
        settings = _tool_settings(tool_configs.get(exposed_name))
        if settings.get("enabled") is False or exposed_name in exclude:
            continue
        source_name = str(settings.get("source") or exposed_name)
        if source_name in exclude:
            continue
        source_tool = source_tools.get(source_name)
        if source_tool is None:
            raise ValueError(f"Toolset config exposes unknown tool '{exposed_name}' from source '{source_name}'")
        selected.register(
            ConfiguredTool(
                source_tool,
                name=exposed_name,
                description=settings.get("description"),
                metadata_defaults=settings.get("metadata_defaults"),
            )
        )

    return selected


def _load_config(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Toolset config must be a JSON object")
    return data


def _tool_settings(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Each toolset tool entry must be an object")
    return dict(value)


def _optional_string_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Toolset include/exclude must be arrays when provided")
    return [str(item) for item in value]
