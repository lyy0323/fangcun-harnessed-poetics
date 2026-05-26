"""Tool registry used by evaluation, API serving, and model orchestration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..config import load_cipai_specs, load_external_tool_configs, load_imagery_lexicon
from ..schemas import ToolRequest, ToolResponse
from .base import Tool
from .configured import apply_toolset_config
from .external import external_http_tool_from_config
from .fangcun import (
    FangcunAllusionSearchTool,
    FangcunBaseProsodyTool,
    FangcunCharLookupTool,
    FangcunDataClient,
    FangcunJudouTool,
    FangcunPhraseSuggestTool,
    FangcunProsodyTool,
    FangcunRhymeListTool,
    FangcunRhymeLookupTool,
    FangcunRuleGuideTool,
    FangcunRuleLookupTool,
)
from .imagery import ImageryRecallTool
from .prosody import ProsodyTool
from .repetition import RepetitionTool
from .repeat_repair import RepeatRepairTool
from .style import build_style_tools


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def items(self) -> List[Tuple[str, Tool]]:
        return [(name, self._tools[name]) for name in self.names()]

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def run(self, request: ToolRequest) -> ToolResponse:
        return self.get(request.tool_name).run(request)

    def run_mapping(self, payload: Mapping[str, Any], tool_name: Optional[str] = None) -> ToolResponse:
        name, arguments = self._normalize_call_payload(payload, tool_name=tool_name)
        tool = self.get(name)
        return tool.run(tool.request_from_input(dict(arguments, tool_name=name)))

    def call(self, tool_name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self.run_mapping(payload, tool_name=tool_name).to_dict()

    def run_many(self, calls: Iterable[Mapping[str, Any]]) -> List[ToolResponse]:
        return [self.run_mapping(call) for call in calls]

    def call_many(self, calls: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [response.to_dict() for response in self.run_many(calls)]

    def model_tool_schemas(self) -> List[dict]:
        return [tool.model_tool_schema() for tool in self._tools.values()]

    def tool_contracts(self) -> List[dict]:
        return [tool.contract() for _, tool in self.items()]

    def contract(self) -> Dict[str, Any]:
        tools = self.tool_contracts()
        return {"tool_groups": _tool_groups(tools), "tools": tools}

    def _normalize_call_payload(
        self,
        payload: Mapping[str, Any],
        tool_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        name = tool_name or payload.get("tool_name") or payload.get("name")
        if not name:
            raise ValueError("Tool call payload must include tool_name or name")
        raw_arguments = payload.get("arguments")
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            arguments = dict(payload)
            arguments.pop("name", None)
            arguments.pop("arguments", None)
        return str(name), arguments


def build_default_registry(
    spec_path: Optional[str] = None,
    imagery_path: Optional[str] = None,
    external_tool_config_path: Optional[str] = None,
    toolset_config_path: Optional[str] = None,
    poetics_path: Optional[str] = None,
    poetics_config_dir: Optional[str] = None,
    poetics_data_dir: Optional[str] = None,
    poetics_genre: Optional[str] = None,
    poetics_rhyme_book: Optional[str] = None,
    poetics_ensure_longpu: Optional[bool] = None,
    fangcun_path: Optional[str] = None,
    fangcun_config_dir: Optional[str] = None,
    fangcun_data_dir: Optional[str] = None,
    fangcun_genre: str = "Ci",
    fangcun_rhyme_book: str = "Cilinzhengyun",
    fangcun_ensure_longpu: bool = True,
    style_tool_config_path: Optional[str] = None,
) -> ToolRegistry:
    specs = load_cipai_specs(spec_path) if spec_path else {}
    imagery = load_imagery_lexicon(imagery_path) if imagery_path else None
    repetition_tool = RepetitionTool()
    registry = ToolRegistry(
        [
            ProsodyTool(specs=specs),
            repetition_tool,
            ImageryRecallTool(lexicon=imagery),
            *build_style_tools(style_tool_config_path),
            RepeatRepairTool(repetition_tool=repetition_tool),
        ]
    )
    if external_tool_config_path:
        for tool_config in load_external_tool_configs(external_tool_config_path):
            registry.register(external_http_tool_from_config(tool_config))
    effective_path = poetics_path or fangcun_path
    effective_config_dir = poetics_data_dir or poetics_config_dir or fangcun_data_dir or fangcun_config_dir
    effective_genre = poetics_genre or fangcun_genre
    effective_rhyme_book = poetics_rhyme_book or fangcun_rhyme_book
    effective_ensure_longpu = fangcun_ensure_longpu if poetics_ensure_longpu is None else poetics_ensure_longpu
    if effective_path or effective_config_dir:
        fangcun_client = FangcunDataClient(source_path=effective_path, config_dir=effective_config_dir)
        fangcun_prosody = FangcunProsodyTool(
            source_path=effective_path,
            config_dir=effective_config_dir,
            default_genre=effective_genre,
            default_rhyme_book=effective_rhyme_book,
            default_ensure_longpu=effective_ensure_longpu,
            client=fangcun_client,
        )
        registry.register(fangcun_prosody)
        registry.register(FangcunRuleGuideTool(fangcun_client, default_genre=effective_genre, default_ensure_longpu=effective_ensure_longpu))
        registry.register(FangcunJudouTool(fangcun_client, default_genre=effective_genre, default_ensure_longpu=effective_ensure_longpu))
        registry.register(FangcunBaseProsodyTool(fangcun_prosody))
        registry.register(FangcunCharLookupTool(fangcun_client, default_book=effective_rhyme_book))
        registry.register(FangcunRhymeLookupTool(fangcun_client, default_book=effective_rhyme_book))
        registry.register(FangcunRhymeListTool(fangcun_client, default_book=effective_rhyme_book))
        registry.register(FangcunRuleLookupTool(fangcun_client, default_genre=effective_genre))
        registry.register(FangcunPhraseSuggestTool(fangcun_client))
        registry.register(FangcunAllusionSearchTool(fangcun_client))
        registry.register(
            RepeatRepairTool(
                repetition_tool=registry.get("repetition"),
                rule_tool=registry.get("get_rule"),
                judou_tool=registry.get("check_judou"),
                prosody_tool=registry.get("check_prosody"),
            )
        )
    if toolset_config_path:
        registry = apply_toolset_config(registry, toolset_config_path)
    return registry


def _tool_groups(tools: List[dict]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {
        "base_tool": {
            "description": "Minimal model-facing drafting tools.",
            "tools": [],
        },
        "extra_tool": {
            "description": "Optional tools that overlap with or broaden base behavior; keep out of default model exposure unless needed.",
            "tools": [],
        },
        "lexical_tool": {
            "description": "Character, rhyme, phrase, and allusion reference tools with metadata.results output.",
            "tools": [],
        },
        "post_check_tool": {
            "description": "After-generation diagnostic tools, not normal drafting tools.",
            "tools": [],
        },
        "style_tool": {
            "description": "Style vectorization, rewrite, mapping, and contradiction tools.",
            "tools": [],
        },
        "repair_tool": {
            "description": "Revision tools that locate defects and return candidate edits with rule windows.",
            "tools": [],
        },
    }
    for tool in tools:
        group = str(tool.get("tool_group") or "general")
        groups.setdefault(group, {"description": "", "tools": []})
        groups[group]["tools"].append(tool["name"])
    return groups
