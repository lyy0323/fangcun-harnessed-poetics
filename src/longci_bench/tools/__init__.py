"""Tool implementations and registry."""

from .imagery import ImageryRecallTool
from .prosody import ProsodyTool
from .registry import ToolRegistry, build_default_registry
from .repetition import RepetitionTool
from .repeat_repair import REPEAT_REPAIR_TOOL_NAME, RepeatRepairTool
from .style import (
    STYLE_TOOL_NAMES,
    ImageryMapTool,
    StyleContradictionScanTool,
    StyleDatabase,
    StyleRewriteTool,
    StyleVectorizeTool,
    SyntaxMapTool,
    build_style_tools,
)
from .external import CallableTool, ExternalHTTPTool
from .configured import ConfiguredTool, apply_toolset_config
from .fangcun import (
    FANGCUN_AUXILIARY_TOOL_NAMES,
    FANGCUN_BASE_TOOL_NAMES,
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
    LOCAL_POETICS_BASE_TOOL_NAMES,
    LOCAL_POETICS_EXTRA_TOOL_NAMES,
    LOCAL_POETICS_LEXICAL_TOOL_NAMES,
)

PoeticsAllusionSearchTool = FangcunAllusionSearchTool
PoeticsBaseProsodyTool = FangcunBaseProsodyTool
PoeticsCharLookupTool = FangcunCharLookupTool
PoeticsDataClient = FangcunDataClient
PoeticsJudouTool = FangcunJudouTool
PoeticsPhraseSuggestTool = FangcunPhraseSuggestTool
PoeticsProsodyTool = FangcunProsodyTool
PoeticsRhymeListTool = FangcunRhymeListTool
PoeticsRhymeLookupTool = FangcunRhymeLookupTool
PoeticsRuleGuideTool = FangcunRuleGuideTool
PoeticsRuleLookupTool = FangcunRuleLookupTool

__all__ = [
    "ImageryRecallTool",
    "CallableTool",
    "ConfiguredTool",
    "ExternalHTTPTool",
    "FANGCUN_AUXILIARY_TOOL_NAMES",
    "FANGCUN_BASE_TOOL_NAMES",
    "LOCAL_POETICS_BASE_TOOL_NAMES",
    "LOCAL_POETICS_EXTRA_TOOL_NAMES",
    "LOCAL_POETICS_LEXICAL_TOOL_NAMES",
    "FangcunAllusionSearchTool",
    "FangcunBaseProsodyTool",
    "FangcunCharLookupTool",
    "FangcunDataClient",
    "FangcunJudouTool",
    "FangcunPhraseSuggestTool",
    "FangcunProsodyTool",
    "FangcunRhymeListTool",
    "FangcunRhymeLookupTool",
    "FangcunRuleGuideTool",
    "FangcunRuleLookupTool",
    "PoeticsAllusionSearchTool",
    "PoeticsBaseProsodyTool",
    "PoeticsCharLookupTool",
    "PoeticsDataClient",
    "PoeticsJudouTool",
    "PoeticsPhraseSuggestTool",
    "PoeticsProsodyTool",
    "PoeticsRhymeListTool",
    "PoeticsRhymeLookupTool",
    "PoeticsRuleGuideTool",
    "PoeticsRuleLookupTool",
    "ProsodyTool",
    "REPEAT_REPAIR_TOOL_NAME",
    "RepeatRepairTool",
    "RepetitionTool",
    "STYLE_TOOL_NAMES",
    "ImageryMapTool",
    "StyleContradictionScanTool",
    "StyleDatabase",
    "StyleRewriteTool",
    "StyleVectorizeTool",
    "SyntaxMapTool",
    "ToolRegistry",
    "apply_toolset_config",
    "build_default_registry",
    "build_style_tools",
]
