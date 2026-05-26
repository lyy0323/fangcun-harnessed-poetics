"""Style-oriented language tools.

The first implementation is deterministic and dependency-free. It keeps the
same public protocol expected from future API-fitted or local-GPU backends.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..schemas import TextSpan, ToolRequest, ToolResponse, issue
from ..text import count_cjk, find_literal_spans, normalized_cjk
from .base import Tool


STYLE_TOOL_NAMES = [
    "style_vectorize",
    "style_rewrite",
    "imagery_map",
    "syntax_map",
    "style_contradiction_scan",
]

STYLE_DIMENSIONS = [
    "imagery_cold_light",
    "imagery_water_travel",
    "sound_distance",
    "diction_grand",
    "domestic_detail",
    "emotional_directness",
    "modern_register",
    "syntax_compression",
]

DIMENSION_TERMS = {
    "imagery_cold_light": ["淡月", "冷月", "疏星", "霜", "雪", "清", "冷", "暗香", "疏影", "苔"],
    "imagery_water_travel": ["江", "水", "舟", "归舟", "孤舟", "岸", "潮", "浦"],
    "sound_distance": ["笛", "角", "砧", "钟", "雁声", "清砧", "短笛"],
    "diction_grand": ["大江", "千古", "万里", "豪", "壮", "风雷", "拍岸", "归去"],
    "domestic_detail": ["帘", "窗", "黄花", "梧桐", "细雨", "酒", "罗衣"],
    "emotional_directness": ["愁", "恨", "悲", "想念", "开心", "伤心", "漂亮"],
    "modern_register": ["开心", "漂亮", "故事", "城市", "照着", "感觉", "非常", "特别"],
    "syntax_compression": ["，", "。", "、"],
}


@dataclass
class StyleProfile:
    style_id: str
    label: str
    vector: Dict[str, float]
    imagery: List[str] = field(default_factory=list)
    syntax_patterns: List[str] = field(default_factory=list)
    avoid_units: List[str] = field(default_factory=list)
    replacements: Dict[str, List[str]] = field(default_factory=dict)


class StyleDatabase:
    def __init__(self, profiles: Optional[Iterable[StyleProfile]] = None):
        self.profiles = {profile.style_id: profile for profile in profiles or _default_profiles()}

    @classmethod
    def from_path(cls, path: Optional[str]) -> "StyleDatabase":
        if not path:
            return cls()
        resolved = Path(path).expanduser()
        if not resolved.exists():
            return cls()
        profiles: List[StyleProfile] = []
        paths = sorted(resolved.glob("*.json")) if resolved.is_dir() else [resolved]
        for json_path in paths:
            profiles.extend(_profiles_from_file(json_path))
        return cls(profiles or _default_profiles())

    def get(self, style_id: Optional[str]) -> StyleProfile:
        if style_id and style_id in self.profiles:
            return self.profiles[style_id]
        return self.profiles["jiang_kui_qingkong"]

    def nearest(self, vector: Mapping[str, float], limit: int = 3) -> List[Dict[str, Any]]:
        scored = []
        for profile in self.profiles.values():
            scored.append(
                {
                    "style_id": profile.style_id,
                    "label": profile.label,
                    "similarity": round(_cosine(vector, profile.vector), 4),
                }
            )
        return sorted(scored, key=lambda item: item["similarity"], reverse=True)[:limit]


class StyleTool(Tool):
    tool_group = "style_tool"
    model_exposure = "style"
    backend = "local_heuristic"

    def __init__(self, style_db: Optional[StyleDatabase] = None, schema_version: Optional[str] = None):
        self.style_db = style_db or StyleDatabase()
        self.schema_version = schema_version or f"{self.name}.v1"

    def _base_metadata(self, request: ToolRequest, operation: Optional[str] = None) -> Dict[str, Any]:
        target_style = _target_style(request.metadata)
        metadata = {
            "operation": operation or self.name,
            "schema_version": self.schema_version,
            "backend": self.backend,
            "cache_key": _cache_key(self.schema_version, request.text, request.cipai, request.metadata),
        }
        if target_style:
            metadata["target_style"] = target_style
            metadata["style_id"] = target_style
        return metadata

    def _missing(self, request: ToolRequest, issue_type: str, message: str) -> ToolResponse:
        return ToolResponse(
            self.name,
            passed=False,
            issues=[issue(issue_type, message)],
            metrics={},
            metadata=self._base_metadata(request),
        )


class StyleVectorizeTool(StyleTool):
    name = "style_vectorize"
    description = (
        "Convert input text into a structured style vector and nearest known style categories. "
        "Use metadata.input_mode='text' and optional metadata.style_db."
    )
    usage_stage = "analysis"
    tags = ("style", "vector", "analysis")

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["text"]["description"] = "Text to vectorize for style analysis."
        schema["properties"]["metadata"]["description"] = (
            "Optional input_mode='text' and style_db name/path selector."
        )
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        if not request.text.strip():
            return self._missing(request, "missing_text", "style_vectorize requires non-empty text.")
        vector = style_vector_for_text(request.text)
        nearest = self.style_db.nearest(vector)
        top_similarity = nearest[0]["similarity"] if nearest else 0.0
        metadata = self._base_metadata(request)
        metadata.update(
            {
                "input_mode": request.metadata.get("input_mode") or "text",
                "style_db": request.metadata.get("style_db"),
                "style_vector": {"dimensions": vector},
                "nearest_styles": nearest,
                "analysis": {
                    "dominant_features": _top_dimensions(vector),
                    "weak_features": _weak_dimensions(vector),
                },
            }
        )
        return ToolResponse(
            self.name,
            passed=bool(nearest),
            issues=[] if nearest else [issue("style_vectorization_failed", "No nearest style could be computed.")],
            metrics={"vector_dimension_count": len(vector), "top_similarity": top_similarity},
            metadata=metadata,
        )


class StyleRewriteTool(StyleTool):
    name = "style_rewrite"
    description = (
        "Rewrite a selected text slot toward a target style under optional prosody constraints. "
        "Returns structured rewrite candidates."
    )
    usage_stage = "draft_operator"
    tags = ("style", "rewrite", "candidate")

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["text"]["description"] = "Text span or phrase to rewrite."
        schema["properties"]["metadata"]["description"] = (
            "Requires slot and target_style; optional prosody_constraints."
        )
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        if not request.text.strip():
            return self._missing(request, "missing_text", "style_rewrite requires non-empty text.")
        slot = request.metadata.get("slot")
        if not slot:
            return self._missing(request, "missing_slot", "style_rewrite requires metadata.slot.")
        target_style = _target_style(request.metadata)
        if not target_style:
            return self._missing(request, "missing_target_style", "style_rewrite requires metadata.target_style.")
        profile = self.style_db.get(target_style)
        constraints = dict(request.metadata.get("prosody_constraints") or {})
        candidates = _rewrite_candidates(request.text, str(slot), profile, constraints)
        if not candidates:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("empty_rewrite_candidates", "No rewrite candidates were produced.")],
                metrics={"candidate_count": 0, "top_style_score": 0.0},
                metadata={**self._base_metadata(request), "candidates": []},
            )
        return ToolResponse(
            self.name,
            passed=True,
            issues=[],
            metrics={
                "candidate_count": len(candidates),
                "top_style_score": candidates[0]["style_score"],
            },
            metadata={**self._base_metadata(request), "candidates": candidates},
        )


class ImageryMapTool(StyleTool):
    name = "imagery_map"
    description = "Map source imagery to replacements compatible with a target style."
    usage_stage = "reference"
    tags = ("style", "imagery", "mapping")

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["text"]["description"] = "Source imagery to replace or adapt."
        schema["properties"]["metadata"]["description"] = "Requires target_style; optional semantic_role."
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        source = request.text.strip()
        if not source:
            return self._missing(request, "missing_source_imagery", "imagery_map requires source imagery in text.")
        target_style = _target_style(request.metadata)
        if not target_style:
            return self._missing(request, "missing_target_style", "imagery_map requires metadata.target_style.")
        profile = self.style_db.get(target_style)
        results = _imagery_results(source, profile, request.metadata.get("semantic_role"))
        if not results:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("imagery_role_mismatch", "No style-compatible imagery replacement was found.")],
                metrics={"result_count": 0, "top_style_score": 0.0},
                metadata={**self._base_metadata(request), "results": []},
            )
        return ToolResponse(
            self.name,
            passed=True,
            issues=[],
            metrics={"result_count": len(results), "top_style_score": results[0]["style_score"]},
            metadata={**self._base_metadata(request), "results": results},
        )


class SyntaxMapTool(StyleTool):
    name = "syntax_map"
    description = "Map a source sentence into style-compatible syntactic alternatives."
    usage_stage = "draft_operator"
    tags = ("style", "syntax", "mapping")

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["text"]["description"] = "Source sentence to map into target-style syntax."
        schema["properties"]["metadata"]["description"] = "Requires target_style."
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        if not request.text.strip():
            return self._missing(request, "missing_sentence", "syntax_map requires a non-empty sentence.")
        target_style = _target_style(request.metadata)
        if not target_style:
            return self._missing(request, "missing_target_style", "syntax_map requires metadata.target_style.")
        profile = self.style_db.get(target_style)
        results = _syntax_results(request.text, profile)
        issues = []
        if not results:
            issues.append(issue("empty_syntax_candidates", "No syntax alternatives were produced."))
        return ToolResponse(
            self.name,
            passed=bool(results),
            issues=issues,
            metrics={"result_count": len(results), "top_style_score": results[0]["style_score"] if results else 0.0},
            metadata={**self._base_metadata(request), "results": results},
        )


class StyleContradictionScanTool(StyleTool):
    name = "style_contradiction_scan"
    description = (
        "Find imagery, syntax, register, or expression patterns that conflict with a target style. "
        "Returns machine-readable spans and suggested operations."
    )
    usage_stage = "post_check"
    tags = ("style", "diagnostic", "scan")

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["text"]["description"] = "Text to scan for target-style contradictions."
        schema["properties"]["metadata"]["description"] = "Requires target_style."
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        if not request.text.strip():
            return self._missing(request, "missing_text", "style_contradiction_scan requires non-empty text.")
        target_style = _target_style(request.metadata)
        if not target_style:
            return self._missing(
                request,
                "missing_target_style",
                "style_contradiction_scan requires metadata.target_style.",
            )
        profile = self.style_db.get(target_style)
        contradictions = _contradictions(request.text, profile)
        issues = [
            issue(
                "style_contradiction",
                f"{item['type']} conflicts with target style.",
                span=TextSpan(**item["span"]),
                severity=item["severity"],
            )
            for item in contradictions
        ]
        max_severity = max((_severity_value(item["severity"]) for item in contradictions), default=0)
        return ToolResponse(
            self.name,
            passed=not contradictions,
            issues=issues,
            metrics={"contradiction_count": len(contradictions), "max_severity": max_severity},
            metadata={**self._base_metadata(request), "contradictions": contradictions},
        )


def build_style_tools(config_path: Optional[str] = None) -> List[Tool]:
    settings = _load_style_config(config_path)
    style_db = StyleDatabase.from_path(settings.get("style_db_path"))
    tool_settings = dict(settings.get("tools") or {})
    return [
        StyleVectorizeTool(style_db, _schema_version(tool_settings, "style_vectorize")),
        StyleRewriteTool(style_db, _schema_version(tool_settings, "style_rewrite")),
        ImageryMapTool(style_db, _schema_version(tool_settings, "imagery_map")),
        SyntaxMapTool(style_db, _schema_version(tool_settings, "syntax_map")),
        StyleContradictionScanTool(style_db, _schema_version(tool_settings, "style_contradiction_scan")),
    ]


def style_vector_for_text(text: str) -> Dict[str, float]:
    compact = normalized_cjk(text)
    total = max(count_cjk(text), 1)
    vector: Dict[str, float] = {}
    for dimension in STYLE_DIMENSIONS:
        terms = DIMENSION_TERMS[dimension]
        if dimension == "syntax_compression":
            punctuation_count = sum(text.count(mark) for mark in terms)
            vector[dimension] = round(min(1.0, (total / max(len(text), 1)) * 0.5 + punctuation_count * 0.1), 4)
            continue
        hits = sum(compact.count(term) if len(term) == 1 else text.count(term) for term in terms)
        vector[dimension] = round(min(1.0, hits / max(1.0, total / 4.0)), 4)
    return vector


def _default_profiles() -> List[StyleProfile]:
    return [
        StyleProfile(
            style_id="jiang_kui_qingkong",
            label="姜夔-清空",
            vector={
                "imagery_cold_light": 0.9,
                "imagery_water_travel": 0.55,
                "sound_distance": 0.7,
                "diction_grand": 0.1,
                "domestic_detail": 0.25,
                "emotional_directness": 0.15,
                "modern_register": 0.0,
                "syntax_compression": 0.75,
            },
            imagery=["淡月", "冷云", "疏星", "暗香", "疏影", "清砧", "短笛", "孤舟", "归舟"],
            syntax_patterns=["image-first compressed clause", "nominal imagery pairing", "turn by absence"],
            avoid_units=["开心", "漂亮", "故事", "城市", "照着", "非常", "特别", "想念"],
            replacements={
                "月光": ["淡月", "冷月", "疏星"],
                "月": ["淡月", "冷月"],
                "船": ["孤舟", "归舟"],
                "花": ["暗香", "疏梅"],
                "风": ["疏风", "冷风"],
                "声音": ["短笛", "清砧"],
            },
        ),
        StyleProfile(
            style_id="su_shi_haofang",
            label="苏轼-豪放",
            vector={
                "imagery_cold_light": 0.25,
                "imagery_water_travel": 0.55,
                "sound_distance": 0.2,
                "diction_grand": 0.9,
                "domestic_detail": 0.05,
                "emotional_directness": 0.35,
                "modern_register": 0.0,
                "syntax_compression": 0.55,
            },
            imagery=["大江", "长风", "明月", "千古", "归去", "沧海"],
            syntax_patterns=["expansive clause", "historical turn", "direct assertion"],
            avoid_units=["漂亮", "开心", "故事"],
            replacements={"江": ["大江", "长江"], "风": ["长风", "东风"], "月": ["明月", "素月"]},
        ),
        StyleProfile(
            style_id="li_qingzhao_wanyue",
            label="李清照-婉约",
            vector={
                "imagery_cold_light": 0.35,
                "imagery_water_travel": 0.2,
                "sound_distance": 0.25,
                "diction_grand": 0.05,
                "domestic_detail": 0.85,
                "emotional_directness": 0.65,
                "modern_register": 0.0,
                "syntax_compression": 0.65,
            },
            imagery=["帘", "黄花", "梧桐", "细雨", "雁", "残酒"],
            syntax_patterns=["domestic image turn", "soft emotional close"],
            avoid_units=["开心", "漂亮", "城市"],
            replacements={"花": ["黄花", "残红"], "雨": ["细雨", "晚雨"], "风": ["帘风", "晚风"]},
        ),
    ]


def _profiles_from_file(path: Path) -> List[StyleProfile]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_profiles = data.get("styles") if isinstance(data, Mapping) else None
    if not isinstance(raw_profiles, list):
        return []
    profiles = []
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            continue
        style_id = str(raw.get("style_id") or "")
        if not style_id:
            continue
        profiles.append(
            StyleProfile(
                style_id=style_id,
                label=str(raw.get("label") or style_id),
                vector=_float_vector(raw.get("vector") or {}),
                imagery=[str(item) for item in raw.get("imagery") or []],
                syntax_patterns=[str(item) for item in raw.get("syntax_patterns") or []],
                avoid_units=[str(item) for item in raw.get("avoid_units") or []],
                replacements={
                    str(key): [str(item) for item in value]
                    for key, value in dict(raw.get("replacements") or {}).items()
                    if isinstance(value, list)
                },
            )
        )
    return profiles


def _float_vector(raw: Mapping[str, Any]) -> Dict[str, float]:
    return {dimension: _clamp01(raw.get(dimension, 0.0)) for dimension in STYLE_DIMENSIONS}


def _rewrite_candidates(
    text: str,
    slot: str,
    profile: StyleProfile,
    constraints: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    raw_candidates = _candidate_texts(text, profile)
    char_count = _optional_int(constraints.get("char_count"))
    ban_units = [str(item) for item in constraints.get("ban_units") or []]
    candidates = []
    for candidate in raw_candidates:
        if char_count is not None and count_cjk(candidate) != char_count:
            continue
        if any(unit in candidate for unit in ban_units):
            continue
        score = _style_score(candidate, profile)
        candidates.append(
            {
                "text": candidate,
                "slot": slot,
                "style_score": score,
                "semantic_preservation": _semantic_preservation(text, candidate),
                "prosody_compatibility": _prosody_compatibility(candidate, constraints),
            }
        )
    return sorted(_dedupe_candidates(candidates), key=lambda item: item["style_score"], reverse=True)


def _candidate_texts(text: str, profile: StyleProfile) -> List[str]:
    if profile.style_id == "jiang_kui_qingkong":
        candidates = []
        if "月" in text and "江" in text:
            candidates.extend(["淡月涵江", "江月微明", "冷月横江"])
        if "船" in text or "舟" in text:
            candidates.extend(["淡月归舟", "孤舟自远", "江月孤舟"])
        if not candidates:
            candidates.extend(profile.imagery[:4])
        return candidates
    if profile.style_id == "su_shi_haofang":
        return ["大江东去", "长风万里", "明月千古", "沧海归舟"]
    if profile.style_id == "li_qingzhao_wanyue":
        return ["帘外晚风", "黄花细雨", "雁过西楼", "残酒醒迟"]
    return profile.imagery[:4]


def _imagery_results(source: str, profile: StyleProfile, semantic_role: Any) -> List[Dict[str, Any]]:
    replacements = []
    for key, values in profile.replacements.items():
        if key in source or source in key:
            replacements.extend(values)
    if not replacements:
        replacements = profile.imagery[:3]
    results = []
    for replacement in _unique(replacements):
        results.append(
            {
                "source_imagery": source,
                "target_imagery": replacement,
                "semantic_role": str(semantic_role or _semantic_role(replacement)),
                "style_score": _style_score(replacement, profile),
                "risk": "low" if _style_score(replacement, profile) >= 0.5 else "medium",
                "reason": "style-compatible imagery replacement",
            }
        )
    return sorted(results, key=lambda item: item["style_score"], reverse=True)


def _syntax_results(text: str, profile: StyleProfile) -> List[Dict[str, Any]]:
    semantics = _kept_semantics(text)
    if profile.style_id == "jiang_kui_qingkong":
        candidates = ["淡月涵江，孤舟自远", "江月微明，归舟一叶"]
    elif profile.style_id == "su_shi_haofang":
        candidates = ["大江横槊，长风万里", "明月千古，归心浩荡"]
    elif profile.style_id == "li_qingzhao_wanyue":
        candidates = ["帘外微寒，雁影初过", "黄花细雨，残酒醒迟"]
    else:
        candidates = profile.imagery[:2]
    patterns = profile.syntax_patterns or ["style-compatible clause"]
    return [
        {
            "syntax_pattern": patterns[index % len(patterns)],
            "rewritten_syntax": candidate,
            "kept_semantics": semantics,
            "style_score": _style_score(candidate, profile),
        }
        for index, candidate in enumerate(candidates)
    ]


def _contradictions(text: str, profile: StyleProfile) -> List[Dict[str, Any]]:
    contradictions = []
    for unit in profile.avoid_units:
        for span in find_literal_spans(text, unit):
            contradictions.append(
                {
                    "type": "register" if unit in DIMENSION_TERMS["modern_register"] else "style_unit",
                    "span": span.to_dict(),
                    "evidence": f"'{unit}' is disfavored for {profile.label}.",
                    "severity": "high" if unit in DIMENSION_TERMS["modern_register"] else "medium",
                    "suggested_operation": "style_register_convert"
                    if unit in DIMENSION_TERMS["modern_register"]
                    else "style_rewrite",
                }
            )
    return contradictions


def _style_score(text: str, profile: StyleProfile) -> float:
    vector_score = _cosine(style_vector_for_text(text), profile.vector)
    term_bonus = 0.0
    profile_terms = profile.imagery + list(profile.replacements.keys())
    if profile_terms:
        term_bonus = min(0.2, sum(1 for term in profile_terms if term in text) / len(profile_terms))
    avoid_penalty = 0.15 if any(unit in text for unit in profile.avoid_units) else 0.0
    return round(max(0.0, min(1.0, vector_score + term_bonus - avoid_penalty)), 4)


def _semantic_preservation(source: str, candidate: str) -> float:
    shared = set(normalized_cjk(source)) & set(normalized_cjk(candidate))
    base = len(shared) / max(len(set(normalized_cjk(source))), 1)
    if ("月" in source and "月" in candidate) or ("船" in source and "舟" in candidate):
        base += 0.25
    if ("江" in source and "江" in candidate) or ("水" in source and "江" in candidate):
        base += 0.15
    return round(max(0.0, min(1.0, base)), 4)


def _prosody_compatibility(candidate: str, constraints: Mapping[str, Any]) -> float:
    char_count = _optional_int(constraints.get("char_count"))
    score = 1.0
    if char_count is not None and count_cjk(candidate) != char_count:
        score -= 0.5
    if constraints.get("tone_pattern"):
        score -= 0.1
    if constraints.get("rhyme"):
        score -= 0.05
    return round(max(0.0, score), 4)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    numerator = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    right_norm = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return round(numerator / (left_norm * right_norm), 4)


def _target_style(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("target_style") or metadata.get("style_id") or "").strip()


def _top_dimensions(vector: Mapping[str, float]) -> List[str]:
    return [key for key, value in sorted(vector.items(), key=lambda item: item[1], reverse=True) if value > 0][:3]


def _weak_dimensions(vector: Mapping[str, float]) -> List[str]:
    return [key for key, value in sorted(vector.items(), key=lambda item: item[1]) if value <= 0.1][:3]


def _dedupe_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for candidate in candidates:
        text = candidate["text"]
        if text in seen:
            continue
        seen.add(text)
        deduped.append(candidate)
    return deduped


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _semantic_role(text: str) -> str:
    if any(term in text for term in ["月", "星", "霜", "雪"]):
        return "cold_light"
    if any(term in text for term in ["舟", "江", "水"]):
        return "water_travel"
    if any(term in text for term in ["笛", "砧", "钟"]):
        return "sound_distance"
    return "imagery"


def _kept_semantics(text: str) -> List[str]:
    semantics = []
    if "月" in text:
        semantics.append("moonlight")
    if "江" in text or "水" in text:
        semantics.append("river")
    if "船" in text or "舟" in text:
        semantics.append("boat")
    return semantics or ["source_semantics"]


def _severity_value(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(value, 0)


def _clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cache_key(schema_version: str, text: str, cipai: Optional[str], metadata: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": schema_version,
        "text": text,
        "cipai": cipai or "",
        "metadata": dict(metadata),
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{schema_version}:{digest}"


def _load_style_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Style tool config must be a JSON object")
    return dict(data)


def _schema_version(tool_settings: Mapping[str, Any], name: str) -> Optional[str]:
    settings = tool_settings.get(name)
    if isinstance(settings, Mapping) and settings.get("schema_version"):
        return str(settings["schema_version"])
    return None


__all__ = [
    "STYLE_TOOL_NAMES",
    "ImageryMapTool",
    "StyleContradictionScanTool",
    "StyleDatabase",
    "StyleRewriteTool",
    "StyleVectorizeTool",
    "SyntaxMapTool",
    "build_style_tools",
    "style_vector_for_text",
]
