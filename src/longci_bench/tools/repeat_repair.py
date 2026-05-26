"""Repeat-location and repair tool.

The first implementation keeps the backend deterministic so the contract can be
tested locally. API-fitted language backends can later replace the semantic and
candidate-generation steps without changing the ToolRequest/ToolResponse shape.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from ..schemas import ToolRequest, ToolResponse
from ..text import CharInfo, cjk_chars, count_cjk, span_from_char_window
from .base import Tool
from .repetition import RepetitionTool


REPEAT_REPAIR_TOOL_NAME = "repeat_repair"


@dataclass(frozen=True)
class SentenceSegment:
    sentence_index: int
    start: int
    end: int
    text: str
    cjk_start_1based: int
    cjk_end_1based: int


class RepeatRepairTool(Tool):
    name = REPEAT_REPAIR_TOOL_NAME
    description = (
        "Locate repeated CJK substrings, bind each repair target to its sentence "
        "and available prosody rule window, then return candidate rewrites."
    )
    tool_group = "repair_tool"
    usage_stage = "repair"
    model_exposure = "extra"
    tags = ("repetition", "repair", "api_fitted_language_tool")

    def __init__(
        self,
        repetition_tool: Optional[Tool] = None,
        rule_tool: Optional[Tool] = None,
        judou_tool: Optional[Tool] = None,
        prosody_tool: Optional[Tool] = None,
        default_min_ngram: int = 2,
        default_max_ngram: int = 6,
        max_repetitions: int = 20,
        max_candidates_per_target: int = 3,
    ):
        self.repetition_tool = repetition_tool or RepetitionTool()
        self.rule_tool = rule_tool
        self.judou_tool = judou_tool
        self.prosody_tool = prosody_tool
        self.default_min_ngram = default_min_ngram
        self.default_max_ngram = default_max_ngram
        self.max_repetitions = max_repetitions
        self.max_candidates_per_target = max_candidates_per_target

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Ci body text to repair. Do not include title, cipai label, prompt, notes, or explanation.",
                },
                "cipai": {"type": "string", "description": "Cipai name used to fetch local rule windows."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "ngram_size": {
                            "type": "integer",
                            "description": "Exact repeated substring length to locate. Defaults to min/max search.",
                        },
                        "min_ngram_size": {"type": "integer", "description": "Minimum substring length, default 2."},
                        "max_ngram_size": {"type": "integer", "description": "Maximum substring length, default 6."},
                        "min_occurrences": {"type": "integer", "description": "Minimum occurrences, default 2."},
                        "forbid_single_char_repetition": {
                            "type": "boolean",
                            "description": "When true, search and final repetition validation include repeated single CJK characters.",
                        },
                        "final_check_min_ngram_size": {
                            "type": "integer",
                            "description": "Minimum repeated-unit length for validating candidate patched text.",
                        },
                        "final_check_max_ngram_size": {
                            "type": "integer",
                            "description": "Maximum repeated-unit length for validating candidate patched text.",
                        },
                        "target_occurrence": {
                            "type": "string",
                            "description": "Which occurrences to repair: later_occurrences or all_occurrences.",
                        },
                        "validation_only": {
                            "type": "boolean",
                            "description": "When true, passed is false whenever repetitions are found, while repair suggestions are still returned.",
                        },
                        "fail_on_repetition": {
                            "type": "boolean",
                            "description": "Alias of validation_only for final validation toolsets.",
                        },
                        "auto_repetition_by_form_length": {
                            "type": "boolean",
                            "description": "When true, use single-character repair/checking for short forms and 2+ character repair/checking for long forms.",
                        },
                        "single_char_repetition_max_chars": {
                            "type": "integer",
                            "description": "Maximum form length that should use single-character repetition repair/checking; default 60.",
                        },
                    },
                },
            },
            "required": ["text"],
        }

    def output_schema(self) -> Dict[str, Any]:
        schema = super().output_schema()
        schema["properties"]["metadata"]["description"] = (
            "Includes repetitions, repair_targets, candidates, best_candidate, rule, and backend status."
        )
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        chars = cjk_chars(request.text)
        rule_summary, backend_issues = self._load_rule_summary(request)
        form_char_count = _form_char_count(rule_summary, len(chars))
        min_ngram, max_ngram = self._ngram_range(metadata, form_char_count)
        final_min_ngram, final_max_ngram = self._final_repetition_range(
            metadata,
            min_ngram,
            max_ngram,
            form_char_count,
        )
        min_occurrences = _positive_int(metadata.get("min_occurrences"), 2)
        segments = self._sentence_segments(request.text, chars, rule_summary)

        groups = self._locate_repetitions(chars, min_ngram, max_ngram, min_occurrences)
        repetitions = [
            self._build_repetition(group_index, substring, starts, chars, segments, len(substring))
            for group_index, (substring, starts) in enumerate(groups, start=1)
        ]
        repair_targets = self._build_repair_targets(repetitions, rule_summary, metadata)
        candidates = self._build_candidates(request, repair_targets, final_min_ngram, final_max_ngram)
        validation_only = _bool(metadata.get("validation_only"), _bool(metadata.get("fail_on_repetition"), False))

        issues = list(backend_issues)
        for repetition in repetitions:
            issues.append(
                {
                    "type": "repeat_repair_target",
                    "message": "Repeated substring located for repair.",
                    "substring": repetition["substring"],
                    "ngram_size": repetition["ngram_size"],
                    "occurrence_count": repetition["occurrence_count"],
                    "spans": [occurrence["span"] for occurrence in repetition["occurrences"]],
                }
            )
        if repetitions and not candidates:
            issues.append(
                {
                    "type": "no_repair_candidate",
                    "message": "Repeated substrings were found, but no candidate rewrite satisfied the local constraints.",
                }
            )

        best_candidate = candidates[0] if candidates else None
        metrics = {
            "total_cjk_chars": len(chars),
            "repetition_count": len(repetitions),
            "repair_target_count": len(repair_targets),
            "candidate_count": len(candidates),
            "validated_candidate_count": sum(1 for candidate in candidates if candidate["validation"]["passed"]),
            "content_repetition_rate": self._content_repetition_rate(repetitions, len(chars)),
            "final_repetition_min_ngram_size": final_min_ngram,
            "final_repetition_max_ngram_size": final_max_ngram,
        }
        return ToolResponse(
            self.name,
            passed=not repetitions if validation_only else (not repetitions or bool(candidates)),
            issues=issues,
            metrics=metrics,
            metadata={
                "operation": self.name,
                "schema_version": "0.1",
                "backend": "local_deterministic",
                "cache_key": _cache_key(request.text, request.cipai, metadata),
                "input": {
                    "ngram_size": metadata.get("ngram_size"),
                    "min_ngram_size": min_ngram,
                    "max_ngram_size": max_ngram,
                    "min_occurrences": min_occurrences,
                    "forbid_single_char_repetition": _bool(metadata.get("forbid_single_char_repetition"), False),
                    "auto_repetition_by_form_length": _bool(metadata.get("auto_repetition_by_form_length"), False),
                    "single_char_repetition_max_chars": _positive_int(
                        metadata.get("single_char_repetition_max_chars"),
                        60,
                    ),
                    "form_char_count": form_char_count,
                    "final_check_min_ngram_size": final_min_ngram,
                    "final_check_max_ngram_size": final_max_ngram,
                    "validation_only": validation_only,
                    "target_occurrence": metadata.get("target_occurrence") or "later_occurrences",
                },
                "rule": rule_summary,
                "repetitions": repetitions,
                "repair_targets": repair_targets,
                "candidates": candidates,
                "best_candidate": best_candidate,
                "results": candidates,
            },
        )

    def _ngram_range(self, metadata: Mapping[str, Any], form_char_count: int) -> Tuple[int, int]:
        exact = metadata.get("ngram_size")
        if exact is not None:
            ngram_size = _positive_int(exact, self.default_min_ngram)
            return ngram_size, ngram_size
        default_min = (
            1
            if _bool(metadata.get("forbid_single_char_repetition"), False)
            or _single_char_for_form(metadata, form_char_count)
            else self.default_min_ngram
        )
        min_ngram = _positive_int(metadata.get("min_ngram_size"), default_min)
        max_ngram = _positive_int(metadata.get("max_ngram_size"), self.default_max_ngram)
        if max_ngram < min_ngram:
            max_ngram = min_ngram
        return min_ngram, max_ngram

    def _final_repetition_range(
        self,
        metadata: Mapping[str, Any],
        search_min_ngram: int,
        search_max_ngram: int,
        form_char_count: int,
    ) -> Tuple[int, int]:
        default_min = (
            1
            if search_min_ngram == 1
            or _bool(metadata.get("forbid_single_char_repetition"), False)
            or _single_char_for_form(metadata, form_char_count)
            else 2
        )
        final_min = _positive_int(metadata.get("final_check_min_ngram_size"), default_min)
        final_max = _positive_int(
            metadata.get("final_check_max_ngram_size"),
            max(search_max_ngram, self.default_max_ngram, final_min),
        )
        if final_max < final_min:
            final_max = final_min
        return final_min, final_max

    def _load_rule_summary(self, request: ToolRequest) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.rule_tool:
            return None, []
        cipai = request.cipai or request.metadata.get("cipai")
        if not cipai:
            return None, [
                {
                    "type": "missing_cipai_for_rule_window",
                    "message": "No cipai was supplied, so repeat repair could not fetch a prosody rule window.",
                }
            ]
        try:
            response = self.rule_tool.run(ToolRequest(self.rule_tool.name, str(cipai), cipai=str(cipai)))
        except Exception as exc:  # pragma: no cover - defensive boundary around optional backend
            return None, [{"type": "rule_backend_error", "message": str(exc)}]
        if not response.passed:
            return None, response.issues or [{"type": "rule_lookup_failed", "message": "Rule lookup failed."}]
        rule = response.metadata.get("rule")
        return (dict(rule) if isinstance(rule, Mapping) else None), []

    def _locate_repetitions(
        self,
        chars: Sequence[CharInfo],
        min_ngram: int,
        max_ngram: int,
        min_occurrences: int,
    ) -> List[Tuple[str, List[int]]]:
        groups: List[Tuple[str, List[int]]] = []
        total = len(chars)
        for ngram_size in range(max_ngram, min_ngram - 1, -1):
            seen: DefaultDict[str, List[int]] = defaultdict(list)
            for start in range(0, max(total - ngram_size + 1, 0)):
                key = "".join(info.char for info in chars[start : start + ngram_size])
                seen[key].append(start)
            repeated = [(key, starts) for key, starts in seen.items() if len(starts) >= min_occurrences]
            repeated.sort(key=lambda item: (len(item[1]), len(item[0]), item[0]), reverse=True)
            groups.extend(repeated)
            if len(groups) >= self.max_repetitions:
                break
        return groups[: self.max_repetitions]

    def _build_repetition(
        self,
        group_index: int,
        substring: str,
        starts: Sequence[int],
        chars: Sequence[CharInfo],
        segments: Sequence[SentenceSegment],
        ngram_size: int,
    ) -> Dict[str, Any]:
        occurrences = []
        for occurrence_index, start in enumerate(starts, start=1):
            span = span_from_char_window(list(chars), start, ngram_size).to_dict()
            char_start = start + 1
            segment = _segment_for_char(segments, char_start)
            semantic = _semantic_label(substring, segment.text if segment else "")
            occurrences.append(
                {
                    "occurrence_id": f"r{group_index}_o{occurrence_index}",
                    "occurrence_index": occurrence_index,
                    "char_start_1based": char_start,
                    "char_end_1based": start + ngram_size,
                    "span": span,
                    "sentence_index": segment.sentence_index if segment else None,
                    "sentence_text": segment.text if segment else "",
                    "meaning": semantic["meaning"],
                    "semantic_role": semantic["role"],
                }
            )
        return {
            "repetition_id": f"r{group_index}",
            "substring": substring,
            "ngram_size": ngram_size,
            "occurrence_count": len(occurrences),
            "occurrences": occurrences,
        }

    def _sentence_segments(
        self,
        text: str,
        chars: Sequence[CharInfo],
        rule_summary: Optional[Mapping[str, Any]],
    ) -> List[SentenceSegment]:
        counts = list(rule_summary.get("sentence_char_counts") or []) if rule_summary else []
        if counts and sum(_positive_int(item, 0) for item in counts) <= len(chars):
            return _segments_from_rule_counts(text, chars, counts)
        return _segments_from_punctuation(text, chars)

    def _build_repair_targets(
        self,
        repetitions: Sequence[Mapping[str, Any]],
        rule_summary: Optional[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        repair_all = metadata.get("target_occurrence") == "all_occurrences"
        targets: List[Dict[str, Any]] = []
        for repetition in repetitions:
            occurrences = repetition["occurrences"] if repair_all else repetition["occurrences"][1:]
            for occurrence in occurrences:
                target = {
                    "target_id": f"{repetition['repetition_id']}:{occurrence['occurrence_id']}",
                    "repetition_id": repetition["repetition_id"],
                    "substring": repetition["substring"],
                    "ngram_size": repetition["ngram_size"],
                    "occurrence": occurrence,
                    "repair_window": _repair_window(occurrence, repetition["ngram_size"], rule_summary, repetition["substring"]),
                }
                targets.append(target)
        return targets

    def _build_candidates(
        self,
        request: ToolRequest,
        targets: Sequence[Mapping[str, Any]],
        final_min_ngram: int,
        final_max_ngram: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for target in targets:
            substring = str(target["substring"])
            occurrence = target["occurrence"]
            span = occurrence["span"]
            replacements = _replacement_pool(substring, occurrence.get("semantic_role", ""))
            filtered = [
                replacement
                for replacement in replacements
                if len(replacement) == target["ngram_size"] and replacement != substring
            ][: self.max_candidates_per_target]
            for replacement_index, replacement in enumerate(filtered, start=1):
                patched_text = request.text[: span["start"]] + replacement + request.text[span["end"] :]
                validation = self._validate_candidate(request, patched_text, final_min_ngram, final_max_ngram)
                candidate = {
                    "candidate_id": f"{target['target_id']}:c{replacement_index}",
                    "target_id": target["target_id"],
                    "target_occurrence_id": occurrence["occurrence_id"],
                    "source": substring,
                    "replacement": replacement,
                    "patched_text": patched_text,
                    "slot": {
                        "span": span,
                        "sentence_index": occurrence.get("sentence_index"),
                        "sentence_text": occurrence.get("sentence_text"),
                    },
                    "style_score": 0.5,
                    "semantic_preservation": _semantic_preservation_score(substring, replacement),
                    "prosody_compatibility": _prosody_compatibility(validation),
                    "repetition_reduction": validation["repetition_reduction"],
                    "validation": validation,
                    "repair_window": target["repair_window"],
                }
                candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                item["validation"]["passed"],
                item["repetition_reduction"],
                item["semantic_preservation"],
                item["prosody_compatibility"],
            ),
            reverse=True,
        )
        return candidates

    def _validate_candidate(
        self,
        request: ToolRequest,
        patched_text: str,
        final_min_ngram: int,
        final_max_ngram: int,
    ) -> Dict[str, Any]:
        repetition_tool = RepetitionTool(min_ngram=final_min_ngram, max_ngram=final_max_ngram)
        repetition_response = repetition_tool.run(ToolRequest("repetition", patched_text, cipai=request.cipai))
        repetition_passed = repetition_response.passed
        judou_passed = self._optional_validation(self.judou_tool, request, patched_text)
        prosody_passed = self._optional_validation(self.prosody_tool, request, patched_text)
        passed = repetition_passed and judou_passed is not False and prosody_passed is not False
        before_rate = repetition_tool.run(ToolRequest("repetition", request.text, cipai=request.cipai)).metrics.get(
            "content_repetition_rate", 0.0
        )
        after_rate = repetition_response.metrics.get("content_repetition_rate", 0.0)
        return {
            "passed": passed,
            "repetition_passed": repetition_passed,
            "final_repetition_min_ngram_size": final_min_ngram,
            "final_repetition_max_ngram_size": final_max_ngram,
            "judou_passed": judou_passed,
            "prosody_passed": prosody_passed,
            "repetition_issue_count": repetition_response.metrics.get("repeated_substring_count", 0),
            "repetition_rate_before": before_rate,
            "repetition_rate_after": after_rate,
            "repetition_reduction": max(0.0, float(before_rate or 0.0) - float(after_rate or 0.0)),
        }

    def _optional_validation(self, tool: Optional[Tool], request: ToolRequest, text: str) -> Optional[bool]:
        if not tool or not request.cipai:
            return None
        try:
            return tool.run(ToolRequest(tool.name, text, cipai=request.cipai)).passed
        except Exception:  # pragma: no cover - validation tools are optional adapters
            return None

    def _content_repetition_rate(self, repetitions: Sequence[Mapping[str, Any]], total_chars: int) -> float:
        if not total_chars:
            return 0.0
        repeated_units = 0
        for repetition in repetitions:
            repeated_units += int(repetition["ngram_size"]) * max(0, int(repetition["occurrence_count"]) - 1)
        return repeated_units / total_chars


def _segments_from_rule_counts(text: str, chars: Sequence[CharInfo], counts: Sequence[Any]) -> List[SentenceSegment]:
    segments: List[SentenceSegment] = []
    start = 0
    for index, raw_count in enumerate(counts, start=1):
        count = _positive_int(raw_count, 0)
        end = min(start + count, len(chars))
        if end <= start:
            continue
        first = chars[start]
        last = chars[end - 1]
        segments.append(
            SentenceSegment(
                sentence_index=index,
                start=first.index,
                end=last.index + 1,
                text=text[first.index : last.index + 1],
                cjk_start_1based=start + 1,
                cjk_end_1based=end,
            )
        )
        start = end
    return segments


def _segments_from_punctuation(text: str, chars: Sequence[CharInfo]) -> List[SentenceSegment]:
    if not chars:
        return []
    break_chars = set("，。！？；、,.!?;\n")
    ranges: List[Tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char in break_chars:
            ranges.append((start, index))
            start = index + 1
    if start < len(text):
        ranges.append((start, len(text)))

    segments: List[SentenceSegment] = []
    consumed = 0
    for start_index, end_index in ranges:
        seg_chars = [info for info in chars if start_index <= info.index < end_index]
        if not seg_chars:
            continue
        first = seg_chars[0]
        last = seg_chars[-1]
        consumed += len(seg_chars)
        segments.append(
            SentenceSegment(
                sentence_index=len(segments) + 1,
                start=first.index,
                end=last.index + 1,
                text=text[first.index : last.index + 1],
                cjk_start_1based=consumed - len(seg_chars) + 1,
                cjk_end_1based=consumed,
            )
        )
    return segments


def _segment_for_char(segments: Sequence[SentenceSegment], char_start_1based: int) -> Optional[SentenceSegment]:
    for segment in segments:
        if segment.cjk_start_1based <= char_start_1based <= segment.cjk_end_1based:
            return segment
    return segments[-1] if segments else None


def _repair_window(
    occurrence: Mapping[str, Any],
    ngram_size: int,
    rule_summary: Optional[Mapping[str, Any]],
    repeated_unit: str,
) -> Dict[str, Any]:
    char_start = int(occurrence["char_start_1based"])
    char_end = int(occurrence["char_end_1based"])
    sentence_counts = list(rule_summary.get("sentence_char_counts") or []) if rule_summary else []
    sentence_index = occurrence.get("sentence_index")
    sentence_start = None
    sentence_end = None
    if sentence_counts and isinstance(sentence_index, int) and 1 <= sentence_index <= len(sentence_counts):
        sentence_start = sum(int(item) for item in sentence_counts[: sentence_index - 1]) + 1
        sentence_end = sentence_start + int(sentence_counts[sentence_index - 1]) - 1

    tone_pattern = list(rule_summary.get("tone_pattern") or []) if rule_summary else []
    rhyme_positions = set(rule_summary.get("rhyme_positions_1based") or []) if rule_summary else set()
    return {
        "span": occurrence["span"],
        "char_range_1based": [char_start, char_end],
        "char_count": ngram_size,
        "sentence_index": sentence_index,
        "sentence_text": occurrence.get("sentence_text", ""),
        "sentence_char_range_1based": [sentence_start, sentence_end] if sentence_start and sentence_end else None,
        "tone_pattern": tone_pattern[char_start - 1 : char_end] if tone_pattern else [],
        "rhyme_required": any(position in rhyme_positions for position in range(char_start, char_end + 1)),
        "forbidden_units": [repeated_unit],
    }


def _semantic_label(substring: str, sentence: str) -> Dict[str, str]:
    if any(char in substring for char in "风月花雪云烟霞"):
        return {
            "role": "scene_imagery",
            "meaning": "自然景物意象，通常承担起景、转景或情绪映照功能。",
        }
    if any(char in substring for char in "江水潮波舟帆"):
        return {
            "role": "water_travel_imagery",
            "meaning": "水路或行旅意象，通常承担空间移动、离别或望远功能。",
        }
    if sentence:
        return {"role": "local_expression", "meaning": "局部表达复现，需要结合所在句意替换。"}
    return {"role": "unknown", "meaning": "未能从本地启发式判断表达含义。"}


def _replacement_pool(substring: str, semantic_role: str) -> List[str]:
    exact = {
        "风月": ["烟水", "云影", "江月", "清霜", "花雨"],
        "明月": ["清影", "寒辉", "疏星", "秋声"],
        "江水": ["烟波", "潮声", "浦云", "沙痕"],
        "花影": ["香雾", "疏枝", "红雨", "春痕"],
        "风": ["云", "烟", "雨", "花", "霜", "灯"],
        "月": ["雨", "霜", "灯", "云", "烟", "花"],
        "花": ["月", "云", "烟", "柳", "灯", "霜"],
        "雨": ["月", "云", "烟", "霜", "灯", "花"],
        "云": ["风", "月", "烟", "霜", "灯", "花"],
        "烟": ["云", "水", "月", "霜", "灯", "花"],
    }
    if substring in exact:
        return exact[substring]
    if semantic_role == "water_travel_imagery":
        return ["烟波", "潮声", "浦云", "沙痕", "兰舟"]
    if semantic_role == "scene_imagery":
        return ["烟水", "云影", "清霜", "疏星", "花雨"]
    length = len(substring)
    generic = ["幽思", "新声", "余情", "清景", "微吟"]
    return [item for item in generic if len(item) == length]


def _semantic_preservation_score(source: str, replacement: str) -> float:
    shared = len(set(source).intersection(replacement))
    if shared:
        return 0.65 + min(0.2, shared * 0.05)
    return 0.6


def _prosody_compatibility(validation: Mapping[str, Any]) -> float:
    score = 0.7
    if validation.get("judou_passed") is False:
        score -= 0.2
    if validation.get("prosody_passed") is False:
        score -= 0.2
    if validation.get("repetition_passed"):
        score += 0.1
    return max(0.0, min(1.0, score))


def _form_char_count(rule_summary: Optional[Mapping[str, Any]], text_char_count: int) -> int:
    if rule_summary:
        parsed = _positive_int(rule_summary.get("char_count"), 0)
        if parsed:
            return parsed
    return text_char_count


def _single_char_for_form(metadata: Mapping[str, Any], form_char_count: int) -> bool:
    if not _bool(metadata.get("auto_repetition_by_form_length"), False):
        return False
    threshold = _positive_int(metadata.get("single_char_repetition_max_chars"), 60)
    return form_char_count <= threshold


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _cache_key(text: str, cipai: Optional[str], metadata: Mapping[str, Any]) -> str:
    size = metadata.get("ngram_size") or f"{metadata.get('min_ngram_size', 2)}-{metadata.get('max_ngram_size', 6)}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{cipai or ''}:{size}:{count_cjk(text)}:{digest}"
