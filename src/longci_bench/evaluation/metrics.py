"""Evaluation metrics for generated long ci records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..schemas import GenerationRecord, ToolRequest
from ..text import count_cjk, iter_nonempty_lines
from ..tools import ToolRegistry
from .entities import EntityExtractor, HeuristicEntityExtractor
from .fluency import FluencyScorer, HeuristicFluencyScorer

DEFAULT_FLUENCY_PASS_THRESHOLD = 0.6


def evaluate_records(
    records: Iterable[GenerationRecord],
    registry: ToolRegistry,
    entity_extractor: Optional[EntityExtractor] = None,
    fluency_scorer: Optional[FluencyScorer] = None,
    fluency_pass_threshold: float = DEFAULT_FLUENCY_PASS_THRESHOLD,
) -> Dict:
    extractor = entity_extractor or HeuristicEntityExtractor()
    scorer = fluency_scorer or HeuristicFluencyScorer()
    rows: List[Dict] = []

    for record in records:
        total_chars = count_cjk(record.text)
        line_count = len(list(iter_nonempty_lines(record.text)))
        judou = _run_optional_tool(registry, "check_judou", record.text, record.cipai)
        prosody_tool_name = "check_prosody" if "check_prosody" in registry.names() else "prosody"
        prosody = registry.run(ToolRequest(prosody_tool_name, record.text, cipai=record.cipai))
        repetition = registry.run(ToolRequest("repetition", record.text, cipai=record.cipai))
        imagery = registry.run(ToolRequest("imagery", record.text, cipai=record.cipai))
        entity_result = extractor.extract_result(
            record.text,
            cipai=record.cipai,
            prompt=record.prompt,
            metadata=record.metadata,
        )
        entities = entity_result.terms
        entity_richness = len(entities) / max(total_chars, 1)
        fluency_result = scorer.score_result(
            record.text,
            cipai=record.cipai,
            prompt=record.prompt,
            metadata=record.metadata,
        )
        fluency = fluency_result.score
        fluency_passed = fluency >= fluency_pass_threshold
        prosody_violation_count = _numeric_metric(prosody.metrics, "violation_count", len(prosody.issues))
        prosody_checked_chars = _prosody_denominator(prosody.metrics, total_chars)
        out_of_prosody_ratio = prosody_violation_count / max(prosody_checked_chars, 1)
        final_validation_passed = bool(prosody.passed and (judou.passed if judou else True))
        post_check_passed = bool(repetition.passed and imagery.passed)
        tool_call_metrics = _tool_call_metrics(record.metadata)

        rows.append(
            {
                "id": record.id,
                "split": record.split,
                "cipai": record.cipai,
                "model": record.model,
                "text_char_count": total_chars,
                "line_count": line_count,
                "judou_passed": judou.passed if judou else None,
                "judou_violation_count": _numeric_metric(judou.metrics, "violation_count", len(judou.issues)) if judou else None,
                "prosody_passed": prosody.passed,
                "prosody_tool_name": prosody_tool_name,
                "prosody_violation_count": prosody_violation_count,
                "prosody_checked_chars": prosody_checked_chars,
                "out_of_prosody_ratio": out_of_prosody_ratio,
                "final_validation_passed": final_validation_passed,
                "content_repetition_rate": repetition.metrics.get("content_repetition_rate", 0.0),
                "repeated_substring_count": repetition.metrics.get("repeated_substring_count", len(repetition.issues)),
                "repeated_units": repetition.metrics.get("repeated_units", 0),
                "repeated_imagery_category_count": imagery.metrics.get("repeated_imagery_category_count", 0),
                "post_check_passed": post_check_passed,
                "entity_count": len(entities),
                "entity_richness": entity_richness,
                "entity_terms": entities,
                "entity_items": entity_result.items,
                "entity_backend": entity_result.backend,
                "entity_metadata": entity_result.metadata,
                "fluency": fluency,
                "fluency_passed": fluency_passed,
                "fluency_backend": fluency_result.backend,
                "fluency_dimensions": fluency_result.dimensions,
                "fluency_issues": fluency_result.issues,
                "fluency_metadata": fluency_result.metadata,
                "tool_calls": tool_call_metrics,
                "tools": {
                    **({"check_judou": judou.to_dict()} if judou else {}),
                    "prosody": prosody.to_dict(),
                    "repetition": repetition.to_dict(),
                    "imagery": imagery.to_dict(),
                },
            }
        )

    total = len(rows)
    if total == 0:
        return {"total": 0, "summary": {}, "records": []}

    prosody_passes = sum(1 for row in rows if row["prosody_passed"])
    judou_rows = [row for row in rows if row["judou_passed"] is not None]
    tool_call_count_totals = Counter()
    for row in rows:
        tool_call_count_totals.update(row["tool_calls"]["per_tool_counts"])
    summary = {
        "total": total,
        "prosody_pass_rate": prosody_passes / total,
        "judou_pass_rate": _pass_rate(row["judou_passed"] for row in judou_rows),
        "final_validation_pass_rate": _pass_rate(row["final_validation_passed"] for row in rows),
        "out_of_prosody_ratio_average": _average(row["out_of_prosody_ratio"] for row in rows),
        "content_repetition_rate_average": _average(row["content_repetition_rate"] for row in rows),
        "repeated_substring_count_average": _average(row["repeated_substring_count"] for row in rows),
        "post_check_pass_rate": _pass_rate(row["post_check_passed"] for row in rows),
        "entity_richness_average": _average(row["entity_richness"] for row in rows),
        "entity_count_average": _average(row["entity_count"] for row in rows),
        "fluency_average": _average(row["fluency"] for row in rows),
        "fluency_pass_rate": _pass_rate(row["fluency_passed"] for row in rows),
        "prosody_violation_average": _average(row["prosody_violation_count"] for row in rows),
        "repeated_imagery_category_average": _average(
            row["repeated_imagery_category_count"] for row in rows
        ),
        "tool_call_count_average": _average(row["tool_calls"]["tool_call_count"] for row in rows),
        "tool_round_count_average": _average(row["tool_calls"]["tool_round_count"] for row in rows),
        "unique_tool_count_average": _average(row["tool_calls"]["unique_tool_count"] for row in rows),
        "tool_call_failure_count_average": _average(row["tool_calls"]["failed_tool_response_count"] for row in rows),
        "final_validation_attempt_count_average": _average(
            row["tool_calls"]["final_validation_attempt_count"] for row in rows
        ),
        "tool_call_counts": dict(sorted(tool_call_count_totals.items())),
    }
    return {"total": total, "summary": summary, "records": rows}


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _pass_rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    items = [item for item in values if item is not None]
    if not items:
        return None
    return sum(1 for item in items if item) / len(items)


def _numeric_metric(metrics: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _prosody_denominator(metrics: Mapping[str, Any], fallback_chars: int) -> float:
    for key in ("checked_chars", "expected_chars", "total_cjk_chars"):
        value = _numeric_metric(metrics, key, 0.0)
        if value > 0:
            return value
    return float(max(fallback_chars, 1))


def _run_optional_tool(registry: ToolRegistry, tool_name: str, text: str, cipai: str):
    if tool_name not in registry.names():
        return None
    return registry.run(ToolRequest(tool_name, text, cipai=cipai))


def _tool_call_metrics(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    trace = _find_tool_trace(metadata)
    per_tool_counts: Counter = Counter()
    failed_tool_response_count = 0
    tool_round_count = 0
    final_validation_attempt_count = 0
    final_validation_passed: Optional[bool] = None

    for round_item in trace:
        observations = round_item.get("observations") if isinstance(round_item, Mapping) else None
        if observations:
            tool_round_count += 1
            for observation in observations:
                call = observation.get("call", {}) if isinstance(observation, Mapping) else {}
                response = observation.get("response", {}) if isinstance(observation, Mapping) else {}
                name = _call_name(call)
                if name:
                    per_tool_counts[name] += 1
                if isinstance(response, Mapping) and response.get("passed") is False:
                    failed_tool_response_count += 1

        final_validation = round_item.get("final_validation") if isinstance(round_item, Mapping) else None
        if final_validation:
            attempts, passed = _final_validation_metrics(final_validation)
            final_validation_attempt_count += attempts
            final_validation_passed = passed

    tool_call_count = sum(per_tool_counts.values())
    return {
        "tool_round_count": tool_round_count,
        "tool_call_count": tool_call_count,
        "unique_tool_count": len(per_tool_counts),
        "per_tool_counts": dict(sorted(per_tool_counts.items())),
        "failed_tool_response_count": failed_tool_response_count,
        "final_validation_attempt_count": final_validation_attempt_count,
        "final_validation_passed": final_validation_passed,
    }


def _find_tool_trace(metadata: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    candidates = [
        metadata.get("tool_trace") if isinstance(metadata, Mapping) else None,
        _nested_get(metadata, ("response", "metadata", "tool_trace")),
        _nested_get(metadata, ("metadata", "tool_trace")),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, Mapping)]
    return []


def _nested_get(value: Any, path: Iterable[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _call_name(call: Any) -> str:
    if not isinstance(call, Mapping):
        return ""
    return str(call.get("name") or call.get("tool_name") or "")


def _final_validation_metrics(final_validation: Mapping[str, Any]) -> tuple:
    if "validations" in final_validation and isinstance(final_validation["validations"], list):
        responses = [
            item.get("response", {})
            for item in final_validation["validations"]
            if isinstance(item, Mapping)
        ]
        if not responses:
            return 0, None
        passed = all(response.get("passed") is True for response in responses if isinstance(response, Mapping))
        return len(responses), passed
    response = final_validation.get("response")
    if isinstance(response, Mapping):
        return 1, bool(response.get("passed"))
    if "passed" in final_validation:
        return 1, bool(final_validation.get("passed"))
    return 0, None
