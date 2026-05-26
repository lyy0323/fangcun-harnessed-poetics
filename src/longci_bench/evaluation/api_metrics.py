"""API-backed metric scorers for entity richness and fluency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .entities import EntityExtractionResult, EntityExtractor, HeuristicEntityExtractor
from .fluency import FluencyResult, FluencyScorer, HeuristicFluencyScorer


class MetricApiError(RuntimeError):
    pass


class JsonMetricCache:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self._data: Dict[str, Any] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self._data = dict(loaded)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        value = self._data.get(key)
        return dict(value) if isinstance(value, dict) else None

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        self._data[key] = dict(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


class ApiPoeticEntityExtractor(EntityExtractor):
    """Extract poetry-aware entities through a JSON-only API judge."""

    def __init__(
        self,
        client: Any,
        cache: Optional[JsonMetricCache] = None,
        metric_version: str = "entity.api_poetic.v1",
        fallback: Optional[EntityExtractor] = None,
    ):
        self.client = client
        self.cache = cache
        self.metric_version = metric_version
        self.fallback = fallback

    def extract_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        messages = _entity_messages(text, cipai=cipai, prompt=prompt)
        try:
            payload, api_metadata = _call_json_metric(
                self.client,
                messages,
                cache=self.cache,
                cache_key=_cache_key(self.metric_version, text, cipai, prompt),
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            result = self.fallback.extract_result(text, cipai=cipai, prompt=prompt, metadata=metadata)
            result.metadata = {**result.metadata, "api_error": str(exc), "api_fallback": True}
            return result

        items = _entity_items(payload)
        terms = _unique_ordered(str(item["text"]) for item in items if item.get("text"))
        return EntityExtractionResult(
            terms=terms,
            items=items,
            backend="api_poetic",
            metadata={**api_metadata, "metric_version": self.metric_version},
        )


class HybridEntityExtractor(EntityExtractor):
    """Union local heuristic candidates with poetry-aware API entities."""

    def __init__(self, extractors: Iterable[EntityExtractor], backend: str = "hybrid_api"):
        self.extractors = list(extractors)
        self.backend = backend

    def extract_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        terms: List[str] = []
        items: List[Dict[str, Any]] = []
        backends: List[str] = []
        for extractor in self.extractors:
            result = extractor.extract_result(text, cipai=cipai, prompt=prompt, metadata=metadata)
            backends.append(result.backend)
            terms.extend(result.terms)
            items.extend(result.items)
        return EntityExtractionResult(
            terms=_unique_ordered(terms),
            items=items,
            backend=self.backend,
            metadata={"component_backends": backends},
        )


class ApiFluencyJudge(FluencyScorer):
    """Score fluency through a JSON-only API judge."""

    def __init__(
        self,
        client: Any,
        cache: Optional[JsonMetricCache] = None,
        metric_version: str = "fluency.api_judge.v1",
        fallback: Optional[FluencyScorer] = None,
    ):
        self.client = client
        self.cache = cache
        self.metric_version = metric_version
        self.fallback = fallback

    def score_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FluencyResult:
        messages = _fluency_messages(text, cipai=cipai, prompt=prompt)
        try:
            payload, api_metadata = _call_json_metric(
                self.client,
                messages,
                cache=self.cache,
                cache_key=_cache_key(self.metric_version, text, cipai, prompt),
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            result = self.fallback.score_result(text, cipai=cipai, prompt=prompt, metadata=metadata)
            result.metadata = {**result.metadata, "api_error": str(exc), "api_fallback": True}
            return result

        dimensions = _float_mapping(payload.get("dimensions") if isinstance(payload, Mapping) else {})
        score = _optional_float(payload.get("score") if isinstance(payload, Mapping) else None)
        if score is None and dimensions:
            score = sum(dimensions.values()) / len(dimensions)
        if score is None:
            raise MetricApiError("Fluency API response must include score or numeric dimensions")
        issues = _issue_items(payload.get("issues") if isinstance(payload, Mapping) else [])
        return FluencyResult(
            score=_clamp01(score),
            dimensions=dimensions,
            issues=issues,
            backend="api_judge",
            metadata={**api_metadata, "metric_version": self.metric_version},
        )


def _call_json_metric(
    client: Any,
    messages: List[Dict[str, str]],
    cache: Optional[JsonMetricCache],
    cache_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached, {"cached": True, "cache_key": cache_key, "model": _client_model(client)}

    response = client.generate_from_messages(messages, metadata={"metric_cache_key": cache_key})
    text = str(getattr(response, "text", "") or "")
    payload = _parse_json_object(text)
    if cache:
        cache.set(cache_key, payload)
    return payload, {
        "cached": False,
        "cache_key": cache_key,
        "model": _client_model(client),
        "raw_response_chars": len(text),
    }


def _entity_messages(text: str, cipai: Optional[str], prompt: Optional[str]) -> List[Dict[str, str]]:
    system = (
        "你是中文古典诗词评估侧的实体与意象抽取器。"
        "只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。"
    )
    user = {
        "task": "extract_poetic_entities",
        "cipai": cipai or "",
        "prompt": prompt or "",
        "text": text,
        "requirements": [
            "抽取有内容承载的实体、意象、典故、地名、人名、时间词、物象和动作状态词。",
            "不要把语气词、虚词、标点或单纯格律停顿当作实体。",
            "同一词只保留一次；短词优先保留能独立表达意象或实体的形式。",
        ],
        "output_schema": {
            "entities": [
                {
                    "text": "string",
                    "category": "person|place|time|imagery|allusion|object|action_state|other",
                    "evidence": "string",
                }
            ]
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _fluency_messages(text: str, cipai: Optional[str], prompt: Optional[str]) -> List[Dict[str, str]]:
    system = (
        "你是中文古典诗词评估侧的通顺性裁判。"
        "只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。"
    )
    user = {
        "task": "score_poetic_fluency",
        "cipai": cipai or "",
        "prompt": prompt or "",
        "text": text,
        "score_definition": "0 表示完全不通，1 表示语义连贯、句间承接自然、古典词风稳定。",
        "dimensions": {
            "syntax": "字词搭配和句法是否顺畅",
            "semantic_coherence": "上下片或相邻句之间是否连贯",
            "classical_register": "是否符合古典诗词语体，避免现代口语硬插",
            "line_connection": "意象和动作推进是否自然",
        },
        "output_schema": {
            "score": "number in [0, 1]",
            "dimensions": {
                "syntax": "number in [0, 1]",
                "semantic_coherence": "number in [0, 1]",
                "classical_register": "number in [0, 1]",
                "line_connection": "number in [0, 1]",
            },
            "issues": [{"type": "string", "message": "string", "evidence": "string"}],
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise MetricApiError(f"Metric API response is not JSON: {text[:200]}")
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise MetricApiError("Metric API response must be a JSON object")
    return parsed


def _entity_items(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_items = payload.get("entities") or payload.get("entity_items") or payload.get("terms") or []
    if not isinstance(raw_items, list):
        return []
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        if isinstance(raw, str):
            text = raw.strip()
            category = "entity"
            evidence = ""
        elif isinstance(raw, Mapping):
            text = str(raw.get("text") or raw.get("term") or raw.get("name") or "").strip()
            category = str(raw.get("category") or raw.get("type") or "entity")
            evidence = str(raw.get("evidence") or raw.get("reason") or "")
        else:
            continue
        if not text:
            continue
        items.append({"text": text, "category": category, "evidence": evidence, "source": "api_poetic"})
    return items


def _issue_items(raw_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    issues: List[Dict[str, Any]] = []
    for raw in raw_items:
        if isinstance(raw, str):
            issues.append({"type": "fluency", "message": raw})
        elif isinstance(raw, Mapping):
            issues.append(
                {
                    "type": str(raw.get("type") or "fluency"),
                    "message": str(raw.get("message") or raw.get("reason") or ""),
                    "evidence": str(raw.get("evidence") or ""),
                }
            )
    return issues


def _float_mapping(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        parsed = _optional_float(raw)
        if parsed is not None:
            result[str(key)] = _clamp01(parsed)
    return result


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unique_ordered(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _cache_key(metric_version: str, text: str, cipai: Optional[str], prompt: Optional[str]) -> str:
    payload = {"metric_version": metric_version, "text": text, "cipai": cipai or "", "prompt": prompt or ""}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{metric_version}:{digest}"


def _client_model(client: Any) -> str:
    return str(getattr(client, "model", None) or getattr(client, "name", None) or "")


__all__ = [
    "ApiFluencyJudge",
    "ApiPoeticEntityExtractor",
    "HybridEntityExtractor",
    "JsonMetricCache",
    "MetricApiError",
]
