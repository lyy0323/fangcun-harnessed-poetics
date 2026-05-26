"""Configuration helpers for optional API-backed evaluation metrics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from ..models import OpenAICompatibleClient
from .api_metrics import ApiFluencyJudge, ApiPoeticEntityExtractor, HybridEntityExtractor, JsonMetricCache
from .entities import EntityExtractor, HeuristicEntityExtractor
from .fluency import FluencyScorer, HeuristicFluencyScorer
from .literary_quality import ApiLiteraryQualityJudge
from .metrics import DEFAULT_FLUENCY_PASS_THRESHOLD


@dataclass
class MetricComponents:
    entity_extractor: Optional[EntityExtractor] = None
    fluency_scorer: Optional[FluencyScorer] = None
    fluency_pass_threshold: float = DEFAULT_FLUENCY_PASS_THRESHOLD


def build_metric_components(
    metric_config_path: Optional[str] = None,
    metric_model_config_path: Optional[str] = None,
) -> MetricComponents:
    if not metric_config_path and not metric_model_config_path:
        return MetricComponents()

    config_dir = Path.cwd()
    config: Dict[str, Any] = {}
    if metric_config_path:
        path = Path(metric_config_path).expanduser()
        config_dir = path.resolve().parent
        config = _load_json_object(path, "--metric-config")
        if _looks_like_model_config(config):
            config = {
                "api": {"model_config": str(path)},
                "entity": {"backend": "hybrid_api"},
                "fluency": {"backend": "api_judge"},
            }

    if metric_model_config_path:
        api_config = dict(config.get("api") or {})
        api_config["model_config"] = metric_model_config_path
        api_config.setdefault("temperature", 0)
        api_config.setdefault("max_tokens", 1200)
        config["api"] = api_config
        config.setdefault("entity", {"backend": "hybrid_api"})
        config.setdefault("fluency", {"backend": "api_judge"})

    cache = _cache_from_config(config, config_dir)
    api_client = None
    if _requires_api(config):
        api_client = _openai_client_from_config(dict(config.get("api") or {}), config_dir)

    entity_config = dict(config.get("entity") or {})
    fluency_config = dict(config.get("fluency") or {})
    entity_extractor = _build_entity_extractor(entity_config, api_client, cache)
    fluency_scorer = _build_fluency_scorer(fluency_config, api_client, cache)
    return MetricComponents(
        entity_extractor=entity_extractor,
        fluency_scorer=fluency_scorer,
        fluency_pass_threshold=_float_or_default(
            fluency_config.get("threshold"),
            config.get("fluency_pass_threshold"),
            DEFAULT_FLUENCY_PASS_THRESHOLD,
        ),
    )


def _build_entity_extractor(
    config: Dict[str, Any],
    api_client: Any,
    cache: Optional[JsonMetricCache],
) -> Optional[EntityExtractor]:
    backend = str(config.get("backend") or "heuristic")
    if backend == "heuristic":
        return HeuristicEntityExtractor(config.get("terms"))
    if backend not in {"api_poetic", "hybrid_api"}:
        raise ValueError(f"Unsupported entity backend: {backend}")
    if api_client is None:
        raise ValueError("Entity API backend requires api.model_config or inline api settings")
    fallback = HeuristicEntityExtractor(config.get("terms")) if _bool(config.get("fallback_on_error"), False) else None
    api_extractor = ApiPoeticEntityExtractor(
        api_client,
        cache=cache,
        metric_version=str(config.get("metric_version") or "entity.api_poetic.v1"),
        fallback=fallback,
    )
    if backend == "api_poetic":
        return api_extractor
    return HybridEntityExtractor([HeuristicEntityExtractor(config.get("terms")), api_extractor])


def _build_fluency_scorer(
    config: Dict[str, Any],
    api_client: Any,
    cache: Optional[JsonMetricCache],
) -> Optional[FluencyScorer]:
    backend = str(config.get("backend") or "heuristic")
    if backend == "heuristic":
        return HeuristicFluencyScorer()
    if backend == "literary_quality":
        if api_client is None:
            raise ValueError("Literary quality backend requires api.model_config or inline api settings")
        fallback = HeuristicFluencyScorer() if _bool(config.get("fallback_on_error"), False) else None
        return ApiLiteraryQualityJudge(
            api_client,
            cache=cache,
            metric_version=str(config.get("metric_version") or "literary_quality.api_judge.v1"),
            fallback=fallback,
        )
    if backend != "api_judge":
        raise ValueError(f"Unsupported fluency backend: {backend}")
    if api_client is None:
        raise ValueError("Fluency API backend requires api.model_config or inline api settings")
    fallback = HeuristicFluencyScorer() if _bool(config.get("fallback_on_error"), False) else None
    return ApiFluencyJudge(
        api_client,
        cache=cache,
        metric_version=str(config.get("metric_version") or "fluency.api_judge.v1"),
        fallback=fallback,
    )


def _openai_client_from_config(config: Dict[str, Any], config_dir: Path) -> OpenAICompatibleClient:
    model_config = config.get("model_config")
    if model_config:
        model_config_path = _resolve_path(str(model_config), config_dir)
        merged = _load_json_object(model_config_path, "metric api.model_config")
        merged.update({key: value for key, value in config.items() if key != "model_config"})
        config = merged

    api_key_env = str(config.get("api_key_env") or "OPENAI_COMPATIBLE_API_KEY")
    endpoint = (
        config.get("endpoint")
        or config.get("chat_url")
        or config.get("url")
        or _endpoint_from_base_config(config)
        or os.environ.get("OPENAI_COMPATIBLE_CHAT_URL")
    )
    model = config.get("model") or os.environ.get("OPENAI_COMPATIBLE_MODEL")
    if not model:
        raise ValueError("Metric API model is required")
    if not endpoint:
        raise ValueError("Metric API endpoint is required")
    return OpenAICompatibleClient(
        model=str(model),
        endpoint=str(endpoint),
        api_key=config.get("api_key") or os.environ.get(api_key_env),
        timeout=_float_or_default(config.get("timeout"), None, 120.0),
        temperature=_optional_float(config.get("temperature", 0.0)),
        max_tokens=_optional_int(config.get("max_tokens", 1200)),
        stream=False,
    )


def _cache_from_config(config: Dict[str, Any], config_dir: Path) -> Optional[JsonMetricCache]:
    raw_path = config.get("cache_path")
    api_config = config.get("api") if isinstance(config.get("api"), dict) else {}
    raw_path = raw_path or api_config.get("cache_path")
    if not raw_path:
        return None
    return JsonMetricCache(str(_resolve_path(str(raw_path), config_dir)))


def _requires_api(config: Dict[str, Any]) -> bool:
    for section in ("entity", "fluency"):
        raw = config.get(section)
        if not isinstance(raw, dict):
            continue
        backend = str(raw.get("backend") or "heuristic")
        if backend in {"api_poetic", "hybrid_api", "api_judge", "literary_quality"}:
            return True
    return bool(config.get("api"))


def _looks_like_model_config(config: Dict[str, Any]) -> bool:
    return bool(config.get("provider") == "openai-compatible" and "entity" not in config and "fluency" not in config)


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must point to a JSON object")
    return dict(value)


def _resolve_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidate = (config_dir / path).resolve()
    if candidate.exists():
        return candidate
    return Path(raw).expanduser().resolve()


def _endpoint_from_base_config(config: Dict[str, Any]) -> Optional[str]:
    base_url = config.get("base_url")
    if not base_url:
        return None
    chat_path = str(config.get("chat_path") or "/v1/chat/completions")
    return urljoin(str(base_url).rstrip("/") + "/", chat_path.lstrip("/"))


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _float_or_default(primary: Any, secondary: Any, default: float) -> float:
    value = primary if primary is not None else secondary
    return default if value is None or value == "" else float(value)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


__all__ = ["MetricComponents", "build_metric_components"]
