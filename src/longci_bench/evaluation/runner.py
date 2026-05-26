"""JSONL evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import GenerationRecord
from ..tools import build_default_registry
from .config import build_metric_components
from .metrics import evaluate_records


def read_jsonl(path: str) -> List[GenerationRecord]:
    records: List[GenerationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(GenerationRecord.from_mapping(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def evaluate_jsonl(
    input_path: str,
    spec_path: Optional[str] = None,
    imagery_path: Optional[str] = None,
    tool_config_path: Optional[str] = None,
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
    metric_config_path: Optional[str] = None,
    metric_model_config_path: Optional[str] = None,
    style_tool_config_path: Optional[str] = None,
) -> Dict:
    registry = build_default_registry(
        spec_path=spec_path,
        imagery_path=imagery_path,
        external_tool_config_path=tool_config_path,
        toolset_config_path=toolset_config_path,
        poetics_path=poetics_path,
        poetics_config_dir=poetics_config_dir,
        poetics_data_dir=poetics_data_dir,
        poetics_genre=poetics_genre,
        poetics_rhyme_book=poetics_rhyme_book,
        poetics_ensure_longpu=poetics_ensure_longpu,
        fangcun_path=fangcun_path,
        fangcun_config_dir=fangcun_config_dir,
        fangcun_data_dir=fangcun_data_dir,
        fangcun_genre=fangcun_genre,
        fangcun_rhyme_book=fangcun_rhyme_book,
        fangcun_ensure_longpu=fangcun_ensure_longpu,
        style_tool_config_path=style_tool_config_path,
    )
    metric_components = build_metric_components(
        metric_config_path=metric_config_path,
        metric_model_config_path=metric_model_config_path,
    )
    return evaluate_records(
        read_jsonl(input_path),
        registry,
        entity_extractor=metric_components.entity_extractor,
        fluency_scorer=metric_components.fluency_scorer,
        fluency_pass_threshold=metric_components.fluency_pass_threshold,
    )
