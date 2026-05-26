"""Configuration loaders for benchmark assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cipai_specs(path: Optional[str]) -> Dict[str, Mapping[str, Any]]:
    data = load_json(path)
    specs = data.get("specs", data)
    if not isinstance(specs, dict):
        raise ValueError("cipai specs must be a JSON object or contain a 'specs' object")
    return specs


def load_imagery_lexicon(path: Optional[str]) -> Dict[str, Any]:
    data = load_json(path)
    categories = data.get("categories", data)
    if not isinstance(categories, dict):
        raise ValueError("imagery lexicon must be a JSON object or contain a 'categories' object")
    return categories


def load_external_tool_configs(path: Optional[str]) -> List[Dict[str, Any]]:
    data = load_json(path)
    tools = data.get("tools", data)
    if not isinstance(tools, list):
        raise ValueError("external tool config must be a JSON array or contain a 'tools' array")
    return [dict(item) for item in tools]
