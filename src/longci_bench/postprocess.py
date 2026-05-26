"""Post-processing utilities for model outputs."""

from __future__ import annotations

import re
from typing import Optional


_BOXED_RE = re.compile(r"\\{1,2}boxed\{(.+)\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
PLACEHOLDER_CHAR = "□"  # □


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...}, or return text as-is if no box found."""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def has_placeholder(text: str) -> bool:
    """Check if text contains □ placeholder characters."""
    return PLACEHOLDER_CHAR in text


def normalize_poem_output(text: str) -> str:
    """Extract boxed content, strip thinking tags and wrapper artifacts."""
    # Strip <think>...</think> blocks (some models put reasoning in content)
    result = _THINK_RE.sub("", text)
    result = extract_boxed(result)
    for prefix in ("```", "```text", "```plaintext"):
        if result.startswith(prefix):
            result = result[len(prefix):]
    if result.endswith("```"):
        result = result[:-3]
    return result.strip()


def validate_output(text: str) -> tuple:
    """Validate extracted output. Returns (is_valid, issues).

    Checks:
    - Not empty
    - No □ placeholder characters
    - Has CJK content
    """
    issues = []
    if not text.strip():
        issues.append("empty_output")
        return False, issues
    if has_placeholder(text):
        issues.append("contains_placeholder")
    cjk_count = sum(1 for ch in text if "一" <= ch <= "鿿")
    if cjk_count < 10:
        issues.append("insufficient_cjk")
    return len(issues) == 0, issues
