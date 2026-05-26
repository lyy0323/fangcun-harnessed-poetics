"""Dependency-light fluency scoring.

This is a deterministic placeholder. Production experiments can replace it
with an API or local LM scorer while keeping the same metric name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Any, Dict, List, Optional

from ..text import count_cjk, iter_nonempty_lines


@dataclass
class FluencyResult:
    score: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FluencyScorer:
    def score(self, text: str) -> float:
        return self.score_result(text).score

    def score_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FluencyResult:
        raise NotImplementedError


class HeuristicFluencyScorer(FluencyScorer):
    def score_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FluencyResult:
        score = heuristic_fluency_score(text)
        lines = [line for _, line, _ in iter_nonempty_lines(text)]
        return FluencyResult(
            score=score,
            dimensions={"line_balance_and_punctuation": score},
            backend="heuristic",
            metadata={"line_count": len(lines)},
        )


def heuristic_fluency_score(text: str) -> float:
    lines = [line for _, line, _ in iter_nonempty_lines(text)]
    if not lines:
        return 0.0

    line_lengths = [count_cjk(line) for line in lines]
    total_chars = sum(line_lengths)
    if total_chars == 0:
        return 0.0

    punctuation_bonus = 0.15 if any(mark in text for mark in "，。！？；、") else 0.0
    length_balance = 1.0
    if len(line_lengths) > 1:
        mean = statistics.mean(line_lengths)
        variance = statistics.pvariance(line_lengths)
        length_balance = max(0.0, 1.0 - (variance / max(mean * mean, 1.0)))

    very_short_penalty = 0.25 if total_chars < 8 else 0.0
    score = 0.2 + 0.65 * length_balance + punctuation_bonus - very_short_penalty
    return max(0.0, min(1.0, score))
