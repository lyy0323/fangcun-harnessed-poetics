"""Repeated substring recall tool."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

from ..schemas import ToolRequest, ToolResponse
from ..text import cjk_chars, span_from_char_window
from .base import Tool


class RepetitionTool(Tool):
    name = "repetition"
    description = "Post-generation check for repeated CJK substrings and their positions."
    tool_group = "post_check_tool"
    usage_stage = "post_check"
    model_exposure = "post_check"
    tags = ("repetition", "after_generation")

    def __init__(self, min_ngram: int = 2, max_ngram: int = 6, max_issues: int = 50):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.max_issues = max_issues

    def run(self, request: ToolRequest) -> ToolResponse:
        chars = cjk_chars(request.text)
        total = len(chars)
        issues = []
        repeated_units = 0

        for ngram_size in range(self.max_ngram, self.min_ngram - 1, -1):
            seen: DefaultDict[str, List[int]] = defaultdict(list)
            for start in range(0, max(total - ngram_size + 1, 0)):
                key = "".join(info.char for info in chars[start : start + ngram_size])
                seen[key].append(start)

            repeated: List[Tuple[str, List[int]]] = [
                (key, starts) for key, starts in seen.items() if len(starts) > 1
            ]
            repeated.sort(key=lambda item: (len(item[1]), len(item[0])), reverse=True)

            for substring, starts in repeated:
                if len(issues) >= self.max_issues:
                    break
                spans = [span_from_char_window(chars, start, ngram_size).to_dict() for start in starts]
                repeated_units += ngram_size * (len(starts) - 1)
                issues.append(
                    {
                        "type": "repeated_substring",
                        "message": "Repeated substring detected.",
                        "substring": substring,
                        "ngram_size": ngram_size,
                        "occurrence_count": len(starts),
                        "spans": spans,
                    }
                )
            if len(issues) >= self.max_issues:
                break

        rate = repeated_units / total if total else 0.0
        return ToolResponse(
            self.name,
            passed=not issues,
            issues=issues,
            metrics={
                "total_cjk_chars": total,
                "repeated_substring_count": len(issues),
                "repeated_units": repeated_units,
                "content_repetition_rate": rate,
            },
        )
