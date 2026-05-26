"""Baseline prosody validation tool.

The real project should replace or extend this with the authoritative long-ci
meter checker. This baseline validates line counts and CJK character counts.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..schemas import ToolRequest, ToolResponse, issue
from ..text import count_cjk, iter_nonempty_lines
from .base import Tool


class ProsodyTool(Tool):
    name = "prosody"
    description = "Validate generated ci text against the requested cipai prosody spec."
    tool_group = "extra_tool"
    usage_stage = "validation"
    model_exposure = "extra"
    tags = ("prosody", "placeholder")

    def __init__(self, specs: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self.specs: Dict[str, Mapping[str, Any]] = dict(specs or {})

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["required"] = ["text", "cipai"]
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        issues = []
        if not request.cipai:
            issues.append(
                issue(
                    "missing_cipai",
                    "Prosody validation requires a cipai name.",
                    expected="non-empty cipai",
                    actual=None,
                )
            )
            return ToolResponse(self.name, passed=False, issues=issues, metrics={"violation_count": 1})

        spec = self.specs.get(request.cipai)
        if spec is None:
            issues.append(
                issue(
                    "missing_spec",
                    "No prosody spec is available for this cipai.",
                    expected="known cipai spec",
                    actual=request.cipai,
                    cipai=request.cipai,
                )
            )
            return ToolResponse(self.name, passed=False, issues=issues, metrics={"violation_count": 1})

        expected_lengths = list(spec.get("line_lengths") or [])
        lines = list(iter_nonempty_lines(request.text))

        if expected_lengths and len(lines) != len(expected_lengths):
            issues.append(
                issue(
                    "line_count",
                    "Line count does not match cipai spec.",
                    expected=str(len(expected_lengths)),
                    actual=str(len(lines)),
                    cipai=request.cipai,
                )
            )

        for index, expected_length in enumerate(expected_lengths):
            if index >= len(lines):
                issues.append(
                    issue(
                        "missing_line",
                        "Expected line is missing.",
                        expected=f"{expected_length} CJK characters",
                        actual="missing",
                        line=index + 1,
                    )
                )
                continue
            _, content, span = lines[index]
            actual_length = count_cjk(content)
            if actual_length != int(expected_length):
                issues.append(
                    issue(
                        "line_length",
                        "Line length does not match cipai spec.",
                        span=span,
                        expected=f"{expected_length} CJK characters",
                        actual=f"{actual_length} CJK characters",
                        line=index + 1,
                    )
                )

        return ToolResponse(
            self.name,
            passed=not issues,
            issues=issues,
            metrics={
                "violation_count": len(issues),
                "checked_lines": len(lines),
                "expected_lines": len(expected_lengths),
            },
            metadata={"cipai": request.cipai, "spec_notes": spec.get("notes")},
        )
