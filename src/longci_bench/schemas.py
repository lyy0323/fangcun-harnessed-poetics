"""Shared JSON-compatible schemas.

The project keeps these as dataclasses instead of Pydantic models so the core
package can run in minimal environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


JsonDict = Dict[str, Any]


@dataclass
class TextSpan:
    start: int
    end: int
    line: Optional[int] = None
    column: Optional[int] = None
    text: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class ToolRequest:
    tool_name: str
    text: str
    cipai: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ToolRequest":
        return cls(
            tool_name=str(data.get("tool_name", "")),
            text=str(data.get("text", "")),
            cipai=data.get("cipai"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ToolResponse:
    tool_name: str
    passed: bool
    issues: List[JsonDict] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ToolResponse":
        return cls(
            tool_name=str(data.get("tool_name", "")),
            passed=bool(data.get("passed", False)),
            issues=list(data.get("issues") or []),
            metrics=dict(data.get("metrics") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> JsonDict:
        return {
            "tool_name": self.tool_name,
            "passed": self.passed,
            "issues": self.issues,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    name: str
    arguments: JsonDict = field(default_factory=dict)
    call_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ToolCall":
        return cls(
            name=str(data.get("name", "")),
            arguments=dict(data.get("arguments") or {}),
            call_id=data.get("call_id") or data.get("id"),
        )

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass
class ModelResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelResponse":
        tool_calls = [ToolCall.from_mapping(item) for item in data.get("tool_calls", [])]
        return cls(
            text=str(data.get("text", "")),
            tool_calls=tool_calls,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> JsonDict:
        return {
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "metadata": self.metadata,
        }


@dataclass
class GenerationRecord:
    id: str
    split: str
    cipai: str
    prompt: str
    text: str
    model: str
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GenerationRecord":
        return cls(
            id=str(data.get("id", "")),
            split=str(data.get("split", "")),
            cipai=str(data.get("cipai", "")),
            prompt=str(data.get("prompt", "")),
            text=str(data.get("text", "")),
            model=str(data.get("model", "")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


def issue(
    issue_type: str,
    message: str,
    span: Optional[TextSpan] = None,
    expected: Optional[str] = None,
    actual: Optional[str] = None,
    **extra: Any,
) -> JsonDict:
    payload: JsonDict = {
        "type": issue_type,
        "message": message,
        "span": span.to_dict() if span else None,
        "expected": expected,
        "actual": actual,
    }
    payload.update(extra)
    return payload
