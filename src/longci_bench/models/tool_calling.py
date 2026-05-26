"""Model/tool orchestration that is backend-agnostic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schemas import ModelResponse, ToolRequest, ToolResponse, issue
from ..tools import ToolRegistry
from .base import ModelClient


@dataclass
class ToolCallingTrace:
    rounds: List[Dict] = field(default_factory=list)


class ToolCallingRunner:
    def __init__(
        self,
        model: ModelClient,
        registry: ToolRegistry,
        max_tool_rounds: int = 3,
        system_prompt: Optional[str] = None,
        max_tool_calls_per_tool: Optional[int] = None,
        require_final_prosody_pass: bool = False,
        final_validation_tool_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls_per_tool = max_tool_calls_per_tool
        self.require_final_prosody_pass = require_final_prosody_pass
        self.final_validation_tool_names = list(final_validation_tool_names or ["prosody"])
        self._tool_call_counts: Dict[str, int] = {}
        self.system_prompt = system_prompt or (
            "You are generating Chinese ci. Use available tools when helpful, "
            "then return the final ci text only when you are satisfied."
        )

    def run(self, prompt: str, cipai: Optional[str] = None) -> ModelResponse:
        self._tool_call_counts = {}
        tools = self.registry.model_tool_schemas()
        if hasattr(self.model, "generate_from_messages"):
            return self._run_chat_messages(prompt, tools, cipai=cipai)
        return self._run_prompt_fallback(prompt, tools, cipai=cipai)

    def _run_prompt_fallback(self, prompt: str, tools: List[Dict], cipai: Optional[str] = None) -> ModelResponse:
        trace = ToolCallingTrace()
        working_prompt = prompt
        final_response = ModelResponse(text="")

        for round_index in range(self.max_tool_rounds + 1):
            response = self.model.generate(
                working_prompt,
                tools=tools,
                metadata={"cipai": cipai, "tool_round": round_index},
            )
            final_response = response
            if not response.tool_calls:
                break

            observations = []
            for call in response.tool_calls:
                arguments = dict(call.arguments)
                arguments.setdefault("text", response.text)
                if cipai:
                    arguments.setdefault("cipai", cipai)
                metadata = {k: v for k, v in arguments.items() if k not in ("text", "cipai")}
                tool_response = self.registry.run(
                    ToolRequest(
                        tool_name=call.name,
                        text=str(arguments.get("text", "")),
                        cipai=arguments.get("cipai"),
                        metadata=metadata,
                    )
                )
                observations.append({"call": call.to_dict(), "response": tool_response.to_dict()})

            trace.rounds.append({"round": round_index, "observations": observations})
            working_prompt = (
                f"{prompt}\n\nTool observations:\n"
                f"{json.dumps([_slim_tool_response_from_dict(o['response']) for o in observations], ensure_ascii=False)}\n"
                "Revise or finalize the ci text."
            )

        final_response.metadata = dict(final_response.metadata)
        final_response.metadata["tool_trace"] = trace.rounds
        _apply_unfinalized_tool_fallback(final_response)
        return final_response

    def _run_chat_messages(self, prompt: str, tools: List[Dict], cipai: Optional[str] = None) -> ModelResponse:
        trace = ToolCallingTrace()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        final_response = ModelResponse(text="")
        generate_from_messages = getattr(self.model, "generate_from_messages")

        for round_index in range(self.max_tool_rounds + 1):
            response = generate_from_messages(
                messages,
                tools=tools,
                metadata={"cipai": cipai, "tool_round": round_index},
            )
            final_response = response
            if not response.tool_calls:
                if not response.text and round_index < self.max_tool_rounds:
                    messages.append(_normalize_assistant_message(response.metadata.get("assistant_message") or _assistant_message(response)))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No visible ci body was returned. Continue the tool workflow: either call the next required "
                                "tool with the ci body in arguments.text, or provide the final ci body in visible content. "
                                "Do not put the draft only in reasoning_content."
                            ),
                        }
                    )
                    continue
                if self.require_final_prosody_pass and response.text:
                    validations = self._run_final_validations(response.text, cipai)
                    validation_passed = all(item.passed for item in validations)
                    trace.rounds.append(
                        {
                            "round": round_index,
                            "final_validation": _final_validation_trace(validations, response.text, cipai),
                        }
                    )
                    response.metadata = dict(response.metadata)
                    response.metadata["final_validation"] = _final_validation_metadata(validations)
                    if validation_passed or round_index >= self.max_tool_rounds:
                        if not validation_passed:
                            response.metadata["finish_reason"] = "final_validation_failed"
                        break
                    messages.append(_normalize_assistant_message(response.metadata.get("assistant_message") or _assistant_message(response)))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Final validation failed. Revise the ci using the validation JSON below. "
                                "You may call tools again, and you must only provide final content after all final validations pass.\n"
                                f"{json.dumps(_final_validation_metadata(validations), ensure_ascii=False)}"
                            ),
                        }
                    )
                    continue
                break

            assistant_message = _normalize_assistant_message(response.metadata.get("assistant_message") or _assistant_message(response))
            messages.append(assistant_message)
            observations = []
            for call_index, call in enumerate(response.tool_calls):
                tool_response = self._run_tool_call(call.name, call.arguments, response.text, cipai)
                observations.append({"call": call.to_dict(), "response": tool_response.to_dict()})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id or f"call_{round_index}_{call_index}",
                        "name": call.name,
                        "content": json.dumps(_slim_tool_response(tool_response), ensure_ascii=False),
                    }
                )

            trace.rounds.append({"round": round_index, "observations": observations})

            # Inject remaining rounds hint (chat path only)
            remaining = self.max_tool_rounds - round_index - 1
            if remaining <= 5 and remaining > 0:
                messages.append({
                    "role": "user",
                    "content": f"[系统提示] 剩余工具调用轮次：{remaining}。{'请尽快完成修订并输出最终结果。' if remaining <= 2 else '请注意控制节奏。'}",
                })
            elif remaining <= 0:
                messages.append({
                    "role": "user",
                    "content": "[系统提示] 这是最后一轮，请立即将当前最佳版本包裹在 \\boxed{} 中输出。",
                })

        final_response.metadata = dict(final_response.metadata)
        final_response.metadata["tool_trace"] = trace.rounds
        final_response.metadata["messages"] = messages
        _apply_unfinalized_tool_fallback(final_response)
        return final_response

    def _run_tool_call(
        self,
        name: str,
        arguments: Dict,
        candidate_text: str,
        cipai: Optional[str],
        count_toward_limit: bool = True,
    ):
        if count_toward_limit and self.max_tool_calls_per_tool is not None:
            count = self._tool_call_counts.get(name, 0)
            if count >= self.max_tool_calls_per_tool:
                return ToolResponse(
                    name,
                    passed=False,
                    issues=[
                        issue(
                            "tool_call_limit",
                            f"Tool '{name}' call limit reached; finalize the poem without calling this tool again.",
                            expected=f"at most {self.max_tool_calls_per_tool} calls",
                            actual=f"{count + 1} attempted calls",
                        )
                    ],
                    metrics={"call_count": count, "call_limit": self.max_tool_calls_per_tool},
                    metadata={"source": "runner"},
                )
            self._tool_call_counts[name] = count + 1
        request_arguments = dict(arguments)
        request_arguments.setdefault("text", candidate_text)
        if cipai:
            request_arguments.setdefault("cipai", cipai)
        metadata = {k: v for k, v in request_arguments.items() if k not in ("text", "cipai")}
        return self.registry.run(
            ToolRequest(
                tool_name=name,
                text=str(request_arguments.get("text", "")),
                cipai=request_arguments.get("cipai"),
                metadata=metadata,
            )
        )

    def _run_final_validations(self, text: str, cipai: Optional[str]) -> List[ToolResponse]:
        validations = []
        for tool_name in self.final_validation_tool_names:
            validations.append(
                self._run_tool_call(
                    tool_name,
                    {"text": text},
                    text,
                    cipai,
                    count_toward_limit=False,
                )
            )
        return validations


def _slim_tool_response(tool_response: ToolResponse) -> Dict:
    """Strip bulky fields before sending tool results back to the LLM.

    The full response is preserved in the trace via observations[].response,
    but the LLM only needs: passed, issues, metrics (without display_segments),
    and hints.
    """
    d = tool_response.to_dict()
    return _slim_tool_response_from_dict(d)


def _slim_tool_response_from_dict(d: Dict) -> Dict:
    """Strip bulky fields from a tool response dict."""
    d = dict(d)
    meta = dict(d.get("metadata") or {})
    d["metadata"] = meta
    meta.pop("raw_response", None)
    meta.pop("warnings", None)
    metrics = dict(d.get("metrics") or {})
    d["metrics"] = metrics
    metrics.pop("display_segments", None)
    results = meta.get("results")
    if isinstance(results, list):
        results = [_slim_rule_entry(r) if isinstance(r, dict) else r for r in results]
        if len(results) > 20:
            meta["results"] = results[:20]
            meta["results_truncated"] = len(results)
        else:
            meta["results"] = results
    return d


def _slim_rule_entry(entry: Dict) -> Dict:
    """For rules_list results: keep only name, char_count, readable; drop tone_pattern etc."""
    if "readable" in entry or "tone_pattern" in entry:
        slim = {"name": entry.get("name"), "char_count": entry.get("char_count")}
        if "readable" in entry:
            slim["readable"] = entry["readable"]
        if "sentence_count" in entry:
            slim["sentence_count"] = entry["sentence_count"]
        return slim
    return entry


def _assistant_message(response: ModelResponse) -> Dict:
    message: Dict = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for index, call in enumerate(response.tool_calls)
        ]
    return message


def _normalize_assistant_message(msg: Dict) -> Dict:
    """Ensure assistant message content is None (not empty string) when absent."""
    if not msg.get("content"):
        msg["content"] = None
    return msg


def _final_validation_trace(validations: List[ToolResponse], text: str, cipai: Optional[str]) -> Dict:
    if len(validations) == 1:
        validation = validations[0]
        return {
            "call": {
                "name": validation.tool_name,
                "arguments": {"text": text, "cipai": cipai},
                "call_id": "final_validation",
            },
            "response": validation.to_dict(),
        }
    return {
        "passed": all(validation.passed for validation in validations),
        "validations": [
            {
                "call": {
                    "name": validation.tool_name,
                    "arguments": {"text": text, "cipai": cipai},
                    "call_id": f"final_validation_{index}",
                },
                "response": validation.to_dict(),
            }
            for index, validation in enumerate(validations)
        ],
    }


def _final_validation_metadata(validations: List[ToolResponse]) -> Dict:
    if len(validations) == 1:
        return validations[0].to_dict()
    return {
        "passed": all(validation.passed for validation in validations),
        "responses": {validation.tool_name: validation.to_dict() for validation in validations},
    }


def _apply_unfinalized_tool_fallback(response: ModelResponse) -> None:
    if response.text or not response.tool_calls:
        return
    candidates = []
    for call in reversed(response.tool_calls):
        candidate = call.arguments.get("text")
        if candidate:
            candidates.append((call.name, str(candidate)))
    for name, candidate in candidates:
        if name == "prosody" or len(candidate) >= 40:
            response.text = candidate
            response.metadata["finish_reason"] = "max_tool_rounds_exhausted"
            response.metadata["fallback_text_source"] = "last_draft_tool_call.arguments.text"
            return
