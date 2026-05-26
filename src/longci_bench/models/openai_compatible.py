"""OpenAI-compatible chat completions adapter.

This uses urllib to avoid a required SDK dependency in the core package.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..schemas import ModelResponse, ToolCall
from .base import ModelClient


class _KeyRotator:
    """Thread-safe round-robin API key rotator."""

    def __init__(self, keys: List[str]):
        self._keys = list(keys)
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        name: Optional[str] = None,
        timeout: float = 120.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_reasoning: bool = False,
        stream_sink: Optional[Callable[[str], None]] = None,
        max_retries: int = 3,
        retry_backoff: float = 5.0,
    ):
        self.model = model
        self.endpoint = endpoint or os.environ.get("OPENAI_COMPATIBLE_CHAT_URL")
        self.name = name or model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.stream_reasoning = stream_reasoning
        self.stream_sink = stream_sink or _stderr_stream_sink
        self.system_prompt = system_prompt or (
            "You are generating Chinese ci. Use available tools when helpful and return the final ci text."
        )
        if not self.endpoint:
            raise ValueError("OpenAI-compatible endpoint is required")

        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        if api_keys and len(api_keys) > 0:
            self._key_rotator: Optional[_KeyRotator] = _KeyRotator(api_keys)
            self.api_key = api_keys[0]
        else:
            self._key_rotator = None
            self.api_key = api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")

    def _current_api_key(self) -> Optional[str]:
        if self._key_rotator:
            return self._key_rotator.next()
        return self.api_key

    def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        return self.generate_from_messages(messages, tools=tools, metadata=metadata)

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.stream:
            payload["stream"] = True

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        current_key = self._current_api_key()
        if current_key:
            headers["Authorization"] = f"Bearer {current_key}"

        req = urllib_request.Request(self.endpoint, data=body, headers=headers, method="POST")

        max_retries = getattr(self, "_max_retries", 3)
        retry_backoff = getattr(self, "_retry_backoff", 5.0)
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                req = urllib_request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib_request.urlopen(req, timeout=self.timeout) as response:
                    if self.stream:
                        return self._read_stream_response(response, metadata=metadata)
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib_error.HTTPError as exc:
                if exc.code == 429 and attempt < max_retries:
                    wait = retry_backoff * (2 ** attempt)
                    time.sleep(wait)
                    last_exc = exc
                    continue
                detail = _read_error_body(exc)
                raise RuntimeError(
                    f"OpenAI-compatible request failed: HTTP {exc.code} {exc.reason} for {self.endpoint}. "
                    f"Response body: {detail or '<empty>'}"
                ) from exc
            except urllib_error.URLError as exc:
                raise RuntimeError(f"OpenAI-compatible request failed for {self.endpoint}: {exc.reason}") from exc
            except socket.timeout as exc:
                raise RuntimeError(
                    f"OpenAI-compatible request timed out while reading response from {self.endpoint} "
                    f"after {self.timeout} seconds"
                ) from exc

        choices = data.get("choices") or [{}]
        first_choice = choices[0] if choices else {}
        message = (first_choice or {}).get("message") or {}
        return _model_response_from_message(
            self.model,
            message,
            request_metadata=metadata,
            stream=False,
            usage=data.get("usage"),
        )

    def _read_stream_response(self, response: Iterable[bytes], metadata: Optional[Dict[str, Any]] = None) -> ModelResponse:
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}

        for event in _iter_sse_events(response):
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                chunk = str(content)
                text_parts.append(chunk)
                self.stream_sink(chunk)

            reasoning_content = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning_content:
                chunk = str(reasoning_content)
                reasoning_parts.append(chunk)
                if self.stream_reasoning:
                    self.stream_sink(chunk)

            for tool_delta in delta.get("tool_calls") or []:
                _merge_tool_call_delta(tool_call_deltas, tool_delta)

        if text_parts or (self.stream_reasoning and reasoning_parts):
            self.stream_sink("\n")

        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        message: Dict[str, Any] = {"role": "assistant", "content": text}
        tool_calls = _stream_tool_calls_to_payloads(tool_call_deltas)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning:
            message["reasoning_content"] = reasoning
        return _model_response_from_message(
            self.model,
            message,
            request_metadata=metadata,
            stream=True,
        )


def _model_response_from_message(
    model: str,
    message: Dict[str, Any],
    request_metadata: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    usage: Optional[Dict[str, Any]] = None,
) -> ModelResponse:
    text = message.get("content") or ""
    tool_calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function", {})
        arguments = _parse_tool_arguments(function.get("arguments") or "{}")
        tool_calls.append(
            ToolCall(
                name=function.get("name", ""),
                arguments=arguments,
                call_id=call.get("id"),
            )
        )
    response_metadata = {"raw_model": model, "assistant_message": message}
    if stream:
        response_metadata["stream"] = True
    if usage:
        response_metadata["usage"] = usage
    if request_metadata:
        response_metadata["request_metadata"] = request_metadata
    return ModelResponse(text=text, tool_calls=tool_calls, metadata=response_metadata)


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {"value": parsed}
    return {"raw": arguments}


def _iter_sse_events(response: Iterable[bytes]) -> Iterable[Dict[str, Any]]:
    for raw_line in response:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").strip()
        else:
            line = str(raw_line).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid streaming response chunk: {payload}") from exc
        if isinstance(parsed, dict):
            yield parsed


def _merge_tool_call_delta(accumulator: Dict[int, Dict[str, Any]], delta: Dict[str, Any]) -> None:
    raw_index = delta.get("index")
    index = int(raw_index) if raw_index is not None else len(accumulator)
    current = accumulator.setdefault(
        index,
        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if delta.get("id"):
        current["id"] = delta["id"]
    if delta.get("type"):
        current["type"] = delta["type"]
    function_delta = delta.get("function") or {}
    current_function = current.setdefault("function", {"name": "", "arguments": ""})
    if function_delta.get("name"):
        current_function["name"] = str(current_function.get("name") or "") + str(function_delta["name"])
    if function_delta.get("arguments"):
        current_function["arguments"] = str(current_function.get("arguments") or "") + str(
            function_delta["arguments"]
        )


def _stream_tool_calls_to_payloads(tool_call_deltas: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_calls = []
    for index in sorted(tool_call_deltas):
        call = tool_call_deltas[index]
        function = call.get("function") or {}
        name = function.get("name") or ""
        arguments = function.get("arguments") or "{}"
        if not name:
            continue
        tool_calls.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": call.get("type") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return tool_calls


def _stderr_stream_sink(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


def _read_error_body(exc: urllib_error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")
