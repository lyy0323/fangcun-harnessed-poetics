"""FANGCUN REST API tool adapter.

Dispatches tool calls to the real FANGCUN API endpoints with correct
HTTP methods and parameter styles (GET query params vs POST JSON body).
Includes per-endpoint rate limiting and retry logic.
"""

from __future__ import annotations

import json
import time
import threading
from typing import Any, Dict, List, Mapping, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from ..schemas import ToolRequest, ToolResponse
from .base import Tool


class RateLimiter:
    """Simple sliding-window rate limiter (thread-safe)."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self.period]
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                sleep_until = self._timestamps[0] + self.period
            time.sleep(max(0.05, sleep_until - time.monotonic()))


DEFAULT_RATE_LIMITS: Dict[str, int] = {
    "validate_meter": 55,
    "free_rhyme": 55,
    "dictionary_search": 110,
    "dictionary_allusion": 110,
    "_default": 55,
}


class FangcunAPITool(Tool):
    """Calls a single FANGCUN API endpoint with correct HTTP method/params."""

    # Class-level registry so tools can cross-call each other for bundling
    _all_tools: Dict[str, "FangcunAPITool"] = {}

    def __init__(
        self,
        name: str,
        base_url: str,
        path: str,
        method: str = "GET",
        description: str = "",
        timeout: float = 30.0,
        input_schema: Optional[Mapping[str, Any]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.method = method.upper()
        self.description = description
        self.timeout = timeout
        self._input_schema = dict(input_schema or {})
        self._allowed_params: set = set(
            (self._input_schema.get("properties") or {}).keys()
        )
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema or super().input_schema()

    def run(self, request: ToolRequest) -> ToolResponse:
        params = self._build_params(request)
        if self.name == "validate_meter":
            rejected = _enforce_genre_rhyme_book(params)
            if rejected:
                return rejected
        try:
            data = self._call_api(params)
        except Exception as exc:
            return ToolResponse(
                tool_name=self.name,
                passed=False,
                issues=[{"type": "api_error", "message": str(exc)}],
                metadata={"hints": _error_hints(self.name, params, exc)},
            )
        return _to_tool_response(self.name, data)

    def _build_params(self, request: ToolRequest) -> Dict[str, Any]:
        """Build API params: only pass parameters declared in this tool's schema."""
        params: Dict[str, Any] = {}
        # Map ToolRequest standard fields to API param names
        field_map = {"poem_text": request.text, "rule_name": request.cipai}
        for api_name, value in field_map.items():
            if api_name in self._allowed_params and value:
                params[api_name] = value
        # Pass through metadata keys that match the schema
        if request.metadata:
            for key, value in request.metadata.items():
                if key in self._allowed_params and value is not None:
                    params[key] = value
        return params

    def _call_api(self, params: Dict[str, Any]) -> Any:
        if self.rate_limiter:
            self.rate_limiter.wait()

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_request(params)
            except HTTPError as exc:
                if exc.code == 429:
                    wait = self.retry_backoff * (2 ** attempt)
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise
            except (URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    last_exc = exc
                    continue
                raise
        raise last_exc or RuntimeError("max retries exceeded")

    def _do_request(self, params: Dict[str, Any]) -> Any:
        if self.method == "GET":
            query = {k: _serialize_param(v) for k, v in params.items() if v is not None}
            url = f"{self.base_url}{self.path}"
            if query:
                url = f"{url}?{urlencode(query)}"
            req = urllib_request.Request(url, method="GET")
        else:
            url = f"{self.base_url}{self.path}"
            body = json.dumps(params, ensure_ascii=False).encode("utf-8")
            req = urllib_request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

        with urllib_request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _serialize_param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _error_hints(tool_name: str, params: Dict[str, Any], exc: Exception) -> List[str]:
    """Generate hints for common API errors."""
    err = str(exc)
    hints: List[str] = []

    if "404" in err:
        if tool_name == "rhyme_lookup":
            cat = params.get("category", "")
            book = params.get("book", "Pingshuiyun")
            # Auto-bundle: fetch rhyme_list to show correct category names
            categories = _auto_rhyme_list(book)
            if categories:
                cat_names = "、".join(c["name"] for c in categories[:8])
                hints.append(
                    f"韵部名「{cat}」不存在。该韵书的韵部名为：{cat_names} 等。"
                    f"请使用以上精确名称重新调用 rhyme_lookup。"
                )
            else:
                hints.append(
                    f"韵部名「{cat}」不存在。词林正韵格式为「第1部_平」（阿拉伯数字），"
                    f"平水韵格式为「一东」。调用 rhyme_list(book=...) 查看正确名称。"
                )
        elif tool_name == "examples":
            rule = params.get("rule_name", "")
            hints.append(
                f"规则名「{rule}」不存在。建议先调用 rules_list(genre=Ci/Shi, search=词牌名) 查询正确的规则名。"
            )
        else:
            hints.append("资源不存在，请检查参数拼写。")

    if "429" in err:
        hints.append("请求频率超限，请稍后重试。")

    return hints


# ---------------------------------------------------------------------------
# Genre ↔ rhyme book enforcement
# ---------------------------------------------------------------------------

GENRE_RHYME_BOOK = {
    "Shi": "Pingshuiyun",
    "Ci": "Cilinzhengyun",
}

RHYME_BOOK_GENRE = {v: k for k, v in GENRE_RHYME_BOOK.items()}


def _enforce_genre_rhyme_book(params: Dict[str, Any]) -> Optional[ToolResponse]:
    """Reject validate_meter calls with mismatched genre/rhyme_book_name.

    Rules:
      Shi → must use Pingshuiyun (平水韵)
      Ci  → must use Cilinzhengyun (词林正韵)

    If rhyme_book_name is missing, auto-fill the correct one.
    If genre is missing, infer from rhyme_book_name or default to Ci.
    If both present but mismatched, reject with a hint.
    """
    genre = params.get("genre")
    book = params.get("rhyme_book_name")

    # Both present — check match
    if genre and book:
        expected_book = GENRE_RHYME_BOOK.get(genre)
        if expected_book and book != expected_book:
            return ToolResponse(
                tool_name="validate_meter",
                passed=False,
                issues=[{
                    "type": "param_mismatch",
                    "message": (
                        f"genre={genre} 应使用 {expected_book}，"
                        f"但传入了 rhyme_book_name={book}。"
                    ),
                }],
                metadata={
                    "rejected": True,
                    "hints": [
                        f"体裁与韵书不匹配：{genre} 必须搭配 {expected_book}。"
                        f"请修改 rhyme_book_name 为 {expected_book} 后重新调用。"
                        f"（诗用平水韵 Pingshuiyun，词用词林正韵 Cilinzhengyun）"
                    ],
                },
            )
        return None

    # Auto-fill missing fields
    if genre and not book:
        params["rhyme_book_name"] = GENRE_RHYME_BOOK.get(genre, "Cilinzhengyun")
    elif book and not genre:
        params["genre"] = RHYME_BOOK_GENRE.get(book, "Ci")
    elif not genre and not book:
        params["genre"] = "Ci"
        params["rhyme_book_name"] = "Cilinzhengyun"

    # Auto-inject warnings based on genre
    if "warnings" not in params:
        resolved_genre = params.get("genre", "Ci")
        if resolved_genre == "Ci":
            params["warnings"] = ["2gram", "rhyme_duplicate"]
        else:
            params["warnings"] = "default"

    return None


def _to_tool_response(tool_name: str, data: Any) -> ToolResponse:
    """Convert raw FANGCUN API response to ToolResponse, with actionable hints."""

    # --- List responses (rules_list, dictionary_search, dictionary_allusion) ---
    if isinstance(data, list):
        return ToolResponse(
            tool_name=tool_name,
            passed=True,
            metadata={"results": data},
        )

    if not isinstance(data, dict):
        return ToolResponse(tool_name=tool_name, passed=True, metadata={"results": data})

    # --- validate_meter (has is_valid / errors) ---
    if data.get("is_valid") is not None or "errors" in data:
        is_valid = data.get("is_valid")
        errors = data.get("errors", [])
        passed = is_valid if is_valid is not None else (len(errors) == 0)
        issues = []
        for err in errors:
            if isinstance(err, dict):
                issues.append({
                    "type": err.get("error_type", "error"),
                    "message": err.get("message", ""),
                    "position": err.get("position"),
                    "character": err.get("character"),
                })
        metrics: Dict[str, Any] = {}
        if "rhyme_name" in data:
            metrics["rhyme_name"] = data["rhyme_name"]
        if "rhyme_chars" in data:
            metrics["rhyme_chars"] = data["rhyme_chars"]
        if "rhyme_positions" in data:
            metrics["rhyme_positions"] = data["rhyme_positions"]
        if "closest_rule" in data:
            metrics["closest_rule"] = data["closest_rule"]
        if "display_segments" in data:
            metrics["display_segments"] = data["display_segments"]
        if errors:
            tone_errors = sum(1 for e in errors if isinstance(e, dict) and e.get("error_type") == "Tone")
            rhyme_errors = sum(1 for e in errors if isinstance(e, dict) and e.get("error_type") == "Rhyme")
            metrics["tone_error_count"] = tone_errors
            metrics["rhyme_error_count"] = rhyme_errors
            metrics["violation_count"] = len(errors)
        if "display_segments" in data:
            total_chars = sum(
                len(seg.get("text_chars", []))
                for seg in data["display_segments"]
                if isinstance(seg, dict)
            )
            metrics["checked_chars"] = total_chars
        warnings = data.get("warnings", [])
        hints = _build_hints(tool_name, data, errors, bool(passed))
        return ToolResponse(
            tool_name=tool_name,
            passed=bool(passed),
            issues=issues,
            metrics=metrics,
            metadata={
                "raw_response": data,
                "warning_count": len(warnings),
                "warnings": warnings,
                "hints": hints,
            },
        )

    # --- All other dict responses (char_lookup, rhyme_lookup, free_rhyme, etc) ---
    # Put the data directly in metadata.results so it survives _slim_tool_response
    return ToolResponse(
        tool_name=tool_name,
        passed=True,
        metadata={"results": data},
    )


# ---------------------------------------------------------------------------
# Orchestration hints — actionable next-step suggestions appended to tool
# responses so the LLM agent knows what to do with errors.
# ---------------------------------------------------------------------------

def _build_hints(tool_name: str, data: Dict, errors: List, passed: bool) -> List[str]:
    """Generate actionable hints from validate_meter, with auto-bundled lookups."""
    if tool_name != "validate_meter":
        return []

    if passed:
        return ["格律校验通过（零错误），可以输出最终结果。将诗词正文包裹在 \\boxed{} 中输出。"]

    hints: List[str] = []
    total_errors = len(errors)
    bundle = total_errors <= 10  # only auto-lookup when errors are few enough to fix individually

    # --- Structural mismatch ---
    closest = data.get("closest_rule") or {}
    expected_chars = closest.get("char_count")
    segments = data.get("display_segments", [])
    actual_chars = sum(len(seg.get("text_chars", [])) for seg in segments if isinstance(seg, dict))
    if expected_chars and actual_chars and actual_chars != expected_chars:
        hints.append(
            f"字数不符：目标 {expected_chars} 字，实际 {actual_chars} 字。"
            f"请调整正文字数后重新校验。"
        )

    # --- Tone errors with auto-bundled dictionary suggestions ---
    tone_errs = [e for e in errors if isinstance(e, dict) and e.get("error_type") == "Tone"]
    if tone_errs:
        need_ping = []
        need_ze = []
        for e in tone_errs:
            char = e.get("character", "?")
            pos = e.get("position", "?")
            msg = e.get("message", "")
            if "应为平" in msg:
                need_ping.append((pos, char))
            else:
                need_ze.append((pos, char))

        if need_ping:
            chars_str = "、".join(f"位置{p}「{c}」" for p, c in need_ping[:5])
            hints.append(
                f"平仄错误（应平实仄）：{chars_str}"
                f"{'等' if len(need_ping) > 5 else ''}，共 {len(need_ping)} 处。"
            )
        if need_ze:
            chars_str = "、".join(f"位置{p}「{c}」" for p, c in need_ze[:5])
            hints.append(
                f"平仄错误（应仄实平）：{chars_str}"
                f"{'等' if len(need_ze) > 5 else ''}，共 {len(need_ze)} 处。"
            )

        # Auto-bundle: dictionary_search suggestions for first few tone errors
        if bundle:
            all_text_chars = []
            for seg in segments:
                if isinstance(seg, dict):
                    all_text_chars.extend(seg.get("text_chars", []))
            if all_text_chars:
                tone_suggestions = _auto_tone_suggestions(tone_errs, all_text_chars)
                hints.extend(tone_suggestions)

    # --- Rhyme errors: auto-bundle lookups ---
    rhyme_errs = [e for e in errors if isinstance(e, dict) and e.get("error_type") == "Rhyme"]
    if rhyme_errs:
        rhyme_name = data.get("rhyme_name")
        rhyme_book = "Cilinzhengyun"
        genre = data.get("closest_rule", {}).get("genre")
        if genre == "Shi":
            rhyme_book = "Pingshuiyun"

        err_chars = [e.get("character", "?") for e in rhyme_errs]
        chars_str = "、".join(f"「{c}」" for c in err_chars[:5])

        if rhyme_name:
            # Auto-bundle: fetch top 20 rhyme chars
            top_chars = _auto_rhyme_lookup(rhyme_name, rhyme_book, limit=20) if bundle else []
            if top_chars:
                hints.append(
                    f"押韵错误：韵脚 {chars_str} 不在目标韵部「{rhyme_name}」中。"
                    f"「{rhyme_name}」常用字：{'、'.join(top_chars)}。"
                    f"请从中选择语义合适的字替换韵脚。"
                )
            else:
                hints.append(
                    f"押韵错误：韵脚 {chars_str} 不在目标韵部「{rhyme_name}」中。"
                    f"调用 rhyme_lookup(category={rhyme_name}, book={rhyme_book}, limit=20) 查看常用字。"
                )
        else:
            # Auto-bundle: lookup each error char's rhyme category
            char_rhymes = _auto_char_lookups(err_chars[:3], rhyme_book) if bundle else []
            if char_rhymes:
                detail = "；".join(f"「{c}」属{cats}" for c, cats in char_rhymes)
                hints.append(
                    f"押韵错误：韵脚 {chars_str} 未能归入同一韵部。"
                    f"各字韵部：{detail}。"
                    f"请选定一个韵部，调用 rhyme_lookup(category=韵部名, book={rhyme_book}, limit=20) 查看可用韵字，统一所有韵脚。"
                )
            else:
                hints.append(
                    f"押韵错误：韵脚 {chars_str} 未能归入同一韵部。"
                    f"用 char_lookup(char=韵脚字, book={rhyme_book}) 确认各字韵部，再统一。"
                )

    # --- Summary ---
    total = len(errors)
    if total > 0 and not hints:
        hints.append(f"共 {total} 处错误，请逐一修改后重新调用 validate_meter 校验。")
    elif total > 0:
        hints.append(f"修改以上 {total} 处错误后，重新调用 validate_meter 校验，直到零错误。")

    return hints


def _auto_rhyme_list(book: str) -> List[Dict]:
    """Auto-call rhyme_list and return categories. Silent on failure."""
    tool = FangcunAPITool._all_tools.get("rhyme_list")
    if not tool:
        return []
    try:
        req = ToolRequest(tool_name="rhyme_list", text="", cipai="",
                          metadata={"book": book})
        resp = tool.run(req)
        results = resp.metadata.get("results", {})
        return results.get("categories", [])
    except Exception:
        return []


def _auto_rhyme_lookup(category: str, book: str, limit: int = 20) -> List[str]:
    """Auto-call rhyme_lookup and return top characters. Silent on failure."""
    tool = FangcunAPITool._all_tools.get("rhyme_lookup")
    if not tool:
        return []
    try:
        req = ToolRequest(tool_name="rhyme_lookup", text="", cipai="",
                          metadata={"category": category, "book": book, "limit": limit})
        resp = tool.run(req)
        results = resp.metadata.get("results", {})
        return results.get("characters", [])[:limit]
    except Exception:
        return []


def _auto_tone_suggestions(errors: List[Dict], text_chars: List[str]) -> List[str]:
    """For tone errors, bundle dictionary_search suggestions.

    For position b with error "应为P实为Z" in context ...a-b-c...:
      - dictionary_search(term=a, mode=head, tone=P, length=2, limit=10) → 2-char words starting with a, ending P
      - dictionary_search(term=c, mode=tail, tone=P, length=2, limit=10) → 2-char words ending with c, starting P
    """
    tool = FangcunAPITool._all_tools.get("dictionary_search")
    if not tool or not text_chars:
        return []

    suggestions = []
    for e in errors[:3]:  # limit to first 3 to avoid too many API calls
        pos = e.get("position")
        char = e.get("character", "?")
        msg = e.get("message", "")
        if pos is None or not isinstance(pos, int):
            continue

        target_tone = "P" if "应为平" in msg else "Z"
        prev_char = text_chars[pos - 1] if pos > 0 and pos - 1 < len(text_chars) else None
        next_char = text_chars[pos + 1] if pos + 1 < len(text_chars) else None

        candidates = []

        if prev_char:
            words = _auto_dict_search(prev_char, mode="head", tone=target_tone, length="2", limit=10)
            if words:
                candidates.append(f"「{prev_char}」开头的{'平' if target_tone == 'P' else '仄'}声词：{'、'.join(words)}")

        if next_char:
            words = _auto_dict_search(next_char, mode="tail", tone=target_tone, length="2", limit=10)
            if words:
                candidates.append(f"「{next_char}」结尾的{'平' if target_tone == 'P' else '仄'}声词：{'、'.join(words)}")

        if candidates:
            suggestions.append(f"位置{pos}「{char}」→需{'平' if target_tone == 'P' else '仄'}：" + "；".join(candidates))

    return suggestions


def _auto_dict_search(term: str, mode: str, tone: str, length: str = "2", limit: int = 10) -> List[str]:
    """Auto-call dictionary_search, return word list. Silent on failure."""
    tool = FangcunAPITool._all_tools.get("dictionary_search")
    if not tool:
        return []
    try:
        req = ToolRequest(tool_name="dictionary_search", text="", cipai="",
                          metadata={"term": term, "mode": mode, "tone": tone, "length": length, "limit": limit})
        resp = tool.run(req)
        results = resp.metadata.get("results", [])
        if isinstance(results, list):
            return [item[0] if isinstance(item, list) else str(item) for item in results[:limit]]
        return []
    except Exception:
        return []
    """Auto-call rhyme_lookup and return top characters. Silent on failure."""
    tool = FangcunAPITool._all_tools.get("rhyme_lookup")
    if not tool:
        return []
    try:
        req = ToolRequest(tool_name="rhyme_lookup", text="", cipai="",
                          metadata={"category": category, "book": book, "limit": limit})
        resp = tool.run(req)
        results = resp.metadata.get("results", {})
        return results.get("characters", [])[:limit]
    except Exception:
        return []


def _auto_char_lookups(chars: List[str], book: str) -> List[tuple]:
    """Auto-call char_lookup for each char, return [(char, category_str), ...]. Silent on failure."""
    tool = FangcunAPITool._all_tools.get("char_lookup")
    if not tool:
        return []
    results = []
    for ch in chars:
        try:
            req = ToolRequest(tool_name="char_lookup", text="", cipai="",
                              metadata={"char": ch, "book": book})
            resp = tool.run(req)
            data = resp.metadata.get("results", {})
            cats = data.get("rhyme_categories", [])
            if isinstance(cats, list):
                cat_names = [c.get("name", c) if isinstance(c, dict) else str(c) for c in cats]
            elif isinstance(cats, dict):
                cat_names = []
                for bk, names in cats.items():
                    if bk == book:
                        cat_names = names if isinstance(names, list) else [str(names)]
            else:
                cat_names = []
            if cat_names:
                results.append((ch, "、".join(cat_names[:3])))
        except Exception:
            pass
    return results


def build_fangcun_api_tools(config_path: str) -> List[FangcunAPITool]:
    """Build FangcunAPITool instances from configs/tools.fangcun_api.json."""
    from pathlib import Path
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)

    base_urls = config.get("base_urls", {})
    if not base_urls:
        base_url = config.get("base_url", os.environ.get("FANGCUN_WRITE_URL", "http://localhost:8901"))
        base_urls = {"write": base_url}

    limiters: Dict[str, RateLimiter] = {}
    for key, rpm in DEFAULT_RATE_LIMITS.items():
        limiters[key] = RateLimiter(rpm, 60.0)

    tools: List[FangcunAPITool] = []
    for tool_def in config.get("tools", []):
        name = tool_def["name"]
        base_key = tool_def.get("base", "write")
        base_url = base_urls.get(base_key, base_urls.get("write", ""))
        limiter = limiters.get(name, limiters.get("_default"))
        tools.append(FangcunAPITool(
            name=name,
            base_url=base_url,
            path=tool_def["path"],
            method=tool_def.get("method", "GET"),
            description=tool_def.get("description", ""),
            input_schema=tool_def.get("input_schema"),
            rate_limiter=limiter,
        ))
    FangcunAPITool._all_tools = {t.name: t for t in tools}
    return tools
