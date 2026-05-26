"""Local poetics tools backed by extracted rule and lexicon JSON data.

The tools in this module are in-process implementations. They use local rule
and lexicon data plus locally implemented checking logic; they do not import or
run any upstream web app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..schemas import TextSpan, ToolRequest, ToolResponse, issue
from .base import Tool


PLACEHOLDER = "\u25a1"
JUDOU_BREAK_CHARS = set("，,。.!！？?；;：:\n\r、")
LOCAL_POETICS_SOURCE = "poetics_data"
DEFAULT_LOCAL_POETICS_CONFIG_DIR = Path(__file__).resolve().parents[3] / "data" / "fangcun" / "config"
LOCAL_POETICS_BASE_TOOL_NAMES = [
    "get_rule",
    "check_judou",
    "check_prosody",
]
LOCAL_POETICS_EXTRA_TOOL_NAMES = [
    "prosody",
    "rule_lookup",
]
LOCAL_POETICS_LEXICAL_TOOL_NAMES = [
    "char_lookup",
    "rhyme_lookup",
    "rhyme_list",
    "phrase_suggest",
    "allusion_search",
]
FANGCUN_AUXILIARY_TOOL_NAMES = LOCAL_POETICS_EXTRA_TOOL_NAMES[1:] + LOCAL_POETICS_LEXICAL_TOOL_NAMES
FANGCUN_BASE_TOOL_NAMES = LOCAL_POETICS_BASE_TOOL_NAMES


class FangcunDataClient:
    """Small local data client for extracted poetics JSON files."""

    def __init__(self, source_path: Optional[str] = None, config_dir: Optional[str] = None):
        self.source_path = str(Path(source_path).expanduser().resolve()) if source_path else ""
        self.config_dir = str(_resolve_config_dir(source_path=source_path, config_dir=config_dir))
        self._json_cache: Dict[str, Any] = {}

    def load_json(self, filename: str, default: Any = None) -> Any:
        if filename not in self._json_cache:
            path = Path(self.config_dir) / filename
            if not path.exists():
                fallback = Path(self.source_path) / "data" / "db_migrated" / filename
                path = fallback if fallback.exists() else path
            try:
                with path.open("r", encoding="utf-8") as handle:
                    self._json_cache[filename] = json.load(handle)
            except FileNotFoundError:
                self._json_cache[filename] = {} if default is None else default
        return self._json_cache[filename]

    def t2s_text(self, text: str) -> str:
        mapping = self.load_json("t2s_map.json", {})
        return "".join(mapping.get(char, char) for char in text)

    def lookup_chars(
        self,
        text: str,
        book: Optional[str] = None,
        include_definitions: bool = False,
    ) -> List[Dict[str, Any]]:
        chars = _unique_cjk_or_placeholder(self.t2s_text(text))
        char_dict = self.load_json("char_dict.json", {})
        rhyme_books = self.load_json("rhyme_books.json", {})
        definitions = self.load_json("char_definitions.json", {}) if include_definitions else {}
        results = []
        for char in chars:
            data = dict(char_dict.get(char) or {})
            rhymes = data.get("rhymes", {}) or {}
            payload: Dict[str, Any] = {
                "char": char,
                "tones": data.get("tones", []),
            }
            if book:
                categories = []
                book_data = rhyme_books.get(book, {})
                book_categories = book_data.get("categories", {}) if isinstance(book_data, dict) else {}
                for name in rhymes.get(book, []) or []:
                    category = book_categories.get(name, {})
                    categories.append(
                        {
                            "name": name,
                            "tone_type": category.get("tone_type"),
                        }
                    )
                payload["rhyme_categories"] = categories
            else:
                payload["rhyme_categories"] = rhymes
            if include_definitions:
                payload["definitions"] = definitions.get(char, [])
            results.append(payload)
        return results

    def rhyme_lookup(
        self,
        book: str,
        category: str,
        include: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        rhyme_books = self.load_json("rhyme_books.json", {})
        book_data = rhyme_books.get(book)
        if not book_data:
            return None
        categories = book_data.get("categories", {})
        primary = self._rhyme_category_payload(categories.get(category), limit=limit)
        if primary is None:
            return None
        include_items = [item for item in (include or []) if item]
        if not include_items:
            return primary
        related = []
        for relation in include_items:
            for related_name in primary.get("relations", {}).get(relation, []):
                related_payload = self._rhyme_category_payload(categories.get(related_name), limit=limit)
                if related_payload is not None:
                    related.append({"relation": relation, "category": related_payload})
        return {"primary": primary, "related": related}

    def rhyme_list(self, book: str, tone: Optional[str] = None, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        rhyme_books = self.load_json("rhyme_books.json", {})
        book_data = rhyme_books.get(book)
        if not book_data:
            return None
        categories = []
        for category in book_data.get("categories", {}).values():
            if tone and category.get("tone_type") != tone:
                continue
            chars = list(category.get("characters", []) or [])
            categories.append(
                {
                    "name": category.get("name"),
                    "tone_type": category.get("tone_type"),
                    "char_count": len(chars),
                    "preview": "".join(chars[:10]),
                }
            )
        if limit is not None:
            categories = categories[: max(limit, 0)]
        return {"book": book, "categories": categories}

    def rule_lookup(
        self,
        genre: str = "Ci",
        search: str = "",
        char_count: Optional[int] = None,
        limit: int = 100,
        include_pattern: bool = False,
        include_constraints: bool = True,
    ) -> List[Dict[str, Any]]:
        filename = "shi_rules.json" if genre == "Shi" else "ci_rules.json"
        rules = self.load_json(filename, [])
        query = self.t2s_text(search).strip()
        results = []
        for rule in rules:
            if char_count is not None and int(rule.get("char_count", -1)) != char_count:
                continue
            haystack = f"{rule.get('name', '')} {rule.get('cipai', '')}"
            if query and query not in haystack:
                continue
            payload = {
                "name": rule.get("name"),
                "genre": rule.get("genre"),
                "cipai": rule.get("cipai"),
                "char_count": rule.get("char_count"),
            }
            if include_pattern:
                payload["tone_pattern"] = rule.get("tone_pattern")
                payload["rhyme_rule"] = rule.get("rhyme_rule")
            if include_constraints:
                payload["constraints"] = _compact_rule_constraints(rule)
            results.append(payload)
        if query:
            results.sort(key=lambda item: _rule_match_score(item, query))
        return results[: max(limit, 0)]

    def phrase_suggest(
        self,
        term: str,
        mode: str = "head",
        length: str = "2",
        tone: str = "all",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        normalized = self.t2s_text(term)
        if mode == "pair":
            return _score_rows(self.load_json("phrase_pairs.json", {}).get(normalized, []), "term", limit)
        if mode == "tongwei":
            return _score_rows(self._lookup_tongwei(normalized), "term", limit)
        if mode not in {"head", "tail"}:
            return []

        filename = "phrase_head.json" if mode == "head" else "phrase_tail.json"
        entry = self.load_json(filename, {}).get(normalized, {})
        rows: List[Dict[str, Any]] = []
        if length == "all":
            for length_key, tone_data in entry.items():
                rows.extend(_phrase_rows_from_tone_data(tone_data, length_key, tone))
        else:
            rows.extend(_phrase_rows_from_tone_data(entry.get(str(length), {}), str(length), tone))
        rows.sort(key=lambda row: row.get("score", 0), reverse=True)
        return rows[: max(limit, 0)]

    def allusion_search(self, term: str, limit: int = 60) -> List[Dict[str, Any]]:
        normalized = self.t2s_text(term)
        if not normalized:
            return []
        index = self.load_json("allusion_index.json", {})
        entries = self.load_json("allusion_entries.json", [])
        if not index or not entries:
            return []
        if len(normalized) == 1:
            ids = index.get(normalized, [])
            results = [entries[item] for item in ids if isinstance(item, int) and item < len(entries)]
        else:
            char_sets = [set(index.get(char, [])) for char in normalized]
            if not all(char_sets):
                return []
            common = char_sets[0]
            for item in char_sets[1:]:
                common &= item
            results = [
                entries[item]
                for item in common
                if isinstance(item, int) and item < len(entries) and normalized in entries[item].get("w", "")
            ]
        return _sort_allusions(results, normalized)[: max(limit, 0)]

    def _rhyme_category_payload(self, category: Optional[Dict[str, Any]], limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not category:
            return None
        characters = list(category.get("characters", []) or [])
        if limit is not None:
            characters = characters[: max(limit, 0)]
        return {
            "category_name": category.get("name"),
            "tone_type": category.get("tone_type"),
            "total": len(category.get("characters", []) or []),
            "characters": characters,
            "relations": category.get("relations", {}) or {},
        }

    def _lookup_tongwei(self, term: str) -> List[List[Any]]:
        tongwei = self.load_json("phrase_tongwei.json", {})
        if len(term) == 1:
            return tongwei.get(term, [])
        pairs = self.load_json("phrase_pairs.json", {})
        my_pairs = pairs.get(term, [])
        if not my_pairs:
            return []
        pair_set = {row[0] for row in my_pairs}
        scores: Dict[str, int] = {}
        for partner in pair_set:
            for sibling, _ in pairs.get(partner, []):
                if sibling != term:
                    scores[sibling] = scores.get(sibling, 0) + 1
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [[term, score] for term, score in top if score >= 2][:50]


@dataclass
class FangcunRule:
    name: str
    genre: str
    cipai: Optional[str]
    char_count: int
    tone_pattern: List[Any] = field(default_factory=list)
    rhyme_rule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FangcunError:
    position: int
    character: str
    error_type: str
    message: str


@dataclass
class FangcunWarning:
    positions: List[int]
    character: str
    warning_type: str
    message: str


@dataclass
class FangcunDisplaySegment:
    text_chars: List[str]
    rule_items: List[Dict[str, Any]]
    start_index: int


@dataclass
class FangcunCheckResult:
    is_valid: bool
    closest_rule: Optional[FangcunRule]
    errors: List[FangcunError]
    cleaned_chars: List[str]
    display_segments: List[FangcunDisplaySegment] = field(default_factory=list)
    warnings: List[FangcunWarning] = field(default_factory=list)
    rhyme_name: Optional[str] = None
    rhyme_positions: List[int] = field(default_factory=list)
    rhyme_chars: List[str] = field(default_factory=list)


class FangcunMeterChecker:
    """Data-driven meter checker for local rule JSON."""

    def __init__(self, client: FangcunDataClient):
        self.client = client
        self.char_dict = client.load_json("char_dict.json", {})
        self.rhyme_books = client.load_json("rhyme_books.json", {})
        self.rule_index_by_length: Dict[str, Dict[int, List[FangcunRule]]] = {"Shi": {}, "Ci": {}}
        self._all_rules_by_genre: Dict[str, List[FangcunRule]] = {"Shi": [], "Ci": []}
        self._load_rules("Shi", "shi_rules.json")
        self._load_rules("Ci", "ci_rules.json")

    def _load_rules(self, genre: str, filename: str) -> None:
        for raw in self.client.load_json(filename, []):
            rule = FangcunRule(
                name=str(raw.get("name", "")),
                genre=str(raw.get("genre", genre)),
                cipai=raw.get("cipai"),
                char_count=int(raw.get("char_count", 0)),
                tone_pattern=list(raw.get("tone_pattern", []) or []),
                rhyme_rule=dict(raw.get("rhyme_rule", {}) or {}),
            )
            self._all_rules_by_genre.setdefault(genre, []).append(rule)
            self.rule_index_by_length.setdefault(genre, {}).setdefault(rule.char_count, []).append(rule)

    def check_auto(
        self,
        poem_text: str,
        genre: str,
        rhyme_book_name: str,
        ensure_longpu: bool = False,
        rule_name: Optional[str] = None,
    ) -> FangcunCheckResult:
        chars = [char for char in poem_text if "\u4e00" <= char <= "\u9fff" or char == PLACEHOLDER]
        count = len(chars)

        if genre not in self.rule_index_by_length:
            return self._error_result(chars, "Config", f"未知的体裁: {genre}")

        if rule_name:
            all_rules = self._all_rules_by_genre.get(genre, [])
            exact = [rule for rule in all_rules if rule.name == rule_name]
            if exact:
                candidates = exact
            else:
                candidates = [rule for rule in all_rules if rule.name.startswith(rule_name) or rule.cipai == rule_name]
                if ensure_longpu:
                    longpu = [rule for rule in candidates if "龙谱" in rule.name]
                    if longpu:
                        candidates = longpu
            if not candidates:
                return self._error_result(chars, "Config", f"未找到匹配的规则: {rule_name}")
            matching_count = [rule for rule in candidates if rule.char_count == count]
            if matching_count:
                candidates = matching_count
            else:
                expected = sorted({rule.char_count for rule in candidates})
                return self._error_result(chars, "Length", f"字数不匹配: 找到 {count} 个汉字, 但 {rule_name} 要求 {expected} 字。")
        else:
            candidates = list(self.rule_index_by_length.get(genre, {}).get(count, []))
            if ensure_longpu:
                candidates = [rule for rule in candidates if "龙谱" in rule.name]
            if not candidates:
                return self._error_result(chars, "Length", f"字数不匹配: 找到 {count} 个汉字, 但未找到适用于此字数的 {genre} 规则。")

        book = self.rhyme_books.get(rhyme_book_name)
        if not book:
            return self._error_result(chars, "Config", f"未找到韵书: {rhyme_book_name}")

        best_rule: Optional[FangcunRule] = None
        best_errors: List[FangcunError] = []
        best_flat_rule: List[Dict[str, Any]] = []
        best_error_count: Optional[int] = None

        for rule in candidates:
            tone_errors, flat_rule = self._check_tone(chars, rule.tone_pattern, rhyme_book_name)
            rhyme_errors = self._check_rhyme(chars, rule.rhyme_rule, book)
            errors = tone_errors + rhyme_errors
            if best_error_count is None or len(errors) < best_error_count:
                best_rule = rule
                best_errors = errors
                best_flat_rule = flat_rule
                best_error_count = len(errors)

        if best_rule is None:
            return self._error_result(chars, "Config", "未找到可用规则。")

        rhyme_positions = sorted(_get_break_positions(best_rule.rhyme_rule))
        return FangcunCheckResult(
            is_valid=not best_errors,
            closest_rule=best_rule,
            errors=best_errors,
            cleaned_chars=chars,
            display_segments=_segment_for_display(chars, best_flat_rule, best_rule),
            warnings=_duplicate_char_warnings(chars),
            rhyme_name=self._find_one_rhyme(best_rule, chars, book),
            rhyme_positions=rhyme_positions,
            rhyme_chars=[chars[index] for index in rhyme_positions if 0 <= index < len(chars)],
        )

    def _error_result(self, chars: List[str], error_type: str, message: str) -> FangcunCheckResult:
        return FangcunCheckResult(
            is_valid=False,
            closest_rule=None,
            errors=[FangcunError(-1, "", error_type, message)],
            cleaned_chars=chars,
        )

    def _lookup_char(self, char: str) -> Optional[Dict[str, Any]]:
        data = self.char_dict.get(char)
        if data:
            return data
        simplified = self.client.t2s_text(char)
        if simplified != char:
            return self.char_dict.get(simplified)
        return None

    def _char_tones(self, char: str, rhyme_book_name: Optional[str] = None) -> set:
        data = self._lookup_char(char)
        if not data:
            return {"A"}
        if rhyme_book_name and rhyme_book_name in self.rhyme_books:
            rhymes = data.get("rhymes", {}).get(rhyme_book_name, [])
            if not rhymes:
                return {"A"}
            categories = self.rhyme_books[rhyme_book_name].get("categories", {})
            mapped = {
                categories.get(rhyme_name, {}).get("tone_type")
                for rhyme_name in rhymes
                if categories.get(rhyme_name, {}).get("tone_type")
            }
            if not mapped or len(mapped) > 1:
                return {"A"}
            return mapped
        mapped = {"P" if tone in {"Ping", "ping", "平", "yinping", "阴平", "yangping", "阳平"} else "Z" for tone in data.get("tones", [])}
        if not mapped or len(mapped) > 1:
            return {"A"}
        return mapped

    def _char_rhymes(self, char: str, book_name: str) -> set:
        data = self._lookup_char(char)
        if not data:
            return set()
        return set(data.get("rhymes", {}).get(book_name, []))

    def _check_tone(
        self,
        chars: List[str],
        tone_pattern: List[Any],
        rhyme_book_name: Optional[str] = None,
    ) -> Tuple[List[FangcunError], List[Dict[str, Any]]]:
        errors: List[FangcunError] = []
        flat_rule: List[Dict[str, Any]] = []
        char_index = 0
        pattern_index = 0
        while char_index < len(chars) and pattern_index < len(tone_pattern):
            item = tone_pattern[pattern_index]
            if isinstance(item, dict):
                rule_item = _normal_rule_item(item)
                flat_rule.append(rule_item)
                error = self._tone_error(chars[char_index], rule_item, char_index, rhyme_book_name)
                if error:
                    errors.append(error)
                char_index += 1
                pattern_index += 1
                continue
            if isinstance(item, list) and item:
                variant_len = len(item[0])
                segment = chars[char_index : char_index + variant_len]
                if len(segment) < variant_len:
                    errors.append(FangcunError(char_index, "".join(segment), "Tone", f"需要 {variant_len} 字的变体, 但诗句结束"))
                    flat_rule.extend([{"tone": "?", "comment": None}] * variant_len)
                    char_index += variant_len
                    pattern_index += 1
                    continue
                best_variant = [_normal_rule_item(rule_item) for rule_item in item[0]]
                best_errors: List[FangcunError] = []
                best_error_count: Optional[int] = None
                for variant in item:
                    normalized = [_normal_rule_item(rule_item) for rule_item in variant]
                    variant_errors = [
                        error
                        for offset, rule_item in enumerate(normalized)
                        for error in [self._tone_error(segment[offset], rule_item, char_index + offset, rhyme_book_name)]
                        if error
                    ]
                    if best_error_count is None or len(variant_errors) < best_error_count:
                        best_variant = normalized
                        best_errors = variant_errors
                        best_error_count = len(variant_errors)
                    if not variant_errors:
                        break
                errors.extend(best_errors)
                flat_rule.extend(best_variant)
                char_index += variant_len
                pattern_index += 1
                continue
            pattern_index += 1
        if len(flat_rule) < len(chars):
            flat_rule.extend([{"tone": "?", "comment": None}] * (len(chars) - len(flat_rule)))
        return errors, flat_rule

    def _tone_error(
        self,
        char: str,
        rule_item: Dict[str, Any],
        position: int,
        rhyme_book_name: Optional[str],
    ) -> Optional[FangcunError]:
        rule_tone = rule_item.get("tone", "A")
        if char == PLACEHOLDER or rule_tone == "A":
            return None
        actual_tones = self._char_tones(char, rhyme_book_name)
        if "A" in actual_tones or rule_tone in actual_tones:
            return None
        actual = _tone_label(next(iter(actual_tones)))
        return FangcunError(position, char, "Tone", f"应为{_tone_label(rule_tone)}, 实为{actual}")

    def _check_rhyme(self, chars: List[str], rule_node: Dict[str, Any], book: Dict[str, Any]) -> List[FangcunError]:
        node_type = rule_node.get("type")
        if not node_type:
            return []
        if node_type == "OR":
            branch_errors = [self._check_rhyme(chars, sub_rule, book) for sub_rule in rule_node.get("rules", [])]
            for errors in branch_errors:
                if not errors:
                    return []
            return min(branch_errors, key=len, default=[])
        if node_type == "AND":
            errors: List[FangcunError] = []
            for sub_rule in rule_node.get("rules", []):
                errors.extend(self._check_rhyme(chars, sub_rule, book))
            return errors
        if node_type == "SAME_CATEGORY":
            return self._check_same_category(chars, rule_node.get("positions", []), book)
        if node_type == "RELATION":
            return self._check_relation(
                chars,
                _int_or_default(rule_node.get("pos1"), -1),
                _int_or_default(rule_node.get("pos2"), -1),
                str(rule_node.get("relation", "")),
                book,
            )
        return []

    def _check_same_category(self, chars: List[str], positions: List[int], book: Dict[str, Any]) -> List[FangcunError]:
        valid_positions = [
            position
            for position in positions
            if isinstance(position, int) and 0 <= position < len(chars) and chars[position] != PLACEHOLDER
        ]
        if not valid_positions:
            return []
        book_name = book.get("name", "")
        first_pos = valid_positions[0]
        common = self._char_rhymes(chars[first_pos], book_name)
        if not common:
            return [FangcunError(first_pos, chars[first_pos], "Rhyme", "押韵字无韵部")]
        for position in valid_positions[1:]:
            rhymes = self._char_rhymes(chars[position], book_name)
            if not rhymes:
                return [FangcunError(position, chars[position], "Rhyme", "押韵字无韵部")]
            common &= rhymes
            if not common:
                return [FangcunError(position, chars[position], "Rhyme", "出韵")]
        return []

    def _check_relation(self, chars: List[str], pos1: int, pos2: int, relation: str, book: Dict[str, Any]) -> List[FangcunError]:
        if pos1 >= len(chars) or pos2 >= len(chars) or pos1 < 0 or pos2 < 0:
            return []
        if chars[pos1] == PLACEHOLDER or chars[pos2] == PLACEHOLDER:
            return []
        book_name = book.get("name", "")
        rhymes1 = self._char_rhymes(chars[pos1], book_name)
        rhymes2 = self._char_rhymes(chars[pos2], book_name)
        if not rhymes1 or not rhymes2:
            return [FangcunError(pos1, chars[pos1], "Rhyme", "相关字无韵部")]
        categories = book.get("categories", {})
        for rhyme_name in rhymes1:
            related = set(categories.get(rhyme_name, {}).get("relations", {}).get(relation, []))
            if related & rhymes2:
                return []
        return [FangcunError(pos1, chars[pos1], "Rhyme", f"其韵部与 {chars[pos2]} (位于{pos2}) 无法构成 '{relation}' 关系")]

    def _find_one_rhyme(self, rule: FangcunRule, chars: List[str], book: Dict[str, Any]) -> Optional[str]:
        positions = sorted(_get_rhyme_positions(rule.rhyme_rule))
        valid_positions = [position for position in positions if position < len(chars) and chars[position] != PLACEHOLDER]
        if not valid_positions:
            return None
        rule_node = rule.rhyme_rule
        if rule_node.get("type") == "AND":
            same_category_count = sum(1 for sub_rule in rule_node.get("rules", []) if sub_rule.get("type") == "SAME_CATEGORY")
            if same_category_count > 1:
                return " (换韵)"
        inference_positions = valid_positions[1:] if rule.genre == "Shi" and rule_node.get("type") == "OR" and len(valid_positions) > 1 else valid_positions
        if not inference_positions:
            return None
        book_name = book.get("name", "")
        common = self._char_rhymes(chars[inference_positions[0]], book_name)
        for position in inference_positions[1:]:
            common &= self._char_rhymes(chars[position], book_name)
            if not common:
                return None
        return ", ".join(sorted(common)) if common else None


class FangcunProsodyTool(Tool):
    name = "prosody"
    description = (
        "Extra validation tool for ci meter and rhyme using local poetics rule data. "
        "Returned issues include exact 1-based character positions, source line/column, "
        "the offending character, and the expected/actual tone or rhyme message."
    )
    tool_group = "extra_tool"
    usage_stage = "validation"
    model_exposure = "extra"
    tags = ("prosody", "meter", "rhyme", "overlaps_base")

    def __init__(
        self,
        source_path: Optional[str] = None,
        config_dir: Optional[str] = None,
        default_genre: str = "Ci",
        default_rhyme_book: str = "Cilinzhengyun",
        default_ensure_longpu: bool = True,
        client: Optional[FangcunDataClient] = None,
    ):
        self.source_path = str(Path(source_path).expanduser().resolve()) if source_path else ""
        self.config_dir = str(_resolve_config_dir(source_path=source_path, config_dir=config_dir))
        self.default_genre = default_genre
        self.default_rhyme_book = default_rhyme_book
        self.default_ensure_longpu = default_ensure_longpu
        self.client = client or FangcunDataClient(source_path=source_path, config_dir=self.config_dir)
        self._checker: Any = None

    def input_schema(self) -> Dict[str, Any]:
        schema = super().input_schema()
        schema["properties"]["cipai"][
            "description"
        ] = "Cipai/rule name, e.g. 卜算子. Can be overridden by metadata.rule_name."
        schema["properties"]["metadata"] = {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "enum": ["Shi", "Ci"]},
                "rhyme_book_name": {"type": "string"},
                "rule_name": {"type": "string"},
                "ensure_longpu": {"type": "boolean"},
                "compact": {"type": "boolean"},
            },
        }
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        genre = str(metadata.get("genre") or self.default_genre)
        rhyme_book_name = str(
            metadata.get("rhyme_book_name")
            or metadata.get("rhyme_book")
            or self.default_rhyme_book
        )
        rule_name = metadata.get("rule_name") or request.cipai or None
        ensure_longpu = _as_bool(metadata.get("ensure_longpu"), self.default_ensure_longpu)
        compact = _as_bool(metadata.get("compact"), False)

        try:
            result = self._get_checker().check_auto(
                request.text,
                genre,
                rhyme_book_name,
                ensure_longpu,
                rule_name,
            )
        except Exception as exc:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[
                    issue(
                        "poetics_runtime_error",
                        f"Local poetics checker failed: {exc}",
                        source_path=self.source_path,
                    )
                ],
                metrics={"violation_count": 1},
                metadata={"source": LOCAL_POETICS_SOURCE, "genre": genre, "rhyme_book_name": rhyme_book_name},
            )

        position_map = _cleaned_position_map(request.text)
        issues = [
            _error_to_issue(error, position_map)
            for error in getattr(result, "errors", [])
        ]
        issue_summary = [_issue_summary(item) for item in issues]
        closest_rule = getattr(result, "closest_rule", None)
        checked_chars = len(getattr(result, "cleaned_chars", []) or [])

        response_metadata = {
            "source": LOCAL_POETICS_SOURCE,
            "genre": genre,
            "rhyme_book_name": rhyme_book_name,
            "rule_name": rule_name,
            "ensure_longpu": ensure_longpu,
            "closest_rule": _rule_to_dict(closest_rule),
            "rhyme_name": getattr(result, "rhyme_name", None),
            "rhyme_positions": getattr(result, "rhyme_positions", None),
            "rhyme_chars": getattr(result, "rhyme_chars", None),
            "issue_summary": issue_summary,
            "warnings": [
                _warning_to_dict(warning, position_map)
                for warning in (getattr(result, "warnings", []) or [])
            ],
        }
        if compact:
            response_metadata["compact"] = True
            response_metadata["warning_summary"] = {
                "count": len(response_metadata["warnings"]),
                "characters": [
                    warning.get("character")
                    for warning in response_metadata["warnings"][:8]
                    if warning.get("character")
                ],
            }
            response_metadata.pop("warnings", None)
        else:
            response_metadata["display_segments"] = [
                _display_segment_to_dict(segment)
                for segment in (getattr(result, "display_segments", []) or [])
            ]

        return ToolResponse(
            self.name,
            passed=bool(getattr(result, "is_valid", False)),
            issues=issues,
            metrics={
                "violation_count": len(issues),
                "warning_count": len(getattr(result, "warnings", []) or []),
                "checked_chars": checked_chars,
                "expected_chars": getattr(closest_rule, "char_count", None),
                "rhyme_position_count": len(getattr(result, "rhyme_positions", []) or []),
            },
            metadata=response_metadata,
        )

    def _get_checker(self) -> Any:
        if self._checker is None:
            self._checker = self._load_checker()
        return self._checker

    def _load_checker(self) -> Any:
        return FangcunMeterChecker(self.client)


class FangcunRuleGuideTool(Tool):
    name = "get_rule"
    description = (
        "Get a lightweight cipai rule summary. Input text must be only the cipai or rule name, "
        "such as 疏影; do not include the poem title, user prompt, or poem body."
    )
    tool_group = "base_tool"
    usage_stage = "pre_draft"
    model_exposure = "base"
    tags = ("rule", "structure", "body_only")

    def __init__(self, client: FangcunDataClient, default_genre: str = "Ci", default_ensure_longpu: bool = True):
        self.client = client
        self.default_genre = default_genre
        self.default_ensure_longpu = default_ensure_longpu

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Only the cipai/rule name, e.g. 疏影. Do not send the title, prompt, or poem body.",
                },
                "cipai": {
                    "type": "string",
                    "description": "Optional cipai/rule name. If provided, it overrides text.",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "genre": {"type": "string", "enum": ["Shi", "Ci"]},
                        "rule_name": {"type": "string"},
                        "ensure_longpu": {"type": "boolean"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        genre = str(metadata.get("genre") or self.default_genre)
        query = str(metadata.get("rule_name") or request.cipai or request.text).strip()
        ensure_longpu = _as_bool(metadata.get("ensure_longpu"), self.default_ensure_longpu)
        selected = _select_rule_dict(self.client, genre=genre, query=query, ensure_longpu=ensure_longpu)
        if selected is None:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("not_found", "No local poetics rule matched the cipai/rule name.", expected=query, actual=None)],
                metrics={"result_count": 0},
                metadata={"source": LOCAL_POETICS_SOURCE, "genre": genre, "query": query},
            )
        summary = _lightweight_rule_summary(selected)
        return ToolResponse(
            self.name,
            passed=True,
            metrics={
                "char_count": summary["char_count"],
                "sentence_count": len(summary["sentence_char_counts"]),
                "caesura_count": len(summary["caesura_1based"]),
                "rhyme_count": len(summary["rhyme_positions_1based"]),
            },
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "genre": genre,
                "query": query,
                "ensure_longpu": ensure_longpu,
                "rule": summary,
            },
        )


class FangcunJudouTool(Tool):
    name = "check_judou"
    description = (
        "Check cipai judou sentence and caesura break positions. Input text must be only the ci body "
        "with its punctuation/newlines; do not include the title, cipai label, prompt, notes, or explanation."
    )
    tool_group = "base_tool"
    usage_stage = "draft_validation"
    model_exposure = "base"
    tags = ("judou", "structure", "body_only")

    def __init__(self, client: FangcunDataClient, default_genre: str = "Ci", default_ensure_longpu: bool = True):
        self.client = client
        self.default_genre = default_genre
        self.default_ensure_longpu = default_ensure_longpu

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Only the ci body to check, including punctuation/newlines. "
                        "Do not include a title such as 《疏影》, the cipai name, prompt text, or commentary."
                    ),
                },
                "cipai": {
                    "type": "string",
                    "description": "Requested cipai/rule name, e.g. 疏影. Required unless metadata.rule_name is set.",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "genre": {"type": "string", "enum": ["Shi", "Ci"]},
                        "rule_name": {"type": "string"},
                        "ensure_longpu": {"type": "boolean"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        genre = str(metadata.get("genre") or self.default_genre)
        query = str(metadata.get("rule_name") or request.cipai or "").strip()
        ensure_longpu = _as_bool(metadata.get("ensure_longpu"), self.default_ensure_longpu)
        selected = _select_rule_dict(self.client, genre=genre, query=query, ensure_longpu=ensure_longpu)
        if selected is None:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("not_found", "No local poetics rule matched the cipai/rule name.", expected=query, actual=None)],
                metrics={"result_count": 0},
                metadata={"source": LOCAL_POETICS_SOURCE, "genre": genre, "query": query},
            )

        summary = _lightweight_rule_summary(selected)
        expected_sentence = set(summary["internal_sentence_breaks_1based"])
        expected_caesura = set(summary["caesura_1based"])
        expected_breaks = expected_sentence | expected_caesura
        char_count = _cleaned_char_count(request.text)
        actual_breaks = set(_actual_break_positions_1based(request.text))
        final_position = int(summary["char_count"])
        actual_internal_breaks = {pos for pos in actual_breaks if pos != final_position}
        missing_sentence = sorted(expected_sentence - actual_breaks)
        missing_caesura = sorted(expected_caesura - actual_breaks)
        extra_breaks = sorted(actual_internal_breaks - expected_breaks)

        issues = []
        if char_count != final_position:
            issues.append(
                issue(
                    "judou_char_count",
                    "Body CJK character count does not match the selected cipai rule. Check that the title/prompt was not included.",
                    expected=str(final_position),
                    actual=str(char_count),
                    expected_chars=final_position,
                    actual_chars=char_count,
                )
            )
        position_map = _cleaned_position_map(request.text)
        for position in missing_sentence:
            issues.append(_judou_position_issue("missing_sentence_break", position, position_map))
        for position in missing_caesura:
            issues.append(_judou_position_issue("missing_caesura", position, position_map))
        for position in extra_breaks:
            issues.append(_judou_position_issue("extra_break", position, position_map))

        return ToolResponse(
            self.name,
            passed=not issues,
            issues=issues,
            metrics={
                "violation_count": len(issues),
                "checked_chars": char_count,
                "expected_chars": final_position,
                "actual_break_count": len(actual_breaks),
                "expected_break_count": len(expected_breaks),
            },
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "genre": genre,
                "query": query,
                "ensure_longpu": ensure_longpu,
                "rule": summary,
                "actual_breaks_1based": sorted(actual_breaks),
                "missing_sentence_breaks_1based": missing_sentence,
                "missing_caesura_1based": missing_caesura,
                "extra_breaks_1based": extra_breaks,
                "issue_summary": [_issue_summary(item) for item in issues],
            },
        )


class FangcunBaseProsodyTool(Tool):
    name = "check_prosody"
    description = (
        "Check ci character count, tones, and rhymes after judou is correct. Input text must be only "
        "the exact ci body; do not include the title, cipai label, prompt, notes, or explanation."
    )
    tool_group = "base_tool"
    usage_stage = "draft_validation"
    model_exposure = "base"
    tags = ("prosody", "meter", "rhyme", "body_only")

    def __init__(self, wrapped: FangcunProsodyTool):
        self.wrapped = wrapped

    def input_schema(self) -> Dict[str, Any]:
        schema = self.wrapped.input_schema()
        schema["properties"]["text"] = {
            "type": "string",
            "description": (
                "Only the exact ci body to check. Do not include a title such as 《疏影》, "
                "the cipai name, prompt text, or commentary."
            ),
        }
        schema["properties"]["cipai"][
            "description"
        ] = "Requested cipai/rule name, e.g. 疏影. Required unless metadata.rule_name is set."
        return schema

    def run(self, request: ToolRequest) -> ToolResponse:
        response = self.wrapped.run(
            ToolRequest(
                "prosody",
                request.text,
                cipai=request.cipai,
                metadata=request.metadata,
            )
        )
        response.tool_name = self.name
        response.metadata = dict(response.metadata)
        response.metadata["base_tool"] = self.name
        return response


class FangcunCharLookupTool(Tool):
    name = "char_lookup"
    description = "Lexical tool: look up character tones, rhyme categories, and optional definitions."
    tool_group = "lexical_tool"
    usage_stage = "reference"
    model_exposure = "lexical"
    tags = ("character", "tone", "rhyme")

    def __init__(self, client: FangcunDataClient, default_book: str = "Cilinzhengyun"):
        self.client = client
        self.default_book = default_book

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "One or more Chinese characters to look up."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "book": {"type": "string"},
                        "include_definitions": {"type": "boolean"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        book = metadata.get("book") or metadata.get("rhyme_book_name") or self.default_book
        include_definitions = _as_bool(metadata.get("include_definitions"), False)
        characters = self.client.lookup_chars(request.text, book=book, include_definitions=include_definitions)
        return ToolResponse(
            self.name,
            passed=bool(characters),
            issues=[] if characters else [issue("not_found", "No CJK characters were found in the query.")],
            metrics={"result_count": len(characters)},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "query": self.client.t2s_text(request.text),
                "book": book,
                "results": characters,
                "characters": characters,
            },
        )


class FangcunRhymeLookupTool(Tool):
    name = "rhyme_lookup"
    description = "Lexical tool: look up a rhyme category and optionally related categories."
    tool_group = "lexical_tool"
    usage_stage = "reference"
    model_exposure = "lexical"
    tags = ("rhyme", "category")

    def __init__(self, client: FangcunDataClient, default_book: str = "Cilinzhengyun"):
        self.client = client
        self.default_book = default_book

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Rhyme category name, e.g. 第4部_仄 or 一东."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "book": {"type": "string"},
                        "include": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        book = metadata.get("book") or metadata.get("rhyme_book_name") or self.default_book
        category = metadata.get("category") or request.text
        include = _split_items(metadata.get("include"))
        limit = _optional_int(metadata.get("limit"))
        result = self.client.rhyme_lookup(book=book, category=str(category), include=include, limit=limit)
        if result is None:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("not_found", "Rhyme category was not found.", expected=str(category), actual=None)],
                metrics={"result_count": 0},
                metadata={
                    "source": LOCAL_POETICS_SOURCE,
                    "query": str(category),
                    "book": book,
                    "category": category,
                    "results": [],
                },
            )
        count = len(result.get("characters", [])) if "characters" in result else len(result.get("primary", {}).get("characters", []))
        return ToolResponse(
            self.name,
            passed=True,
            metrics={"result_count": count},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "query": str(category),
                "book": book,
                "category": category,
                "results": [result],
                "result": result,
            },
        )


class FangcunRhymeListTool(Tool):
    name = "rhyme_list"
    description = "Lexical tool: list rhyme categories for a rhyme book."
    tool_group = "lexical_tool"
    usage_stage = "reference"
    model_exposure = "lexical"
    tags = ("rhyme", "category_list")

    def __init__(self, client: FangcunDataClient, default_book: str = "Cilinzhengyun"):
        self.client = client
        self.default_book = default_book

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Optional tone filter: P or Z."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "book": {"type": "string"},
                        "tone": {"type": "string", "enum": ["P", "Z"]},
                        "limit": {"type": "integer"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        book = metadata.get("book") or metadata.get("rhyme_book_name") or self.default_book
        tone = metadata.get("tone") or (request.text.strip() if request.text.strip() in {"P", "Z"} else None)
        result = self.client.rhyme_list(book=book, tone=tone, limit=_optional_int(metadata.get("limit")))
        if result is None:
            return ToolResponse(
                self.name,
                passed=False,
                issues=[issue("not_found", "Rhyme book was not found.", expected=str(book), actual=None)],
                metrics={"result_count": 0},
                metadata={"source": LOCAL_POETICS_SOURCE, "query": tone, "book": book, "results": []},
            )
        return ToolResponse(
            self.name,
            passed=True,
            metrics={"result_count": len(result["categories"])},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "query": tone,
                "book": book,
                "tone": tone,
                "results": result["categories"],
                "result": result,
            },
        )


class FangcunRuleLookupTool(Tool):
    name = "rule_lookup"
    description = "Extra tool: search shi/ci rule names, cipai names, and character counts."
    tool_group = "extra_tool"
    usage_stage = "reference"
    model_exposure = "extra"
    tags = ("rule", "lookup", "overlaps_base")

    def __init__(self, client: FangcunDataClient, default_genre: str = "Ci"):
        self.client = client
        self.default_genre = default_genre

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Search query for rule name or cipai."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "genre": {"type": "string", "enum": ["Shi", "Ci"]},
                        "char_count": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "include_pattern": {"type": "boolean"},
                        "compact": {"type": "boolean"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        genre = metadata.get("genre") or self.default_genre
        limit = _int_or_default(metadata.get("limit"), 100)
        compact = _as_bool(metadata.get("compact"), True)
        results = self.client.rule_lookup(
            genre=genre,
            search=request.text,
            char_count=_optional_int(metadata.get("char_count")),
            limit=limit,
            include_pattern=_as_bool(metadata.get("include_pattern"), False) and not compact,
            include_constraints=compact,
        )
        return ToolResponse(
            self.name,
            passed=bool(results),
            issues=[] if results else [issue("not_found", "No local poetics rules matched the query.")],
            metrics={"result_count": len(results)},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "genre": genre,
                "query": request.text,
                "results": results,
                "rules": results,
            },
        )


class FangcunPhraseSuggestTool(Tool):
    name = "phrase_suggest"
    description = "Lexical tool: suggest poetic phrases, paired terms, or same-position terms."
    tool_group = "lexical_tool"
    usage_stage = "reference"
    model_exposure = "lexical"
    tags = ("phrase", "suggestion")

    def __init__(self, client: FangcunDataClient):
        self.client = client

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Anchor character or phrase."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["head", "tail", "pair", "tongwei"]},
                        "length": {"type": "string"},
                        "tone": {"type": "string", "enum": ["P", "Z", "all"]},
                        "limit": {"type": "integer"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        metadata = request.metadata or {}
        mode = metadata.get("mode") or "head"
        length = str(metadata.get("length", "2"))
        tone = metadata.get("tone") or "all"
        limit = _int_or_default(metadata.get("limit"), 50)
        suggestions = self.client.phrase_suggest(
            term=request.text,
            mode=mode,
            length=length,
            tone=tone,
            limit=limit,
        )
        return ToolResponse(
            self.name,
            passed=bool(suggestions),
            issues=[] if suggestions else [issue("not_found", "No phrase suggestions matched the query.")],
            metrics={"result_count": len(suggestions)},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "query": self.client.t2s_text(request.text),
                "term": self.client.t2s_text(request.text),
                "mode": mode,
                "length": length,
                "tone": tone,
                "results": suggestions,
                "suggestions": suggestions,
            },
        )


class FangcunAllusionSearchTool(Tool):
    name = "allusion_search"
    description = "Lexical tool: search allusion entries by keyword."
    tool_group = "lexical_tool"
    usage_stage = "reference"
    model_exposure = "lexical"
    tags = ("allusion", "phrase")

    def __init__(self, client: FangcunDataClient):
        self.client = client

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Allusion keyword."},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                    },
                },
            },
            "required": ["text"],
        }

    def run(self, request: ToolRequest) -> ToolResponse:
        limit = _int_or_default((request.metadata or {}).get("limit"), 20)
        entries = self.client.allusion_search(request.text, limit=limit)
        return ToolResponse(
            self.name,
            passed=bool(entries),
            issues=[] if entries else [issue("not_found", "No allusions matched the query.")],
            metrics={"result_count": len(entries)},
            metadata={
                "source": LOCAL_POETICS_SOURCE,
                "query": self.client.t2s_text(request.text),
                "term": self.client.t2s_text(request.text),
                "results": entries,
                "entries": entries,
            },
        )


def _resolve_config_dir(source_path: Optional[str] = None, config_dir: Optional[str] = None) -> Path:
    if config_dir:
        return Path(config_dir).expanduser().resolve()
    if source_path:
        source = Path(source_path).expanduser().resolve()
        if (source / "static" / "config").exists():
            return source / "static" / "config"
        if (source / "config").exists():
            return source / "config"
        return source
    return DEFAULT_LOCAL_POETICS_CONFIG_DIR


def _unique_cjk_or_placeholder(text: str) -> List[str]:
    chars = []
    seen = set()
    for char in text:
        if not ("\u4e00" <= char <= "\u9fff" or char == PLACEHOLDER):
            continue
        if char not in seen:
            seen.add(char)
            chars.append(char)
    return chars


def _normal_rule_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return {"tone": item.get("tone", "?"), "comment": item.get("comment")}
    return {"tone": "?", "comment": None}


def _tone_label(tone: str) -> str:
    return {"P": "平", "Z": "仄", "A": "中", "?": "?"}.get(tone, "?")


def _get_rhyme_positions(rule_node: Dict[str, Any]) -> set:
    node_type = rule_node.get("type")
    if not node_type:
        return set()
    if node_type == "SAME_CATEGORY":
        return _position_set(rule_node.get("positions", []))
    if node_type in {"AND", "OR"}:
        positions = set()
        for sub_rule in rule_node.get("rules", []):
            positions.update(_get_rhyme_positions(sub_rule))
        return positions
    return set()


def _get_break_positions(rule_node: Dict[str, Any]) -> set:
    node_type = rule_node.get("type")
    if not node_type:
        return set()
    if node_type == "SAME_CATEGORY":
        return _position_set(rule_node.get("positions", []))
    if node_type == "RELATION":
        return _position_set([rule_node.get("pos2")])
    if node_type in {"AND", "OR"}:
        positions = set()
        for sub_rule in rule_node.get("rules", []):
            positions.update(_get_break_positions(sub_rule))
        return positions
    return set()


def _select_rule_dict(
    client: FangcunDataClient,
    genre: str,
    query: str,
    ensure_longpu: bool = True,
    char_count: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    filename = "shi_rules.json" if genre == "Shi" else "ci_rules.json"
    normalized = client.t2s_text(query).strip()
    rules = client.load_json(filename, [])
    candidates = []
    for rule in rules:
        name = str(rule.get("name") or "")
        cipai = str(rule.get("cipai") or "")
        if normalized not in {name, cipai} and not name.startswith(normalized) and not cipai.startswith(normalized):
            continue
        if char_count is not None and _optional_int(rule.get("char_count")) != char_count:
            continue
        candidates.append(dict(rule))
    if ensure_longpu:
        longpu = [rule for rule in candidates if "龙谱" in str(rule.get("name") or "")]
        if longpu:
            candidates = longpu
    candidates.sort(key=lambda item: _rule_match_score(item, normalized))
    return candidates[0] if candidates else None


def _lightweight_rule_summary(rule: Dict[str, Any]) -> Dict[str, Any]:
    char_count = _int_or_default(rule.get("char_count"), 0)
    rule_items = _flatten_tone_pattern(rule.get("tone_pattern", []) or [])
    caesura_positions = _comment_positions_1based(rule_items, "读")
    rhyme_positions = sorted(position + 1 for position in _get_rhyme_positions(rule.get("rhyme_rule", {}) or {}))
    comment_sentence_breaks = _comment_positions_1based(rule_items, "句")
    internal_sentence_breaks = sorted(
        set(comment_sentence_breaks + [position for position in rhyme_positions if position != char_count])
    )
    sentence_end_positions = sorted(set(internal_sentence_breaks + ([char_count] if char_count else [])))
    return {
        "name": rule.get("name"),
        "genre": rule.get("genre"),
        "cipai": rule.get("cipai"),
        "char_count": char_count,
        "comment_sentence_breaks_1based": comment_sentence_breaks,
        "internal_sentence_breaks_1based": internal_sentence_breaks,
        "sentence_end_positions_1based": sentence_end_positions,
        "sentence_char_counts": _segment_lengths_from_breaks(char_count, [position - 1 for position in sentence_end_positions]),
        "tone_pattern": [item.get("tone", "?") for item in rule_items],
        "caesura_1based": caesura_positions,
        "rhyme_positions_1based": rhyme_positions,
        "judou_breaks_1based": sorted(set(internal_sentence_breaks + caesura_positions)),
        "writing_note": (
            "Use only the ci body. Do not include title, cipai label, prompt, notes, or explanation. "
            "Place sentence breaks after internal_sentence_breaks_1based and light pauses after caesura_1based; "
            "the final character is an implicit sentence end."
        ),
    }


def _flatten_tone_pattern(pattern: List[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in pattern:
        if isinstance(item, dict):
            items.append(_normal_rule_item(item))
            continue
        if isinstance(item, list) and item:
            first_variant = item[0]
            if isinstance(first_variant, list):
                items.extend(_normal_rule_item(rule_item) for rule_item in first_variant)
    return items


def _comment_positions_1based(rule_items: List[Dict[str, Any]], comment: str) -> List[int]:
    return [index + 1 for index, item in enumerate(rule_items) if item.get("comment") == comment]


def _cleaned_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff" or char == PLACEHOLDER)


def _actual_break_positions_1based(text: str) -> List[int]:
    positions = []
    current_position = 0
    last_cjk_position: Optional[int] = None
    for char in text:
        if "\u4e00" <= char <= "\u9fff" or char == PLACEHOLDER:
            current_position += 1
            last_cjk_position = current_position
            continue
        if char in JUDOU_BREAK_CHARS and last_cjk_position is not None:
            if not positions or positions[-1] != last_cjk_position:
                positions.append(last_cjk_position)
    return positions


def _judou_position_issue(issue_type: str, position_1based: int, position_map: List[TextSpan]) -> Dict[str, Any]:
    span = position_map[position_1based - 1] if 0 < position_1based <= len(position_map) else None
    messages = {
        "missing_sentence_break": f"第 {position_1based} 字后应有句断。",
        "missing_caesura": f"第 {position_1based} 字后应有读断。",
        "extra_break": f"第 {position_1based} 字后出现了词谱未要求的断点。",
    }
    expected = "break" if issue_type != "extra_break" else "no break"
    actual = "no break" if issue_type != "extra_break" else "break"
    return issue(
        issue_type,
        messages.get(issue_type, "句读断点不符合词谱。"),
        span=span,
        expected=expected,
        actual=actual,
        position_1based=position_1based,
        line=span.line if span else None,
        column=span.column if span else None,
        text=span.text if span else None,
    )


def _compact_rule_constraints(rule: Dict[str, Any]) -> Dict[str, Any]:
    char_count = _int_or_default(rule.get("char_count"), 0)
    break_positions = sorted(_get_break_positions(rule.get("rhyme_rule", {}) or {}))
    segment_lengths = _segment_lengths_from_breaks(char_count, break_positions)
    summary = _lightweight_rule_summary(rule)
    return {
        "char_count": char_count,
        "rhyme_positions_1based": [position + 1 for position in break_positions],
        "rhyme_count": len(break_positions),
        "segment_char_counts": segment_lengths,
        "internal_sentence_breaks_1based": summary["internal_sentence_breaks_1based"],
        "sentence_end_positions_1based": summary["sentence_end_positions_1based"],
        "sentence_char_counts": summary["sentence_char_counts"],
        "caesura_1based": summary["caesura_1based"],
        "writing_note": (
            "Write this many CJK characters excluding punctuation. "
            "Use sentence_char_counts for judou; rhyme at the listed 1-based positions. "
            "Do not request full tone_pattern during drafting; run prosody on the complete draft."
        ),
    }


def _segment_lengths_from_breaks(char_count: int, break_positions: List[int]) -> List[int]:
    if char_count <= 0:
        return []
    lengths: List[int] = []
    start = 0
    for position in break_positions:
        if position < start or position >= char_count:
            continue
        lengths.append(position - start + 1)
        start = position + 1
    if start < char_count:
        lengths.append(char_count - start)
    return lengths


def _position_set(values: Iterable[Any]) -> set:
    positions = set()
    for value in values or []:
        parsed = _optional_int(value)
        if parsed is not None and parsed >= 0:
            positions.add(parsed)
    return positions


def _segment_for_display(chars: List[str], flat_rule: List[Dict[str, Any]], rule: FangcunRule) -> List[FangcunDisplaySegment]:
    if not chars:
        return []
    rule_items = flat_rule or [{"tone": "?", "comment": None}] * len(chars)
    segments: List[FangcunDisplaySegment] = []
    text_chars: List[str] = []
    segment_rule_items: List[Dict[str, Any]] = []
    start_index = 0
    if rule.genre == "Ci":
        break_positions = _get_break_positions(rule.rhyme_rule)
        for index, char in enumerate(chars):
            text_chars.append(char)
            segment_rule_items.append(rule_items[index] if index < len(rule_items) else {"tone": "?", "comment": None})
            if index in break_positions and index < len(chars) - 1:
                segments.append(FangcunDisplaySegment(text_chars, segment_rule_items, start_index))
                text_chars = []
                segment_rule_items = []
                start_index = index + 1
    else:
        sentence_len = 7 if rule.char_count % 7 == 0 else 5
        couplet_len = sentence_len * 2
        for index, char in enumerate(chars):
            text_chars.append(char)
            segment_rule_items.append(rule_items[index] if index < len(rule_items) else {"tone": "?", "comment": None})
            if (index + 1) % couplet_len == 0 and index < len(chars) - 1:
                segments.append(FangcunDisplaySegment(text_chars, segment_rule_items, start_index))
                text_chars = []
                segment_rule_items = []
                start_index = index + 1
    if text_chars:
        segments.append(FangcunDisplaySegment(text_chars, segment_rule_items, start_index))
    return segments


def _duplicate_char_warnings(chars: List[str]) -> List[FangcunWarning]:
    positions: Dict[str, List[int]] = {}
    for index, char in enumerate(chars):
        if char == PLACEHOLDER or not ("\u4e00" <= char <= "\u9fff"):
            continue
        positions.setdefault(char, []).append(index)
    warnings = []
    for char, char_positions in positions.items():
        if len(char_positions) < 2:
            continue
        non_overlap = []
        for position in char_positions:
            if not any(abs(other - position) == 1 for other in char_positions if other != position):
                non_overlap.append(position)
        if len(non_overlap) >= 2 or (non_overlap and len(char_positions) > len(non_overlap)):
            labels = "、".join(str(position + 1) for position in char_positions)
            warnings.append(
                FangcunWarning(
                    positions=char_positions,
                    character=char,
                    warning_type="Duplicate",
                    message=f"「{char}」在第{labels}字出现{len(char_positions)}次",
                )
            )
    return warnings


def _score_rows(rows: Iterable[Any], value_key: str, limit: int) -> List[Dict[str, Any]]:
    results = []
    for row in rows:
        if not row:
            continue
        value = row[0]
        score = row[1] if len(row) > 1 else None
        results.append({value_key: value, "score": score})
        if len(results) >= max(limit, 0):
            break
    return results


def _phrase_rows_from_tone_data(tone_data: Dict[str, Any], length: str, tone: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tone_keys = [tone] if tone in {"P", "Z"} else ["P", "Z"]
    merged: Dict[str, Dict[str, Any]] = {}
    for tone_key in tone_keys:
        for phrase, score in tone_data.get(tone_key, []) or []:
            current = merged.get(phrase)
            if current is None or score > current["score"]:
                merged[phrase] = {
                    "phrase": phrase,
                    "score": score,
                    "length": length,
                    "tone": tone_key,
                }
    rows.extend(merged.values())
    return rows


def _sort_allusions(entries: List[Dict[str, Any]], term: str) -> List[Dict[str, Any]]:
    def score(entry: Dict[str, Any]) -> Tuple[int, int, int, int]:
        word = entry.get("w", "")
        exact = 0 if word == term else 1
        position = word.find(term) if term in word else len(word)
        length_score = len(word) if len(word) <= 4 else len(word) + 2
        recency_count = -min(int(entry.get("rc", 0) or 0), 20)
        return exact, position, length_score, recency_count

    return sorted(entries, key=score)


def _rule_match_score(rule: Dict[str, Any], query: str) -> Tuple[int, int, int, str]:
    name = str(rule.get("name") or "")
    cipai = str(rule.get("cipai") or "")
    exact = 0 if query in {name, cipai} else 1
    prefix = 0 if name.startswith(query) or cipai.startswith(query) else 1
    length = min(len(name), len(cipai) if cipai else len(name))
    return exact, prefix, length, name


def _split_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _error_to_issue(error: Any, position_map: List[TextSpan]) -> Dict[str, Any]:
    position = int(getattr(error, "position", -1))
    span = position_map[position] if 0 <= position < len(position_map) else None
    error_type = str(getattr(error, "error_type", "Error"))
    span_dict = span.to_dict() if span else None
    position_1based = position + 1 if position >= 0 else None
    return issue(
        error_type.lower(),
        str(getattr(error, "message", "")),
        span=span,
        actual=getattr(error, "character", None),
        data_position=position,
        position_1based=position_1based,
        line=span_dict.get("line") if span_dict else None,
        column=span_dict.get("column") if span_dict else None,
        text=span_dict.get("text") if span_dict else None,
        rule_error_type=error_type,
    )


def _issue_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": item.get("type"),
        "position_1based": item.get("position_1based"),
        "line": item.get("line"),
        "column": item.get("column"),
        "text": item.get("text") or item.get("actual"),
        "message": item.get("message"),
    }


def _warning_to_dict(warning: Any, position_map: List[TextSpan]) -> Dict[str, Any]:
    positions = list(getattr(warning, "positions", []) or [])
    return {
        "type": getattr(warning, "warning_type", "Warning"),
        "message": getattr(warning, "message", ""),
        "character": getattr(warning, "character", ""),
        "positions": positions,
        "spans": [
            position_map[pos].to_dict()
            for pos in positions
            if isinstance(pos, int) and 0 <= pos < len(position_map)
        ],
    }


def _display_segment_to_dict(segment: Any) -> Dict[str, Any]:
    return {
        "start_index": getattr(segment, "start_index", 0),
        "text_chars": list(getattr(segment, "text_chars", []) or []),
        "rule_items": list(getattr(segment, "rule_items", []) or []),
    }


def _rule_to_dict(rule: Any) -> Optional[Dict[str, Any]]:
    if rule is None:
        return None
    return {
        "name": getattr(rule, "name", None),
        "genre": getattr(rule, "genre", None),
        "cipai": getattr(rule, "cipai", None),
        "char_count": getattr(rule, "char_count", None),
    }


def _cleaned_position_map(text: str) -> List[TextSpan]:
    spans: List[TextSpan] = []
    line = 1
    column = 1
    for index, char in enumerate(text):
        if char == "\n":
            line += 1
            column = 1
            continue
        if "\u4e00" <= char <= "\u9fff" or char == PLACEHOLDER:
            spans.append(TextSpan(start=index, end=index + 1, line=line, column=column, text=char))
        column += 1
    return spans


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)
