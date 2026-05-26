"""Text normalization helpers for ci evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .schemas import TextSpan


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(frozen=True)
class CharInfo:
    char: str
    index: int
    line: int
    column: int


def is_cjk(char: str) -> bool:
    return bool(CJK_RE.match(char))


def cjk_chars(text: str) -> List[CharInfo]:
    chars: List[CharInfo] = []
    line = 1
    column = 1
    for index, char in enumerate(text):
        if char == "\n":
            line += 1
            column = 1
            continue
        if is_cjk(char):
            chars.append(CharInfo(char=char, index=index, line=line, column=column))
        column += 1
    return chars


def count_cjk(text: str) -> int:
    return sum(1 for char in text if is_cjk(char))


def normalized_cjk(text: str) -> str:
    return "".join(info.char for info in cjk_chars(text))


def span_from_char_window(chars: List[CharInfo], start: int, length: int) -> TextSpan:
    first = chars[start]
    last = chars[start + length - 1]
    return TextSpan(
        start=first.index,
        end=last.index + 1,
        line=first.line,
        column=first.column,
        text="".join(info.char for info in chars[start : start + length]),
    )


def iter_nonempty_lines(text: str) -> Iterable[Tuple[int, str, TextSpan]]:
    offset = 0
    for line_no, raw in enumerate(text.splitlines(keepends=True), start=1):
        content = raw.strip()
        if content:
            leading = len(raw) - len(raw.lstrip())
            start = offset + leading
            end = start + len(content)
            yield line_no, content, TextSpan(
                start=start,
                end=end,
                line=line_no,
                column=leading + 1,
                text=content,
            )
        offset += len(raw)


def find_literal_spans(text: str, term: str) -> List[TextSpan]:
    spans: List[TextSpan] = []
    if not term:
        return spans
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            break
        line, column = line_column_at(text, index)
        spans.append(TextSpan(start=index, end=index + len(term), line=line, column=column, text=term))
        start = index + 1
    return spans


def line_column_at(text: str, index: int) -> Tuple[Optional[int], Optional[int]]:
    if index < 0 or index > len(text):
        return None, None
    line = text.count("\n", 0, index) + 1
    last_newline = text.rfind("\n", 0, index)
    column = index + 1 if last_newline < 0 else index - last_newline
    return line, column
