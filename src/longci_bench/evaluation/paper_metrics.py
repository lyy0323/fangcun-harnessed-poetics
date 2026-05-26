"""Paper-aligned prosodic compliance metrics.

Computes CTA, ZER, SA, RA, WER from validate_meter tool responses:

  CTA = (1/N) Σ 1[τ(cᵢ) ∈ Tᵢ]        character-level tonal accuracy
  ZER = |{p: errors(p)=0}| / |P|        zero-error rate (errors only, NOT warnings)
  SA  = line_count + char_count match    structural accuracy
  RA  = correct rhyme positions / total  rhyme accuracy
  WER = (errors + 0.5×warnings) / chars  weighted error rate

Warning exemptions: 调笑令, 皂罗特髻 — warnings zeroed for these cipai.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


WARNING_EXEMPT_CIPAI = {
    "调笑令_钦谱_格一",
    "皂罗特髻_钦谱_格一",
}


def compute_paper_metrics(
    prosody_response: Mapping[str, Any],
    text: str,
    expected_rule: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive CTA/ZER/SA/RA/WER from a validate_meter response.

    Error classification:
      - fatal: Length/structural mismatch (char count wrong) — invalidates all other metrics
      - shifted: Tone/Rhyme errors that co-occur with a Length error (caused by offset, not real)
      - true: Tone/Rhyme errors with no Length error (genuine prosodic violations)
    """
    raw = _unwrap(prosody_response)
    errors = _get_errors(raw)
    segments = _get_segments(raw)
    warnings = _get_warnings(raw)

    tone_errors = [e for e in errors if _error_type(e) == "Tone"]
    rhyme_errors = [e for e in errors if _error_type(e) == "Rhyme"]
    length_errors = [e for e in errors if _error_type(e) in ("Length", "Structural")]
    punct_errors = [e for e in errors if _error_type(e) == "Punctuation"]

    total_chars = _total_chars_from_segments(segments) or _count_cjk(text)
    content_errors = len(tone_errors) + len(rhyme_errors)
    has_fatal = len(length_errors) > 0 or content_errors > 10

    # SA: structural accuracy
    closest_rule = raw.get("closest_rule") or _nested(raw, "metrics", "closest_rule") or {}
    expected_chars = closest_rule.get("char_count")
    sa = 0.0 if has_fatal else (1.0 if expected_chars is None or total_chars == expected_chars else 0.0)

    if has_fatal:
        # Fatal: char count mismatch — CTA/RA are meaningless (shifted data)
        cta = 0.0
        ra = 0.0
        zer = 0.0
        error_class = "fatal"
    else:
        # CTA
        cta = (total_chars - len(tone_errors)) / max(total_chars, 1)
        # RA
        rhyme_positions = (
            raw.get("rhyme_positions")
            or _nested(raw, "metrics", "rhyme_positions")
            or []
        )
        total_rhyme_positions = max(len(rhyme_positions), 1)
        ra = (total_rhyme_positions - len(rhyme_errors)) / total_rhyme_positions
        # ZER: zero ERRORS (warnings excluded)
        zer = 1.0 if len(errors) == 0 else 0.0
        error_class = "clean" if len(errors) == 0 else "true"

    # WER
    cipai = expected_rule or closest_rule.get("name") or ""
    warning_count = len(warnings) if cipai not in WARNING_EXEMPT_CIPAI else 0
    wer = (len(errors) + 0.5 * warning_count) / max(total_chars, 1)

    return {
        "cta": round(cta, 4),
        "zer": zer,
        "sa": sa,
        "ra": round(ra, 4),
        "wer": round(wer, 4),
        "error_class": error_class,
        "has_fatal": has_fatal,
        "tone_error_count": len(tone_errors),
        "rhyme_error_count": len(rhyme_errors),
        "length_error_count": len(length_errors),
        "punctuation_error_count": len(punct_errors),
        "total_error_count": len(errors),
        "warning_count": warning_count,
        "warning_exempt": cipai in WARNING_EXEMPT_CIPAI,
        "total_chars": total_chars,
    }


def aggregate_paper_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-poem paper metrics into dataset-level stats.

    CTA and RA are computed only over non-fatal records (where char count matches).
    ZER and SA include all records.
    """
    n = len(records)
    if n == 0:
        return {}

    non_fatal = [r for r in records if not r.get("has_fatal", False)]
    n_nf = len(non_fatal)

    zer_rate = sum(r["zer"] for r in records) / n
    sa_rate = sum(r["sa"] for r in records) / n
    wer_avg = sum(r["wer"] for r in records) / n
    fatal_rate = sum(1 for r in records if r.get("has_fatal")) / n

    cta_avg = sum(r["cta"] for r in non_fatal) / n_nf if n_nf else 0
    ra_avg = sum(r["ra"] for r in non_fatal) / n_nf if n_nf else 0

    return {
        "CTA": round(cta_avg, 4),
        "ZER": round(zer_rate, 4),
        "SA": round(sa_rate, 4),
        "RA": round(ra_avg, 4),
        "WER": round(wer_avg, 4),
        "fatal_rate": round(fatal_rate, 4),
        "n": n,
        "n_non_fatal": n_nf,
    }


def _unwrap(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Handle both raw API response and ToolResponse wrapper."""
    d = dict(data)
    raw = d.get("metadata", {}).get("raw_response")
    if isinstance(raw, dict):
        merged = dict(raw)
        merged.update({k: v for k, v in d.items() if k not in ("metadata",)})
        return merged
    return d


def _get_errors(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = data.get("errors")
    if isinstance(errors, list):
        return [e for e in errors if isinstance(e, dict)]
    issues = data.get("issues")
    if isinstance(issues, list):
        return [e for e in issues if isinstance(e, dict)]
    return []


def _get_warnings(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    warnings = data.get("warnings")
    if isinstance(warnings, list):
        return [w for w in warnings if isinstance(w, (dict, str))]
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        warnings = meta.get("warnings")
        if isinstance(warnings, list):
            return [w for w in warnings if isinstance(w, (dict, str))]
    return []


def _get_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = data.get("display_segments")
    if isinstance(segments, list):
        return [s for s in segments if isinstance(s, dict)]
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict):
        segments = metrics.get("display_segments")
        if isinstance(segments, list):
            return [s for s in segments if isinstance(s, dict)]
    return []


def _total_chars_from_segments(segments: List[Dict[str, Any]]) -> int:
    return sum(len(seg.get("text_chars", [])) for seg in segments)


def _error_type(error: Dict[str, Any]) -> str:
    return str(error.get("error_type") or error.get("type") or "")


def _nested(data: Dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if "一" <= ch <= "鿿")
