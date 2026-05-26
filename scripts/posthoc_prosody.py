#!/usr/bin/env python3
"""Final prosody evaluation: batch validate all poems with correct parameters.

For each poem:
  - A/D1: Extract final text (>300ch = badcase, fallback to last validate_meter text)
         Compare posthoc result with trace's last validate_meter, keep the better one
  - B/C/D2: Use text field directly

Validation params:
  - Ci: Cilinzhengyun, warnings=["2gram","rhyme_duplicate"]
  - Shi: Pingshuiyun, warnings="default"
  - rule_name: cipai prefix (allow variant matching)

Output: {run}/prosody_final/{condition}.jsonl

Usage:
    python scripts/posthoc_prosody.py --run-a results/run_003 --run-bcd results/run_002
"""

import argparse
import os
import json
import glob
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request

BATCH_URL = os.environ.get("FANGCUN_CHECKER_URL", "http://localhost:8901") + "/api/validate_batch"
BATCH_SIZE = 24
RATE_SLEEP = 2.5
WARNING_EXEMPT = {"调笑令_钦谱_格一", "皂罗特髻_钦谱_格一"}


def validate_batch(items: List[Dict]) -> List[Dict]:
    body = json.dumps({"items": items}, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(BATCH_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib_request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result if isinstance(result, list) else result.get("results", [])


def classify_errors(raw: Dict) -> Dict:
    errors = raw.get("errors", [])
    warnings = raw.get("warnings", [])
    closest = raw.get("closest_rule", {})

    def et(e): return e.get("error_type", "") if isinstance(e, dict) else ""

    tone = sum(1 for e in errors if et(e) == "Tone")
    rhyme = sum(1 for e in errors if et(e) == "Rhyme")
    punct = sum(1 for e in errors if et(e) == "Punctuation")
    length = sum(1 for e in errors if et(e) in ("Length", "Structural"))
    other = len(errors) - tone - rhyme - punct - length

    content_errors = tone + rhyme
    has_fatal = length > 0 or content_errors > 10

    cipai_name = closest.get("name", "") if isinstance(closest, dict) else ""
    warn_count = 0 if cipai_name in WARNING_EXEMPT else len(warnings)

    return {
        "is_valid": raw.get("is_valid", False),
        "error_count": len(errors),
        "tone_errors": tone,
        "rhyme_errors": rhyme,
        "punctuation_errors": punct,
        "length_errors": length,
        "other_errors": other,
        "warning_count": warn_count,
        "warning_exempt": cipai_name in WARNING_EXEMPT,
        "has_fatal": has_fatal,
        "closest_rule": cipai_name,
        "total_score": len(errors) + 0.5 * warn_count,  # for comparison
    }


def extract_text_for_validation(record: Dict, condition: str, run_dir: str, model: str) -> tuple:
    """Extract the best text for validation. Returns (text, source)."""
    text = record.get("text", "").strip()

    if condition not in ("A", "D1"):
        return text, "output"

    # A/D1: check if text is suspiciously long (thinking dump)
    if len(text) > 300:
        # Fallback to last validate_meter text from trace
        trace_file = record.get("trace_file")
        if trace_file:
            trace_path = f"{run_dir}/{model}/{trace_file}"
            try:
                trace = json.loads(open(trace_path).read())
                for rd in reversed(trace.get("tool_trace", [])):
                    for obs in reversed(rd.get("observations", [])):
                        call = obs.get("call", {})
                        if call.get("name") == "validate_meter":
                            args = call.get("arguments", {})
                            vm_text = args.get("poem_text") or args.get("text", "")
                            if vm_text and len(vm_text) <= 300:
                                return vm_text.strip(), "trace_fallback"
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return text, "output_long"

    return text, "output"


def extract_trace_validation(record: Dict, run_dir: str, model: str) -> Optional[Dict]:
    """Extract last validate_meter result from trace for A/D1."""
    trace_file = record.get("trace_file")
    if not trace_file:
        return None
    trace_path = f"{run_dir}/{model}/{trace_file}"
    try:
        trace = json.loads(open(trace_path).read())
        for rd in reversed(trace.get("tool_trace", [])):
            for obs in reversed(rd.get("observations", [])):
                call = obs.get("call", {})
                if call.get("name") == "validate_meter":
                    resp = obs.get("response", {})
                    raw = resp.get("metadata", {}).get("raw_response")
                    if raw:
                        return raw
                    # Reconstruct from response fields
                    return {
                        "is_valid": resp.get("passed"),
                        "errors": resp.get("issues", []),
                        "warnings": resp.get("metadata", {}).get("warnings", []),
                        "closest_rule": resp.get("metrics", {}).get("closest_rule"),
                    }
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def process_run(run_dir: str, conditions: List[str], output_dir: Path, other_run: Optional[str] = None):
    """Process all conditions in a run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        search_dir = run_dir if cond == "A" or cond == "D1" else (other_run or run_dir)
        files = sorted(glob.glob(f"{search_dir}/*/{cond}.jsonl"))
        if not files:
            continue

        all_results = []

        for f in files:
            model = f.split("/")[-2]
            records = [json.loads(l) for l in open(f) if l.strip()]

            # Prepare validation items
            items_to_validate = []
            for i, r in enumerate(records):
                text, source = extract_text_for_validation(r, cond, search_dir, model)
                if not text:
                    continue
                cipai = r.get("cipai", "")
                is_ci = not any(cipai.startswith(p) for p in ("五绝", "七绝", "五律", "七律"))
                prefix = cipai.split("_")[0] if "_" in cipai else cipai

                items_to_validate.append({
                    "idx": i,
                    "record": r,
                    "model": model,
                    "text": text,
                    "source": source,
                    "api_item": {
                        "poem_text": text,
                        "genre": "Ci" if is_ci else "Shi",
                        "rule_name": prefix,
                        "rhyme_book_name": "Cilinzhengyun" if is_ci else "Pingshuiyun",
                        "warnings": ["2gram", "rhyme_duplicate"] if is_ci else "default",
                    },
                })

            if not items_to_validate:
                continue

            print(f"  {model}/{cond}: {len(items_to_validate)} poems...", end="", flush=True)

            # Batch validate
            posthoc_results = {}
            for batch_start in range(0, len(items_to_validate), BATCH_SIZE):
                batch = items_to_validate[batch_start:batch_start + BATCH_SIZE]
                api_items = [item["api_item"] for item in batch]
                try:
                    responses = validate_batch(api_items)
                    for item, raw in zip(batch, responses):
                        posthoc_results[item["idx"]] = classify_errors(raw)
                except Exception as exc:
                    print(f" batch_err({exc})", end="")
                if batch_start + BATCH_SIZE < len(items_to_validate):
                    time.sleep(RATE_SLEEP)

            # For A/D1: compare with trace validation, keep better
            for item in items_to_validate:
                idx = item["idx"]
                r = item["record"]
                posthoc = posthoc_results.get(idx)
                if not posthoc:
                    continue

                final = posthoc
                final_source = "posthoc"

                if cond in ("A", "D1"):
                    trace_raw = extract_trace_validation(r, search_dir, model)
                    if trace_raw:
                        trace_result = classify_errors(trace_raw)
                        # Keep the one with fewer total_score (errors + 0.5*warnings)
                        if trace_result["total_score"] <= posthoc["total_score"]:
                            final = trace_result
                            final_source = "trace"

                all_results.append({
                    "id": r.get("id"),
                    "model": model,
                    "condition": cond,
                    "cipai": r.get("cipai", ""),
                    "cipai_name": r.get("cipai_name", ""),
                    "split": r.get("split", ""),
                    "text_source": item["source"],
                    "validation_source": final_source,
                    **final,
                })

            print(f" done ({len(posthoc_results)} validated)")

        # Write output
        out_file = output_dir / f"{cond}.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for r in all_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  → {out_file} ({len(all_results)} records)")


def print_summary(output_dir: Path):
    print(f"\n{'=' * 80}")
    print("FINAL PROSODY SUMMARY")
    print(f"{'=' * 80}\n")

    for f in sorted(glob.glob(f"{output_dir}/*.jsonl")):
        cond = Path(f).stem
        records = [json.loads(l) for l in open(f) if l.strip()]
        if not records:
            continue

        by_model = defaultdict(list)
        for r in records:
            by_model[r["model"]].append(r)

        print(f"--- Condition {cond} ---")
        print(f"{'Model':28s} {'n':>4} {'ZER':>5} {'ZER%':>5} {'tone':>5} {'rhyme':>5} {'punct':>5} {'fatal':>5} {'warn':>5}")
        print("-" * 75)

        for model in sorted(by_model.keys()):
            recs = by_model[model]
            n = len(recs)
            zer = sum(1 for r in recs if r["error_count"] == 0)
            tone = sum(r["tone_errors"] for r in recs)
            rhyme = sum(r["rhyme_errors"] for r in recs)
            punct = sum(r["punctuation_errors"] for r in recs)
            fatal = sum(1 for r in recs if r["has_fatal"])
            warn = sum(r["warning_count"] for r in recs)
            print(f"  {model:26s} {n:4d} {zer:5d} {zer/n*100:4.0f}% {tone:5d} {rhyme:5d} {punct:5d} {fatal:5d} {warn:5d}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="results/run_003", help="Run with A/D1")
    parser.add_argument("--run-bcd", default="results/run_002", help="Run with B/C/D2")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D1", "D2"])
    args = parser.parse_args()

    output_dir = Path(args.run_a) / "prosody_final"

    process_run(args.run_a, [c for c in args.conditions if c in ("A", "D1")],
                output_dir, other_run=None)
    process_run(args.run_bcd, [c for c in args.conditions if c not in ("A", "D1")],
                output_dir, other_run=None)

    print_summary(output_dir)


if __name__ == "__main__":
    main()
