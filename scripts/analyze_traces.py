#!/usr/bin/env python3
"""Tool call finite state machine analysis.

Analyzes tool-calling traces to extract behavioral patterns:
- State transitions (which tool follows which)
- Phase classification (plan → draft → verify → revise)
- Convergence patterns (error count progression)
- Creative tool usage (e.g., free_rhyme as proxy validator in D2)

Usage:
    python scripts/analyze_traces.py --run results/run_002 [--condition A]
"""

import json
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# FSM states based on tool calls
TOOL_PHASES = {
    "rules_list": "plan",
    "examples": "plan",
    "search_text": "plan",
    "validate_meter": "verify",
    "char_lookup": "revise",
    "rhyme_lookup": "revise",
    "rhyme_list": "revise",
    "dictionary_search": "revise",
    "dictionary_allusion": "draft",
    "free_rhyme": "verify_proxy",
}

SPLIT_GROUPS = {
    "changdiao_popular": "热门长调",
    "changdiao_rare": "冷门长调",
    "zhongdiao_popular": "热门小令+中调",
    "zhongdiao_rare": "冷门小令+中调",
    "xiaoling_popular": "热门小令+中调",
    "xiaoling_rare": "冷门小令+中调",
    "regulated": "诗体",
    "special": "难点",
}


def extract_tool_sequence(trace: Dict) -> List[str]:
    """Extract ordered tool call sequence from trace."""
    seq = []
    for rd in trace.get("tool_trace", []):
        for obs in rd.get("observations", []):
            call = obs.get("call", {})
            name = call.get("name", "")
            if name:
                seq.append(name)
    return seq


def extract_phase_sequence(tool_seq: List[str]) -> List[str]:
    """Map tool sequence to phase sequence."""
    return [TOOL_PHASES.get(t, "other") for t in tool_seq]


def extract_validate_progression(trace: Dict) -> List[int]:
    """Extract error count at each validate_meter call."""
    progression = []
    for rd in trace.get("tool_trace", []):
        for obs in rd.get("observations", []):
            call = obs.get("call", {})
            if call.get("name") == "validate_meter":
                resp = obs.get("response", {})
                n_issues = len(resp.get("issues", []))
                progression.append(n_issues)
    return progression


def classify_convergence(progression: List[int]) -> str:
    """Classify the convergence pattern."""
    if not progression:
        return "no_validation"
    if progression[-1] == 0:
        if len(progression) == 1:
            return "first_pass"
        return "converged"
    if len(progression) >= 3 and progression[-1] >= progression[-2] >= progression[-3]:
        return "stagnant"
    if len(progression) >= 2 and progression[-1] > progression[0]:
        return "diverged"
    return "incomplete"


def detect_creative_usage(tool_seq: List[str], condition: str) -> List[str]:
    """Detect creative/unexpected tool usage patterns."""
    findings = []
    if condition == "D2" and "free_rhyme" in tool_seq:
        findings.append("free_rhyme_as_validator")
    if condition == "D1" and "search_text" in tool_seq:
        findings.append("search_as_template_proxy")
    # Excessive reference before drafting
    plan_tools = [t for t in tool_seq if TOOL_PHASES.get(t) == "plan"]
    if len(plan_tools) > 4:
        findings.append("excessive_planning")
    return findings


def analyze_run(run_dir: str, conditions: Optional[List[str]] = None):
    """Analyze all traces in a run directory."""
    if conditions is None:
        conditions = ["A", "D1", "D2"]

    all_results = []

    for cond in conditions:
        for jsonl_path in sorted(glob.glob(f"{run_dir}/*/{cond}.jsonl")):
            model = jsonl_path.split("/")[-2]
            records = [json.loads(l) for l in open(jsonl_path) if l.strip()]

            for rec in records:
                trace_file = rec.get("trace_file")
                if not trace_file:
                    # No trace (B/C conditions or missing)
                    result = {
                        "model": model,
                        "condition": cond,
                        "cipai": rec.get("cipai_name", ""),
                        "split": rec.get("split", ""),
                        "split_group": SPLIT_GROUPS.get(rec.get("split", ""), "other"),
                        "tool_sequence": [],
                        "phase_sequence": [],
                        "validate_progression": [],
                        "convergence": "no_tools",
                        "creative_usage": [],
                        "tool_rounds": rec.get("tool_rounds", 0),
                        "tool_call_count": rec.get("tool_call_count", 0),
                        "error_count": rec.get("error_count"),
                        "zer": 1 if rec.get("error_count", 999) == 0 else 0,
                        "elapsed_s": rec.get("elapsed_s", 0),
                        "total_tokens": rec.get("total_tokens", 0),
                    }
                    all_results.append(result)
                    continue

                # Load trace
                trace_path = f"{run_dir}/{model}/{trace_file}"
                try:
                    trace = json.loads(open(trace_path).read())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

                tool_seq = extract_tool_sequence(trace)
                phase_seq = extract_phase_sequence(tool_seq)
                val_prog = extract_validate_progression(trace)
                convergence = classify_convergence(val_prog)
                creative = detect_creative_usage(tool_seq, cond)

                result = {
                    "model": model,
                    "condition": cond,
                    "cipai": rec.get("cipai_name", ""),
                    "split": rec.get("split", ""),
                    "split_group": SPLIT_GROUPS.get(rec.get("split", ""), "other"),
                    "tool_sequence": tool_seq,
                    "phase_sequence": phase_seq,
                    "validate_progression": val_prog,
                    "convergence": convergence,
                    "creative_usage": creative,
                    "tool_rounds": rec.get("tool_rounds", 0),
                    "tool_call_count": rec.get("tool_call_count", 0),
                    "error_count": rec.get("error_count"),
                    "zer": 1 if rec.get("error_count", 999) == 0 else 0,
                    "elapsed_s": rec.get("elapsed_s", 0),
                    "total_tokens": rec.get("total_tokens", 0),
                }
                all_results.append(result)

    return all_results


def print_summary(results: List[Dict]):
    """Print analysis summary."""
    by_model_cond = defaultdict(list)
    for r in results:
        by_model_cond[(r["model"], r["condition"])].append(r)

    print("=== Convergence Patterns ===\n")
    for (model, cond), recs in sorted(by_model_cond.items()):
        conv = Counter(r["convergence"] for r in recs)
        n = len(recs)
        zer = sum(r["zer"] for r in recs)
        print(f"  {model:20s}/{cond}  n={n:3d}  ZER={zer:3d}  {dict(conv)}")

    print("\n=== Tool Usage Frequency ===\n")
    for (model, cond), recs in sorted(by_model_cond.items()):
        tool_counts = Counter()
        for r in recs:
            tool_counts.update(r["tool_sequence"])
        if tool_counts:
            top = tool_counts.most_common(5)
            top_str = ", ".join(f"{t}:{c}" for t, c in top)
            print(f"  {model:20s}/{cond}  {top_str}")

    print("\n=== State Transitions (top 10) ===\n")
    for (model, cond), recs in sorted(by_model_cond.items()):
        transitions = Counter()
        for r in recs:
            seq = r["tool_sequence"]
            for i in range(len(seq) - 1):
                transitions[(seq[i], seq[i + 1])] += 1
        if transitions:
            top = transitions.most_common(5)
            top_str = ", ".join(f"{a}→{b}:{c}" for (a, b), c in top)
            print(f"  {model:20s}/{cond}  {top_str}")

    print("\n=== Creative Usage ===\n")
    for r in results:
        if r["creative_usage"]:
            print(f"  {r['model']:20s}/{r['condition']}  {r['cipai']:16s}  {r['creative_usage']}")

    print("\n=== ZER by Split Group ===\n")
    for (model, cond), recs in sorted(by_model_cond.items()):
        by_group = defaultdict(list)
        for r in recs:
            by_group[r["split_group"]].append(r)
        parts = []
        for group in ["热门长调", "冷门长调", "热门小令+中调", "冷门小令+中调", "诗体", "难点"]:
            gr = by_group.get(group, [])
            if gr:
                zer = sum(r["zer"] for r in gr)
                parts.append(f"{group}:{zer}/{len(gr)}")
        if parts:
            print(f"  {model:20s}/{cond}  {', '.join(parts)}")


def export_json(results: List[Dict], output_path: str):
    """Export results as JSON for the viewer."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nExported {len(results)} records to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/run_002")
    parser.add_argument("--conditions", nargs="+", default=["A", "D1", "D2"])
    parser.add_argument("--export", help="Export JSON path")
    args = parser.parse_args()

    results = analyze_run(args.run, args.conditions)
    print_summary(results)

    if args.export:
        export_json(results, args.export)


if __name__ == "__main__":
    main()
