#!/usr/bin/env python3
"""Generate the full results table for the paper.

Row: task_category × model
Column: metric × condition
Cell: mean value

Usage:
    python scripts/full_results_table.py \
        --run-a results/run_003 --run-bcd results/run_002 \
        [--format md|csv|latex]
"""

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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
GROUP_ORDER = ["热门长调", "冷门长调", "热门小令+中调", "冷门小令+中调", "诗体", "难点", "Overall"]
COND_ORDER = ["A", "B", "C", "D1", "D2"]

MODEL_RENAME = {
    "Qwen3.5-Plus": "Qwen",
    "Qwen3.5-27B": "Qwen",
}


def load_prosody(prosody_dir: str) -> List[Dict]:
    records = []
    for f in sorted(glob.glob(f"{prosody_dir}/*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line.strip())
                r["split_group"] = SPLIT_GROUPS.get(r.get("split", ""), "other")
                r["model"] = MODEL_RENAME.get(r["model"], r["model"])
                records.append(r)
    return records


def load_generation(run_a: str, run_bcd: str) -> List[Dict]:
    """Load generation records for token/time stats."""
    records = []
    for f in sorted(glob.glob(f"{run_a}/*/A.jsonl")):
        if "prosody_final" in f or "quality_rankings" in f:
            continue
        model = f.split("/")[-2]
        model = MODEL_RENAME.get(model, model)
        for line in open(f):
            r = json.loads(line.strip())
            if r.get("text", "").strip() and not r.get("error_message"):
                records.append({"model": model, "condition": "A",
                                "split": r.get("split", ""), "split_group": SPLIT_GROUPS.get(r.get("split",""), "other"),
                                "total_tokens": r.get("total_tokens", 0),
                                "elapsed_s": r.get("elapsed_s", 0),
                                "tool_rounds": r.get("tool_rounds", 0)})

    for cond in ["B", "C", "D1", "D2"]:
        for f in sorted(glob.glob(f"{run_bcd}/*/{cond}.jsonl")):
            if "prosody_final" in f or "quality_rankings" in f:
                continue
            model = f.split("/")[-2]
            model = MODEL_RENAME.get(model, model)
            for line in open(f):
                r = json.loads(line.strip())
                if r.get("text", "").strip() and not r.get("error_message"):
                    records.append({"model": model, "condition": cond,
                                    "split": r.get("split", ""), "split_group": SPLIT_GROUPS.get(r.get("split",""), "other"),
                                    "total_tokens": r.get("total_tokens", 0),
                                    "elapsed_s": r.get("elapsed_s", 0),
                                    "tool_rounds": r.get("tool_rounds", 0)})
    return records


def load_quality(run_a: str) -> Dict:
    """Load 3-judge average quality scores."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_quality import compute_leaderboard, compute_calibrated_scores, DIMENSIONS

    all_records = []
    for f in sorted(glob.glob(f"{run_a}/quality_rankings/*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line.strip())
                r["model"] = MODEL_RENAME.get(r["model"], r["model"])
                all_records.append(r)

    if not all_records:
        return {}

    # Per-judge leaderboard + calibrated, then average
    by_judge = defaultdict(list)
    for r in all_records:
        by_judge[r.get("judge", "?")].append(r)

    # Collect per (model, condition, dimension) across judges
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for judge, jrecs in by_judge.items():
        lb = compute_leaderboard(jrecs)
        cal = compute_calibrated_scores(jrecs, lb)
        for model, dims in lb.items():
            model = MODEL_RENAME.get(model, model)
            for dim in DIMENSIONS:
                scores[model]["A"][dim].append(dims.get(dim, 0))
        for model, conds in cal.items():
            model = MODEL_RENAME.get(model, model)
            for cond, dims in conds.items():
                for dim, vals in dims.items():
                    scores[model][cond][dim].append(sum(vals) / len(vals))

    # Average across judges
    result = {}
    for model in scores:
        result[model] = {}
        for cond in scores[model]:
            dim_avgs = {}
            for dim in DIMENSIONS:
                vals = scores[model][cond].get(dim, [])
                dim_avgs[dim] = sum(vals) / len(vals) if vals else 0
            dim_avgs["avg"] = sum(dim_avgs[d] for d in DIMENSIONS) / len(DIMENSIONS)
            result[model][cond] = dim_avgs
    return result


def compute_table(prosody: List[Dict], generation: List[Dict], quality: Dict):
    """Build the full table data structure."""
    # Index prosody by (model, condition, group)
    p_idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in prosody:
        p_idx[r["model"]][r["condition"]][r["split_group"]].append(r)
        p_idx[r["model"]][r["condition"]]["Overall"].append(r)

    # Index generation
    g_idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in generation:
        g_idx[r["model"]][r["condition"]][r["split_group"]].append(r)
        g_idx[r["model"]][r["condition"]]["Overall"].append(r)

    all_models = sorted(set(r["model"] for r in prosody))

    rows = []
    for group in GROUP_ORDER:
        for model in all_models:
            row = {"group": group, "model": model}
            for cond in COND_ORDER:
                precs = p_idx[model][cond].get(group, [])
                grecs = g_idx[model][cond].get(group, [])
                n = len(precs)
                prefix = f"{cond}_"

                if n == 0:
                    row[f"{prefix}n"] = None
                    for m in ["zer", "cta", "fatal_pct", "warn_avg", "avg_err", "tok", "time", "rounds", "quality"]:
                        row[f"{prefix}{m}"] = None
                    continue

                zer = sum(1 for r in precs if r["error_count"] == 0) / n * 100
                non_fatal = [r for r in precs if not r["has_fatal"]]
                n_nf = len(non_fatal)
                total_chars = sum(r.get("total_chars", 100) for r in non_fatal) if non_fatal else 1
                cta = (total_chars - sum(r["tone_errors"] for r in non_fatal)) / total_chars * 100 if non_fatal else 0
                fatal_pct = sum(1 for r in precs if r["has_fatal"]) / n * 100
                warn_avg = sum(r["warning_count"] for r in precs) / n
                avg_err = sum(r["error_count"] for r in precs) / n

                row[f"{prefix}n"] = n
                row[f"{prefix}zer"] = round(zer, 1)
                row[f"{prefix}cta"] = round(cta, 1)
                row[f"{prefix}fatal_pct"] = round(fatal_pct, 1)
                row[f"{prefix}warn_avg"] = round(warn_avg, 2)
                row[f"{prefix}avg_err"] = round(avg_err, 1)

                if grecs:
                    row[f"{prefix}tok"] = round(sum(r["total_tokens"] for r in grecs) / len(grecs))
                    row[f"{prefix}time"] = round(sum(r["elapsed_s"] for r in grecs) / len(grecs), 1)
                    row[f"{prefix}rounds"] = round(sum(r.get("tool_rounds", 0) for r in grecs) / len(grecs), 1)
                else:
                    row[f"{prefix}tok"] = None
                    row[f"{prefix}time"] = None
                    row[f"{prefix}rounds"] = None

                # Quality (only overall level, not per-group)
                if group == "Overall" and quality.get(model, {}).get(cond):
                    row[f"{prefix}quality"] = round(quality[model][cond]["avg"], 1)
                else:
                    row[f"{prefix}quality"] = None

            rows.append(row)
    return rows


def print_markdown(rows: List[Dict]):
    # Metrics to show
    metrics = [
        ("n", "N", ""),
        ("zer", "ZER", "%"),
        ("cta", "CTA", "%"),
        ("avg_err", "AvgErr", ""),
        ("fatal_pct", "Fatal", "%"),
        ("warn_avg", "Warn", ""),
        ("quality", "Quality", ""),
        ("tok", "Tokens", ""),
        ("time", "Time", "s"),
        ("rounds", "Rounds", ""),
    ]

    conds_available = set()
    for r in rows:
        for c in COND_ORDER:
            if r.get(f"{c}_n") is not None:
                conds_available.add(c)
    conds = [c for c in COND_ORDER if c in conds_available]

    # Header
    header1 = f"| {'Category':12s} | {'Model':26s} |"
    header2 = f"| {'-'*12} | {'-'*26} |"
    for metric_key, metric_name, unit in metrics:
        for c in conds:
            header1 += f" {c}_{metric_name:s}{unit} |"
            header2 += f" {'-'*max(len(f'{c}_{metric_name}{unit}'), 6)} |"

    print(header1)
    print(header2)

    # Rows
    current_group = None
    for row in rows:
        group = row["group"]
        model = row["model"]

        # Skip if model has no data at all
        has_any = any(row.get(f"{c}_n") is not None for c in conds)
        if not has_any:
            continue

        line = f"| {group if group != current_group else '':12s} | {model:26s} |"
        current_group = group

        for metric_key, metric_name, unit in metrics:
            for c in conds:
                val = row.get(f"{c}_{metric_key}")
                if val is None:
                    line += f" {'—':>6s} |"
                elif isinstance(val, float):
                    line += f" {val:6.1f} |"
                elif isinstance(val, int):
                    line += f" {val:6d} |"
                else:
                    line += f" {str(val):>6s} |"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="results/run_003")
    parser.add_argument("--run-bcd", default="results/run_002")
    parser.add_argument("--prosody-dir", default=None)
    args = parser.parse_args()

    prosody_dir = args.prosody_dir or f"{args.run_a}/prosody_final"

    print("Loading data...", flush=True)
    prosody = load_prosody(prosody_dir)
    generation = load_generation(args.run_a, args.run_bcd)
    quality = load_quality(args.run_a)

    print(f"Prosody: {len(prosody)} records, Generation: {len(generation)} records\n")

    rows = compute_table(prosody, generation, quality)

    # Print Overall section first (most important)
    overall_rows = [r for r in rows if r["group"] == "Overall"]
    other_rows = [r for r in rows if r["group"] != "Overall"]

    print("=" * 80)
    print("FULL RESULTS TABLE — Overall")
    print("=" * 80)
    print_markdown(overall_rows)

    print("\n")
    print("=" * 80)
    print("FULL RESULTS TABLE — By Form Category")
    print("=" * 80)
    print_markdown(other_rows)


if __name__ == "__main__":
    main()
