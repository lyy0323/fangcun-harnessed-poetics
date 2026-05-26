#!/usr/bin/env python3
"""Analyze similarity / originality results across runs and conditions.

Usage:
    python scripts/analyze_similarity.py --run-a results/run_003 --run-bcd results/run_002
"""

import argparse
import json
import glob
from collections import defaultdict
from typing import Dict, List, Optional


def collect_data(run_a: str, run_bcd: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Collect similarity metrics grouped by model → condition."""
    data = defaultdict(lambda: defaultdict(list))

    # A from run_003
    for f in sorted(glob.glob(f"{run_a}/*/A.jsonl")):
        model = f.split("/")[-2]
        for line in open(f):
            r = json.loads(line.strip())
            if r.get("max_similarity") is not None and r.get("text", "").strip():
                data[model]["A"].append(r)

    # B/C/D from run_002
    for cond in ["B", "C", "D1", "D2"]:
        for f in sorted(glob.glob(f"{run_bcd}/*/{cond}.jsonl")):
            model = f.split("/")[-2]
            for line in open(f):
                r = json.loads(line.strip())
                if r.get("max_similarity") is not None and r.get("text", "").strip():
                    data[model][cond].append(r)

    return data


def print_report(data: Dict):
    all_conds = ["A", "B", "C", "D1", "D2"]

    # --- Originality overview ---
    print("=== Originality Score (1 = fully original, 0 = all copied) ===\n")
    print(f"{'Model':28s} {'A':>7} {'B':>7} {'C':>7} {'D1':>7} {'D2':>7}  A−C")
    print("-" * 78)

    sorted_models = sorted(data.keys(),
        key=lambda m: -(sum(r.get("originality_score", 0) for r in data[m].get("A", [{"originality_score": 0}]))
                        / max(len(data[m].get("A", [1])), 1)))

    for model in sorted_models:
        avgs = {}
        for c in all_conds:
            recs = data[model].get(c, [])
            if recs:
                avgs[c] = sum(r.get("originality_score", 0) for r in recs) / len(recs)

        parts = [f"{avgs[c]:.3f}" if c in avgs else "    — " for c in all_conds]
        delta = f"{avgs['A'] - avgs['C']:+.3f}" if "A" in avgs and "C" in avgs else "   — "
        print(f"  {model:26s} {'  '.join(parts)}  {delta}")

    # --- Plagiarism flags ---
    print(f"\n=== Plagiarism Flags (score=1.0 match to pre-modern author, >4 chars) ===\n")
    print(f"{'Model':28s} {'Cond':>4} {'n':>4} {'Plag':>5} {'Rate':>6}")
    print("-" * 52)

    for model in sorted_models:
        for c in all_conds:
            recs = data[model].get(c, [])
            if not recs:
                continue
            plag = sum(1 for r in recs if r.get("plagiarism_flag"))
            if plag > 0:
                print(f"  {model:26s} {c:>4} {len(recs):4d} {plag:5d} {plag/len(recs)*100:5.1f}%")

    # --- High similarity sentences ---
    print(f"\n=== High Similarity Sentences (score ≥ 0.8) ===\n")
    print(f"{'Model':28s} {'Cond':>4} {'n':>4} {'HiSim':>6} {'per_poem':>8}")
    print("-" * 55)

    for model in sorted_models:
        for c in all_conds:
            recs = data[model].get(c, [])
            if not recs:
                continue
            hi = sum(r.get("high_similarity_count", 0) for r in recs)
            if hi > 0:
                print(f"  {model:26s} {c:>4} {len(recs):4d} {hi:6d} {hi/len(recs):8.2f}")

    # --- Max similarity distribution ---
    print(f"\n=== Max Similarity Distribution ===\n")
    print(f"{'Model':28s} {'Cond':>4} {'n':>4} {'avg':>6} {'<0.5':>5} {'0.5-0.8':>7} {'≥0.8':>5} {'=1.0':>5}")
    print("-" * 70)

    for model in sorted_models:
        for c in ["A", "C"]:
            recs = data[model].get(c, [])
            if not recs:
                continue
            maxsims = [r.get("max_similarity", 0) for r in recs]
            avg = sum(maxsims) / len(maxsims)
            lo = sum(1 for s in maxsims if s < 0.5)
            mid = sum(1 for s in maxsims if 0.5 <= s < 0.8)
            hi = sum(1 for s in maxsims if s >= 0.8)
            exact = sum(1 for s in maxsims if s >= 1.0)
            print(f"  {model:26s} {c:>4} {len(recs):4d} {avg:6.3f} {lo:5d} {mid:7d} {hi:5d} {exact:5d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", default="results/run_003", help="Run with Condition A")
    parser.add_argument("--run-bcd", default="results/run_002", help="Run with B/C/D conditions")
    args = parser.parse_args()

    data = collect_data(args.run_a, args.run_bcd)
    print_report(data)


if __name__ == "__main__":
    main()
