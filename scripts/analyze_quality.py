#!/usr/bin/env python3
"""Analyze quality ranking results and produce summary tables.

Reads quality_rankings/*.jsonl, computes calibrated scores where:
  - A score = leaderboard score (north star, consistent across all tables)
  - B/C/D scores can exceed A: if A=80 and C ranks above A, C = 80 / (80/100) = 100

Usage:
    python scripts/analyze_quality.py --run results/run_003
"""

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


DIMENSIONS = ["fluency", "coherence", "poetic_quality"]
CONDITION_RANK_TO_FACTOR = {1: 1.0, 2: 0.8, 3: 0.6}  # rank position → factor


def load_rankings(run_dir: str) -> Dict[str, List[Dict]]:
    """Load all ranking records grouped by judge."""
    by_judge = defaultdict(list)
    for f in sorted(glob.glob(f"{run_dir}/quality_rankings/*.jsonl")):
        judge = Path(f).stem
        for line in open(f):
            if line.strip():
                r = json.loads(line.strip())
                by_judge[judge].append(r)
    return by_judge


def compute_leaderboard(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute per-model average leaderboard scores (north star A scores)."""
    lb = [r for r in records if r["task"] == "leaderboard"]
    model_dim_scores = defaultdict(lambda: defaultdict(list))
    for r in lb:
        model_dim_scores[r["model"]][r["dimension"]].append(r["score"])

    result = {}
    for model, dims in model_dim_scores.items():
        result[model] = {}
        for dim in DIMENSIONS:
            scores = dims.get(dim, [])
            result[model][dim] = sum(scores) / len(scores) if scores else 0
        result[model]["avg"] = sum(result[model][d] for d in DIMENSIONS) / len(DIMENSIONS)
    return result


def compute_calibrated_scores(records: List[Dict], leaderboard: Dict) -> Dict:
    """Compute calibrated B/C/D scores.

    For cross-condition/ablation tasks:
      - Find A's rank and other conditions' ranks
      - A score = leaderboard score (fixed)
      - Other conditions: if ranked above A, score > A; if below, score < A
      - Formula: score = A_score × (condition_factor / A_factor)
    """
    # Group cross-condition and ablation records by (model, prompt_id, task_type, dimension)
    task_groups = defaultdict(list)
    for r in records:
        if r["task"] in ("cross_condition", "ablation"):
            key = (r["model"], r["prompt_id"], r["task"], r["dimension"])
            task_groups[key].append(r)

    # For each group, recalibrate
    calibrated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (model, prompt_id, task_type, dim), group in task_groups.items():
        a_base = leaderboard.get(model, {}).get(dim, 70)

        # Find A's rank factor and each condition's rank factor
        rank_map = {}
        for r in group:
            n = r["n_candidates"]
            rank = r["rank"]
            # Map rank to factor: 1st=1.0, 2nd=0.8, 3rd=0.6
            factor = CONDITION_RANK_TO_FACTOR.get(rank, max(0.4, 1.0 - 0.2 * (rank - 1)))
            rank_map[r["condition"]] = factor

        a_factor = rank_map.get("A", 1.0)

        for r in group:
            cond = r["condition"]
            cond_factor = rank_map[cond]
            if cond == "A":
                score = a_base
            else:
                score = min(100, a_base * (cond_factor / a_factor)) if a_factor > 0 else a_base
            calibrated[model][cond][dim].append(round(score, 1))

    return calibrated


def print_report(run_dir: str):
    by_judge = load_rankings(run_dir)

    for judge, records in sorted(by_judge.items()):
        print(f"\n{'=' * 75}")
        print(f"Judge: {judge} ({len(records)} records)")
        print(f"{'=' * 75}")

        leaderboard = compute_leaderboard(records)
        calibrated = compute_calibrated_scores(records, leaderboard)

        # --- Leaderboard ---
        print(f"\n--- Leaderboard (Condition A, North Star) ---\n")
        print(f"{'Model':28s} {'fluency':>8} {'coherence':>10} {'poetic_q':>9} {'avg':>6}")
        print("-" * 65)
        rows = [(v["avg"], m, v) for m, v in leaderboard.items()]
        for avg, model, scores in sorted(rows, reverse=True):
            print(f"  {model:26s} {scores['fluency']:8.1f} {scores['coherence']:10.1f} "
                  f"{scores['poetic_quality']:9.1f} {avg:6.1f}")

        # --- Big table: all models × all conditions ---
        all_models = sorted(leaderboard.keys(), key=lambda m: -leaderboard[m]["avg"])
        all_conds = ["A", "B", "C", "D1", "D2"]

        print(f"\n--- Full Score Table (calibrated) ---\n")
        print(f"{'Model':28s} {'Cond':>4} {'fluency':>8} {'coherence':>10} {'poetic_q':>9} {'avg':>6} {'n':>4}")
        print("-" * 75)

        for model in all_models:
            for cond in all_conds:
                if cond == "A":
                    scores = leaderboard.get(model, {})
                    if not scores:
                        continue
                    n = len([r for r in records if r["task"] == "leaderboard"
                             and r["model"] == model and r["dimension"] == "fluency"])
                    avg = scores.get("avg", 0)
                    print(f"  {model:26s} {cond:>4} {scores.get('fluency',0):8.1f} "
                          f"{scores.get('coherence',0):10.1f} {scores.get('poetic_quality',0):9.1f} "
                          f"{avg:6.1f} {n:4d}")
                else:
                    dims = calibrated.get(model, {}).get(cond, {})
                    if not dims:
                        continue
                    scores = {d: sum(s) / len(s) for d, s in dims.items()}
                    n = len(dims.get("fluency", []))
                    avg = sum(scores.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS)
                    print(f"  {model:26s} {cond:>4} {scores.get('fluency',0):8.1f} "
                          f"{scores.get('coherence',0):10.1f} {scores.get('poetic_quality',0):9.1f} "
                          f"{avg:6.1f} {n:4d}")
            print()

        # --- Condition comparison summary ---
        print(f"--- A vs B vs C Summary ---\n")
        print(f"{'Model':28s} {'A avg':>7} {'B avg':>7} {'C avg':>7} {'B-A':>6} {'C-A':>6}")
        print("-" * 60)
        for model in all_models:
            a_avg = leaderboard.get(model, {}).get("avg", 0)
            b_dims = calibrated.get(model, {}).get("B", {})
            c_dims = calibrated.get(model, {}).get("C", {})
            b_avg = sum(sum(s)/len(s) for s in b_dims.values()) / len(DIMENSIONS) if b_dims else None
            c_avg = sum(sum(s)/len(s) for s in c_dims.values()) / len(DIMENSIONS) if c_dims else None
            b_str = f"{b_avg:7.1f}" if b_avg else "     — "
            c_str = f"{c_avg:7.1f}" if c_avg else "     — "
            b_delta = f"{b_avg - a_avg:+6.1f}" if b_avg else "    — "
            c_delta = f"{c_avg - a_avg:+6.1f}" if c_avg else "    — "
            print(f"  {model:26s} {a_avg:7.1f} {b_str} {c_str} {b_delta} {c_delta}")

        # --- Ablation summary ---
        ab_models = [m for m in all_models if calibrated.get(m, {}).get("D1") or calibrated.get(m, {}).get("D2")]
        if ab_models:
            print(f"\n--- Ablation Summary ---\n")
            print(f"{'Model':28s} {'A avg':>7} {'D1 avg':>7} {'D2 avg':>7} {'D1-A':>6} {'D2-A':>6}")
            print("-" * 60)
            for model in ab_models:
                a_avg = leaderboard.get(model, {}).get("avg", 0)
                d1_dims = calibrated.get(model, {}).get("D1", {})
                d2_dims = calibrated.get(model, {}).get("D2", {})
                d1_avg = sum(sum(s)/len(s) for s in d1_dims.values()) / len(DIMENSIONS) if d1_dims else None
                d2_avg = sum(sum(s)/len(s) for s in d2_dims.values()) / len(DIMENSIONS) if d2_dims else None
                d1_str = f"{d1_avg:7.1f}" if d1_avg else "     — "
                d2_str = f"{d2_avg:7.1f}" if d2_avg else "     — "
                d1_delta = f"{d1_avg - a_avg:+6.1f}" if d1_avg else "    — "
                d2_delta = f"{d2_avg - a_avg:+6.1f}" if d2_avg else "    — "
                print(f"  {model:26s} {a_avg:7.1f} {d1_str} {d2_str} {d1_delta} {d2_delta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/run_003")
    args = parser.parse_args()
    print_report(args.run)


if __name__ == "__main__":
    main()
