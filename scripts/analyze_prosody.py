#!/usr/bin/env python3
"""Comprehensive prosody statistics report.

Reads prosody_final/*.jsonl and produces:
  1. Overall ZER/CTA/SA/RA by model × condition
  2. Error breakdown (tone/rhyme/punct/fatal/warnings)
  3. Split-group breakdown (热门长调, 冷门长调, etc.)
  4. Cross-condition delta tables

Usage:
    python scripts/analyze_prosody.py --dir results/run_003/prosody_final
"""

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


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
GROUP_ORDER = ["热门长调", "冷门长调", "热门小令+中调", "冷门小令+中调", "诗体", "难点"]
COND_ORDER = ["A", "B", "C", "D1", "D2"]


def load_all(prosody_dir: str) -> List[Dict]:
    records = []
    for f in sorted(glob.glob(f"{prosody_dir}/*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line.strip())
                r["split_group"] = SPLIT_GROUPS.get(r.get("split", ""), "other")
                records.append(r)
    return records


def compute_metrics(recs: List[Dict]) -> Dict:
    n = len(recs)
    if n == 0:
        return {"n": 0, "zer": 0, "zer_pct": 0, "cta": 0, "tone": 0, "rhyme": 0,
                "punct": 0, "fatal": 0, "fatal_pct": 0, "warn": 0, "warn_avg": 0, "avg_err": 0}
    zer = sum(1 for r in recs if r["error_count"] == 0)
    non_fatal = [r for r in recs if not r["has_fatal"]]
    n_nf = len(non_fatal)
    tone = sum(r["tone_errors"] for r in recs)
    rhyme = sum(r["rhyme_errors"] for r in recs)
    punct = sum(r["punctuation_errors"] for r in recs)
    fatal = sum(1 for r in recs if r["has_fatal"])
    warn = sum(r["warning_count"] for r in recs)
    total_chars = sum(r.get("total_chars", 0) or 100 for r in non_fatal) if non_fatal else 1
    cta = (total_chars - sum(r["tone_errors"] for r in non_fatal)) / total_chars if non_fatal else 0
    return {
        "n": n, "zer": zer, "zer_pct": round(zer / n * 100, 1),
        "cta": round(cta * 100, 1),
        "tone": tone, "rhyme": rhyme, "punct": punct,
        "fatal": fatal, "fatal_pct": round(fatal / n * 100, 1),
        "warn": warn, "warn_avg": round(warn / n, 2),
        "avg_err": round(sum(r["error_count"] for r in recs) / n, 1),
    }


def print_report(records: List[Dict]):
    by_model_cond = defaultdict(lambda: defaultdict(list))
    by_model_cond_group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_models = set()
    all_conds = set()

    for r in records:
        model = r["model"]
        cond = r["condition"]
        by_model_cond[model][cond].append(r)
        by_model_cond_group[model][cond][r["split_group"]].append(r)
        all_models.add(model)
        all_conds.add(cond)

    conds = [c for c in COND_ORDER if c in all_conds]
    models_sorted = sorted(all_models, key=lambda m: -compute_metrics(by_model_cond[m].get("A", []))["zer_pct"])

    # === Table 1: Overall ZER ===
    print("=" * 80)
    print("TABLE 1: Zero-Error Rate (ZER%) by Model × Condition")
    print("=" * 80)
    header = f"{'Model':28s}" + "".join(f"{c:>8}" for c in conds)
    print(header)
    print("-" * len(header))
    for model in models_sorted:
        parts = [f"  {model:26s}"]
        for c in conds:
            m = compute_metrics(by_model_cond[model].get(c, []))
            if m["n"] == 0:
                parts.append(f"{'—':>8}")
            else:
                parts.append(f"{m['zer_pct']:>7.1f}%")
        print("".join(parts))
    print()

    # === Table 2: Error breakdown ===
    print("=" * 80)
    print("TABLE 2: Error Breakdown (Condition A)")
    print("=" * 80)
    print(f"{'Model':28s} {'n':>4} {'ZER%':>6} {'CTA%':>6} {'tone':>6} {'rhyme':>6} {'fatal':>6} {'warn':>6} {'avg_e':>6}")
    print("-" * 80)
    for model in models_sorted:
        m = compute_metrics(by_model_cond[model].get("A", []))
        if m["n"] == 0:
            continue
        print(f"  {model:26s} {m['n']:4d} {m['zer_pct']:5.1f}% {m['cta']:5.1f}% {m['tone']:6d} {m['rhyme']:6d} {m['fatal']:6d} {m['warn']:6d} {m['avg_err']:6.1f}")
    print()

    # === Table 3: By split group (Condition A) ===
    print("=" * 80)
    print("TABLE 3: ZER% by Form Category (Condition A)")
    print("=" * 80)
    header = f"{'Model':28s}" + "".join(f"{g:>12}" for g in GROUP_ORDER)
    print(header)
    print("-" * len(header))
    for model in models_sorted:
        parts = [f"  {model:26s}"]
        for g in GROUP_ORDER:
            recs = by_model_cond_group[model].get("A", {}).get(g, [])
            if not recs:
                parts.append(f"{'—':>12}")
            else:
                m = compute_metrics(recs)
                parts.append(f"{m['zer']}/{m['n']}({m['zer_pct']:.0f}%)")
                parts[-1] = f"{parts[-1]:>12}"
        print("".join(parts))
    print()

    # === Table 4: By split group (Condition C baseline) ===
    print("=" * 80)
    print("TABLE 4: ZER% by Form Category (Condition C, Baseline)")
    print("=" * 80)
    header = f"{'Model':28s}" + "".join(f"{g:>12}" for g in GROUP_ORDER)
    print(header)
    print("-" * len(header))
    for model in models_sorted:
        parts = [f"  {model:26s}"]
        for g in GROUP_ORDER:
            recs = by_model_cond_group[model].get("C", {}).get(g, [])
            if not recs:
                parts.append(f"{'—':>12}")
            else:
                m = compute_metrics(recs)
                parts.append(f"{m['zer']}/{m['n']}({m['zer_pct']:.0f}%)")
                parts[-1] = f"{parts[-1]:>12}"
        print("".join(parts))
    print()

    # === Table 5: A vs C delta by split group ===
    print("=" * 80)
    print("TABLE 5: ZER% Improvement (A − C) by Form Category")
    print("=" * 80)
    header = f"{'Model':28s}" + "".join(f"{g:>12}" for g in GROUP_ORDER) + f"{'Overall':>12}"
    print(header)
    print("-" * len(header))
    for model in models_sorted:
        parts = [f"  {model:26s}"]
        for g in GROUP_ORDER:
            a_recs = by_model_cond_group[model].get("A", {}).get(g, [])
            c_recs = by_model_cond_group[model].get("C", {}).get(g, [])
            if a_recs and c_recs:
                a_m = compute_metrics(a_recs)
                c_m = compute_metrics(c_recs)
                delta = a_m["zer_pct"] - c_m["zer_pct"]
                parts.append(f"{delta:+11.0f}pp")
            else:
                parts.append(f"{'—':>12}")
        # Overall
        a_all = compute_metrics(by_model_cond[model].get("A", []))
        c_all = compute_metrics(by_model_cond[model].get("C", []))
        if a_all["n"] and c_all["n"]:
            delta = a_all["zer_pct"] - c_all["zer_pct"]
            parts.append(f"{delta:+11.0f}pp")
        else:
            parts.append(f"{'—':>12}")
        print("".join(parts))
    print()

    # === Table 6: Ablation ===
    ablation_models = [m for m in models_sorted
                       if by_model_cond[m].get("D1") or by_model_cond[m].get("D2")]
    if ablation_models:
        print("=" * 80)
        print("TABLE 6: Ablation (D1=−Template, D2=−Validation)")
        print("=" * 80)
        print(f"{'Model':28s} {'A ZER%':>8} {'D1 ZER%':>8} {'D2 ZER%':>8} {'D1−A':>7} {'D2−A':>7}")
        print("-" * 70)
        for model in ablation_models:
            a = compute_metrics(by_model_cond[model].get("A", []))
            d1 = compute_metrics(by_model_cond[model].get("D1", []))
            d2 = compute_metrics(by_model_cond[model].get("D2", []))
            a_str = f"{a['zer_pct']:7.1f}%" if a["n"] else "     — "
            d1_str = f"{d1['zer_pct']:7.1f}%" if d1["n"] else "     — "
            d2_str = f"{d2['zer_pct']:7.1f}%" if d2["n"] else "     — "
            d1_d = f"{d1['zer_pct']-a['zer_pct']:+6.0f}pp" if d1["n"] and a["n"] else "    — "
            d2_d = f"{d2['zer_pct']-a['zer_pct']:+6.0f}pp" if d2["n"] and a["n"] else "    — "
            print(f"  {model:26s} {a_str} {d1_str} {d2_str} {d1_d} {d2_d}")

        # D1/D2 by split group
        print(f"\n  D1 by Form Category:")
        for model in ablation_models:
            if not by_model_cond[model].get("D1"):
                continue
            parts = [f"    {model:24s}"]
            for g in GROUP_ORDER:
                recs = by_model_cond_group[model].get("D1", {}).get(g, [])
                if recs:
                    m = compute_metrics(recs)
                    parts.append(f"{m['zer']}/{m['n']}({m['zer_pct']:.0f}%)")
                    parts[-1] = f"{parts[-1]:>12}"
                else:
                    parts.append(f"{'—':>12}")
            print("".join(parts))
        print()

    # === Table 7: Warnings ===
    print("=" * 80)
    print("TABLE 7: Warning Distribution (Condition A)")
    print("=" * 80)
    print(f"{'Model':28s} {'n':>4} {'total_w':>8} {'avg_w':>6} {'0warn%':>7}")
    print("-" * 55)
    for model in models_sorted:
        recs = by_model_cond[model].get("A", [])
        if not recs:
            continue
        m = compute_metrics(recs)
        zero_warn = sum(1 for r in recs if r["warning_count"] == 0)
        print(f"  {model:26s} {m['n']:4d} {m['warn']:8d} {m['warn_avg']:6.2f} {zero_warn/m['n']*100:6.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/run_003/prosody_final")
    args = parser.parse_args()

    records = load_all(args.dir)
    print(f"Loaded {len(records)} records from {args.dir}\n")
    print_report(records)


if __name__ == "__main__":
    main()
