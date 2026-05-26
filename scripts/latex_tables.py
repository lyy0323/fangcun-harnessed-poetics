#!/usr/bin/env python3
"""Generate comprehensive LaTeX tables for the paper appendix.

Outputs 4 tables:
  1. Overall: all metrics × all conditions × all models
  2. Form category breakdown (A vs C)
  3. Ablation (D1/D2) with form category
  4. Literary quality + originality

Usage:
    python scripts/latex_tables.py \
        --run-a results_ours/run_003 \
        --run-bcd results_ours/run_002 \
        --prosody results_ours/prosody_final
"""

import argparse
import json
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

SPLIT_GROUPS = {
    "changdiao_popular": "Pop.CQ", "changdiao_rare": "Rare CQ",
    "zhongdiao_popular": "Pop.Short", "zhongdiao_rare": "Rare Short",
    "xiaoling_popular": "Pop.Short", "xiaoling_rare": "Rare Short",
    "regulated": "Reg.Verse", "special": "Special",
}
GROUP_ORDER = ["Pop.CQ", "Rare CQ", "Pop.Short", "Rare Short", "Reg.Verse", "Special"]
COND_ORDER = ["A", "B", "C", "D1", "D2"]
RENAME = {"Qwen3.5-Plus": "Qwen", "Qwen3.5-27B": "Qwen"}
QUALITY_DIMS = ["fluency", "coherence", "poetic_quality"]


def load_prosody(d):
    recs = []
    for f in sorted(glob.glob(f"{d}/*.jsonl")):
        cond = f.split("/")[-1].replace(".jsonl", "")
        for line in open(f):
            if line.strip():
                r = json.loads(line.strip())
                r["model"] = RENAME.get(r["model"], r["model"])
                r["condition"] = cond
                r["split_group"] = SPLIT_GROUPS.get(r.get("split", ""), "other")
                recs.append(r)
    return recs


def load_generation(run_a, run_bcd):
    recs = []
    for run, conds in [(run_a, ["A"]), (run_bcd, ["B", "C", "D1", "D2"])]:
        for c in conds:
            for f in sorted(glob.glob(f"{run}/*/{c}.jsonl")):
                if "prosody" in f or "quality" in f:
                    continue
                model = RENAME.get(f.split("/")[-2], f.split("/")[-2])
                for line in open(f):
                    r = json.loads(line.strip())
                    if r.get("text", "").strip() and not r.get("error_message"):
                        recs.append({
                            "model": model, "condition": c,
                            "split_group": SPLIT_GROUPS.get(r.get("split", ""), "other"),
                            "total_tokens": r.get("total_tokens", 0),
                            "elapsed_s": r.get("elapsed_s", 0),
                            "tool_rounds": r.get("tool_rounds", 0),
                            "tool_call_count": r.get("tool_call_count", 0),
                            "originality_score": r.get("originality_score"),
                            "plagiarism_flag": r.get("plagiarism_flag", False),
                        })
    return recs


def load_quality(quality_dir):
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_quality import compute_leaderboard, compute_calibrated_scores, DIMENSIONS

    all_recs = []
    for pattern in [f"{quality_dir}/quality_rankings/*.jsonl", f"{quality_dir}/*.jsonl"]:
        for f in sorted(glob.glob(pattern)):
            for line in open(f):
                if line.strip():
                    r = json.loads(line.strip())
                    r["model"] = RENAME.get(r["model"], r["model"])
                    all_recs.append(r)
        if all_recs:
            break
    if not all_recs:
        return {}

    by_judge = defaultdict(list)
    for r in all_recs:
        by_judge[r.get("judge", "?")].append(r)

    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for judge, jrecs in by_judge.items():
        lb = compute_leaderboard(jrecs)
        cal = compute_calibrated_scores(jrecs, lb)
        for model, dims in lb.items():
            model = RENAME.get(model, model)
            for dim in DIMENSIONS:
                scores[model]["A"][dim].append(dims.get(dim, 0))
        for model, conds in cal.items():
            model = RENAME.get(model, model)
            for cond, dims in conds.items():
                for dim, vals in dims.items():
                    scores[model][cond][dim].append(sum(vals) / len(vals))

    result = {}
    for model in scores:
        result[model] = {}
        for cond in scores[model]:
            da = {}
            for dim in DIMENSIONS:
                vals = scores[model][cond].get(dim, [])
                da[dim] = sum(vals) / len(vals) if vals else None
            vals = [v for v in da.values() if v is not None]
            da["avg"] = sum(vals) / len(vals) if vals else None
            result[model][cond] = da
    return result


def c(val, fmt=".1f", suffix=""):
    if val is None:
        return "—"
    return f"{val:{fmt}}{suffix}"


def b(val, all_vals, fmt=".1f", suffix="", higher=True):
    if val is None:
        return "—"
    valid = [v for v in all_vals if v is not None]
    s = f"{val:{fmt}}{suffix}"
    if valid and val == (max(valid) if higher else min(valid)) and len(valid) > 1:
        return f"\\textbf{{{s}}}"
    return s


def avg(recs, key):
    vals = [r.get(key) for r in recs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def zer(recs):
    n = len(recs)
    return sum(1 for r in recs if r["error_count"] == 0) / n * 100 if n else None


def cta(recs):
    nf = [r for r in recs if not r["has_fatal"]]
    if not nf:
        return None
    tc = sum(r.get("total_chars", 100) for r in nf)
    return (tc - sum(r["tone_errors"] for r in nf)) / tc * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", default="results_ours/run_003")
    parser.add_argument("--run-bcd", default="results_ours/run_002")
    parser.add_argument("--prosody", default="results_ours/prosody_final")
    parser.add_argument("--quality", default=None, help="Quality rankings dir (default: {run-a}/quality_rankings)")
    args = parser.parse_args()

    prosody = load_prosody(args.prosody)
    gen = load_generation(args.run_a, args.run_bcd)
    quality_dir = args.quality or args.run_a
    quality = load_quality(quality_dir)

    # Index
    pi = defaultdict(lambda: defaultdict(list))          # model → cond → [recs]
    pg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model → cond → group → [recs]
    for r in prosody:
        pi[r["model"]][r["condition"]].append(r)
        pg[r["model"]][r["condition"]][r["split_group"]].append(r)

    gi = defaultdict(lambda: defaultdict(list))
    for r in gen:
        gi[r["model"]][r["condition"]].append(r)

    models = sorted(set(r["model"] for r in prosody),
                    key=lambda m: -(zer(pi[m].get("A", [])) or 0))

    # =====================================================================
    # TABLE 1: Full Overall
    # =====================================================================
    print(r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\scriptsize
\caption{Complete experimental results. N = valid poems evaluated; ZER = zero-error rate (\%); CTA = character-level tonal accuracy (\%); Err = mean error count; Fatal = structural failure rate (\%); Warn = mean warnings; Rounds = mean tool-calling rounds; Calls = mean tool calls; Tok = mean tokens (k); Time = mean seconds per poem.}
\label{tab:full-overall}
\begin{tabular}{l|ccccc|ccccc|ccccc|ccccc|cc|cc}
\toprule
& \multicolumn{5}{c|}{N} & \multicolumn{5}{c|}{ZER (\%)} & \multicolumn{5}{c|}{CTA (\%)} & \multicolumn{5}{c|}{Avg Err} & \multicolumn{2}{c|}{Rounds (A)} & \multicolumn{2}{c}{Cost (A)} \\
\textbf{Model} & A & B & C & D1 & D2 & A & B & C & D1 & D2 & A & B & C & D1 & D2 & A & B & C & D1 & D2 & Rnd & Calls & Tok(k) & Time \\
\midrule""")

    for model in models:
        parts = [model]
        # N
        for co in COND_ORDER:
            recs = pi[model].get(co, [])
            parts.append(str(len(recs)) if recs else "—")
        # ZER
        for co in COND_ORDER:
            parts.append(c(zer(pi[model].get(co, [])), ".0f"))
        # CTA
        for co in COND_ORDER:
            parts.append(c(cta(pi[model].get(co, [])), ".1f"))
        # Avg Err
        for co in COND_ORDER:
            recs = pi[model].get(co, [])
            parts.append(c(sum(r["error_count"] for r in recs) / len(recs) if recs else None, ".1f"))
        # Rounds, Calls (A only)
        ga = gi[model].get("A", [])
        parts.append(c(avg(ga, "tool_rounds"), ".1f"))
        parts.append(c(avg(ga, "tool_call_count"), ".0f"))
        # Tok, Time (A only)
        tok = avg(ga, "total_tokens")
        parts.append(c(tok / 1000 if tok else None, ".1f"))
        parts.append(c(avg(ga, "elapsed_s"), ".0f"))

        print(" & ".join(parts) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")

    # =====================================================================
    # TABLE 2: Warnings + Fatal breakdown
    # =====================================================================
    print()
    print(r"""\begin{table*}[t]
\centering
\small
\caption{Error type breakdown and warning analysis. Fatal = poems with structural mismatch or $>$10 content errors (\%); Tone/Rhyme = total error counts; Warn = mean warnings per poem (2-gram + rhyme duplicate for cí, default for shī).}
\label{tab:error-breakdown}
\begin{tabular}{l|rrrrr|rrrrr|rrrrr}
\toprule
& \multicolumn{5}{c|}{Fatal (\%)} & \multicolumn{5}{c|}{Total Tone Errors} & \multicolumn{5}{c}{Mean Warnings} \\
\textbf{Model} & A & B & C & D1 & D2 & A & B & C & D1 & D2 & A & B & C & D1 & D2 \\
\midrule""")

    for model in models:
        parts = [model]
        for co in COND_ORDER:
            recs = pi[model].get(co, [])
            parts.append(c(sum(1 for r in recs if r["has_fatal"]) / len(recs) * 100 if recs else None, ".0f"))
        for co in COND_ORDER:
            recs = pi[model].get(co, [])
            parts.append(str(sum(r["tone_errors"] for r in recs)) if recs else "—")
        for co in COND_ORDER:
            recs = pi[model].get(co, [])
            parts.append(c(sum(r["warning_count"] for r in recs) / len(recs) if recs else None, ".2f"))
        print(" & ".join(parts) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")

    # =====================================================================
    # TABLE 3: Form category (A vs C)
    # =====================================================================
    print()
    print(r"""\begin{table*}[t]
\centering
\small
\caption{ZER (\%) by form category. Condition A (full tool access) vs Condition C (baseline). $\Delta$ columns show the improvement from tool augmentation.}
\label{tab:form-category}
\begin{tabular}{l|rrrrrr|rrrrrr|rrrrrr}
\toprule
& \multicolumn{6}{c|}{Condition A} & \multicolumn{6}{c|}{Condition C} & \multicolumn{6}{c}{$\Delta$ (A$-$C, pp)} \\
\textbf{Model} & """ + " & ".join(GROUP_ORDER) + " & " + " & ".join(GROUP_ORDER) + " & " + " & ".join(GROUP_ORDER) + r""" \\
\midrule""")

    for model in models:
        parts = [model]
        a_vals = []
        c_vals = []
        for grp in GROUP_ORDER:
            a_recs = pg[model].get("A", {}).get(grp, [])
            c_recs = pg[model].get("C", {}).get(grp, [])
            za = zer(a_recs)
            zc = zer(c_recs)
            a_vals.append(za)
            c_vals.append(zc)
            parts.append(c(za, ".0f"))
        for zc in c_vals:
            parts.append(c(zc, ".0f"))
        for za, zc in zip(a_vals, c_vals):
            if za is not None and zc is not None:
                parts.append(f"+{za - zc:.0f}")
            else:
                parts.append("—")
        print(" & ".join(parts) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")

    # =====================================================================
    # TABLE 4: Ablation
    # =====================================================================
    abl_models = [m for m in models if pi[m].get("D1") or pi[m].get("D2")]
    if abl_models:
        print()
        print(r"""\begin{table*}[t]
\centering
\small
\caption{Ablation study. D1 = template lookup removed (rules\_list, examples); D2 = prosody validation removed (validate\_meter). D1 form-category breakdown reveals which models rely on memorized vs.\ queried templates.}
\label{tab:ablation}
\begin{tabular}{l|rrr|rr|rrrrrr}
\toprule
& \multicolumn{3}{c|}{Overall ZER (\%)} & \multicolumn{2}{c|}{$\Delta$ (pp)} & \multicolumn{6}{c}{D1 ZER by Form Category (\%)} \\
\textbf{Model} & A & D1 & D2 & D1$-$A & D2$-$A & """ + " & ".join(GROUP_ORDER) + r""" \\
\midrule""")

        for model in abl_models:
            za = zer(pi[model].get("A", []))
            zd1 = zer(pi[model].get("D1", []))
            zd2 = zer(pi[model].get("D2", []))
            dd1 = f"{zd1 - za:+.0f}" if zd1 is not None and za is not None else "—"
            dd2 = f"{zd2 - za:+.0f}" if zd2 is not None and za is not None else "—"
            parts = [model, c(za, ".0f"), c(zd1, ".0f"), c(zd2, ".0f"), dd1, dd2]
            for grp in GROUP_ORDER:
                recs = pg[model].get("D1", {}).get(grp, [])
                parts.append(c(zer(recs), ".0f") if recs else "—")
            print(" & ".join(parts) + r" \\")

        print(r"""\bottomrule
\end{tabular}
\end{table*}""")

    # =====================================================================
    # TABLE 5: Literary Quality + Originality
    # =====================================================================
    print()
    print(r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\scriptsize
\caption{Literary quality (3-judge ranking average, capped at 100) and originality. Quality dimensions: Flu = fluency, Coh = coherence, Poe = poetic quality. Orig = originality score (1 $-$ mean bigram Jaccard to classical corpus, higher = more original). Plag = plagiarism rate (\%, exact match to pre-modern work on sentences $>$4 chars).}
\label{tab:quality-originality}
\begin{tabular}{l|rrrr|rrrr|rrrr|rrrr|rrrr|rr|rr}
\toprule
& \multicolumn{4}{c|}{Quality (A)} & \multicolumn{4}{c|}{Quality (B)} & \multicolumn{4}{c|}{Quality (C)} & \multicolumn{4}{c|}{Quality (D1)} & \multicolumn{4}{c|}{Quality (D2)} & \multicolumn{2}{c|}{Orig (A)} & \multicolumn{2}{c}{Orig (C)} \\
\textbf{Model} & Flu & Coh & Poe & Avg & Flu & Coh & Poe & Avg & Flu & Coh & Poe & Avg & Flu & Coh & Poe & Avg & Flu & Coh & Poe & Avg & Scr & Pl\% & Scr & Pl\% \\
\midrule""")

    for model in models:
        parts = [model]
        # Quality for each condition
        for co in ["A", "B", "C", "D1", "D2"]:
            qd = quality.get(model, {}).get(co, {})
            for dim in ["fluency", "coherence", "poetic_quality", "avg"]:
                parts.append(c(qd.get(dim), ".1f"))

        # Originality + plagiarism for A and C
        for co in ["A", "C"]:
            grecs = gi[model].get(co, [])
            orig_vals = [r["originality_score"] for r in grecs if r.get("originality_score") is not None]
            orig = sum(orig_vals) / len(orig_vals) if orig_vals else None
            parts.append(c(orig, ".3f"))
            plag_total = sum(1 for r in grecs if r.get("originality_score") is not None)
            plag_count = sum(1 for r in grecs if r.get("plagiarism_flag"))
            plag_pct = plag_count / plag_total * 100 if plag_total else None
            parts.append(c(plag_pct, ".1f"))

        print(" & ".join(parts) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


if __name__ == "__main__":
    main()
