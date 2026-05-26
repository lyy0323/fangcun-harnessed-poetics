#!/usr/bin/env python3
"""Posthoc similarity analysis: detect plagiarism and measure originality.

Reads generated JSONL records, queries /api/search/similar for each poem,
and appends similarity metrics to the records.

Usage:
    python scripts/posthoc_similarity.py \
        --input results/run_002/gpt-5.4/C.jsonl \
        [--all results/run_002]  # process all JSONL files under a run dir
"""

from __future__ import annotations
import os
import argparse
import json
import glob
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.parse import urlencode, quote

BASE_URL = os.environ.get("FANGCUN_SHI_URL", "http://localhost:8901")
LIMIT_PER_SENTENCE = 5
RATE_SLEEP = 0.3  # seconds between requests


def search_similar(text: str, limit: int = LIMIT_PER_SENTENCE) -> List[Dict[str, Any]]:
    """Call /api/search/similar and return results."""
    if not text or len(text.strip()) < 4:
        return []
    url = f"{BASE_URL}/api/search/similar?q={quote(text)}&limit={limit}"
    req = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])
    except Exception:
        return []


def analyze_similarity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute similarity metrics from search results.

    Leniency: ignore matches where the query sentence is <= 4 chars
    (common 4-char phrases like 世路茫茫 are idiomatic, not plagiarism).
    """
    if not results:
        return {
            "max_similarity": None,
            "mean_similarity": None,
            "plagiarism_sentences": 0,
            "high_similarity_count": 0,
            "plagiarism_flag": False,
            "originality_score": None,
            "similar_details": [],
        }

    # Filter out short query sentences (4 chars or fewer)
    def _query_len(r: Dict) -> int:
        qs = r.get("query_sentence") or r.get("text") or ""
        return sum(1 for ch in qs if "一" <= ch <= "鿿")

    scored = [r for r in results if _query_len(r) > 4]
    all_scores = [r.get("score", 0) for r in scored] if scored else [0]

    max_sim = max(all_scores)
    mean_sim = sum(all_scores) / len(all_scores)

    plagiarism_sentences = 0
    high_sim_count = 0
    details = []

    for r in scored:
        score = r.get("score", 0)
        if score >= 0.8:
            high_sim_count += 1
        if score >= 1.0:
            poems = r.get("poems", [])
            has_classical_author = any(
                p.get("dynasty", "") not in ("", "当代", "现代", "近现代")
                for p in poems
            )
            if has_classical_author:
                plagiarism_sentences += 1

        if score >= 0.6:
            poems = r.get("poems", [])
            top_poem = poems[0] if poems else {}
            details.append({
                "query": r.get("query_sentence", ""),
                "match": r.get("text", ""),
                "score": score,
                "author": top_poem.get("author", ""),
                "title": top_poem.get("title", ""),
                "dynasty": top_poem.get("dynasty", ""),
            })

    return {
        "max_similarity": round(max_sim, 3),
        "mean_similarity": round(mean_sim, 3),
        "plagiarism_sentences": plagiarism_sentences,
        "high_similarity_count": high_sim_count,
        "plagiarism_flag": plagiarism_sentences > 0,
        "originality_score": round(1 - mean_sim, 3),
        "similar_details": details[:10],
    }


def process_file(path: str) -> None:
    """Process a single JSONL file: add similarity metrics to each record."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return

    model = records[0].get("model", "?")
    condition = records[0].get("condition", "?")
    has_text = sum(1 for r in records if r.get("text", "").strip())
    already_done = sum(1 for r in records if r.get("max_similarity") is not None)

    if already_done >= has_text:
        print(f"  {path}: already done ({already_done}/{has_text}), skip")
        return

    print(f"  {path}: {has_text} poems to analyze ({already_done} already done)")

    processed = 0
    for i, r in enumerate(records):
        text = r.get("text", "").strip()
        if not text or r.get("max_similarity") is not None:
            continue

        results = search_similar(text)
        metrics = analyze_similarity(results)
        r.update(metrics)
        processed += 1

        if processed % 10 == 0:
            print(f"    {processed}/{has_text - already_done}...", flush=True)

        time.sleep(RATE_SLEEP)

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    all_max = [r["max_similarity"] for r in records if r.get("max_similarity") is not None]
    all_orig = [r["originality_score"] for r in records if r.get("originality_score") is not None]
    plag = sum(1 for r in records if r.get("plagiarism_flag"))
    high = sum(r.get("high_similarity_count", 0) for r in records)
    avg_orig = sum(all_orig) / len(all_orig) if all_orig else 0

    print(f"    Done: {processed} processed, plagiarism={plag}, "
          f"high_sim_sentences={high}, avg_originality={avg_orig:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Single JSONL file to process")
    parser.add_argument("--all", help="Process all JSONL files under this directory")
    args = parser.parse_args()

    if args.input:
        process_file(args.input)
    elif args.all:
        files = sorted(glob.glob(f"{args.all}/*/*.jsonl"))
        print(f"Found {len(files)} JSONL files under {args.all}")
        for f in files:
            process_file(f)
    else:
        parser.error("Specify --input or --all")


if __name__ == "__main__":
    main()
