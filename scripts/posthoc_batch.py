#!/usr/bin/env python3
"""Batch posthoc: similarity analysis + LLM literary quality scoring.

Processes 12 poems concurrently per batch.

Usage:
    python scripts/posthoc_batch.py --all results/run_002 --jobs sim
    python scripts/posthoc_batch.py --all results/run_002 --jobs quality
    python scripts/posthoc_batch.py --all results/run_002 --jobs sim,quality
"""

import argparse
import os
import concurrent.futures
import json
import glob
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List
from urllib import request as urllib_request
from urllib.parse import quote

SIM_BASE = os.environ.get("FANGCUN_SHI_URL", "http://localhost:8901")
SIM_LIMIT = 5
CONCURRENCY = 12
PLACEHOLDER_CHAR = "□"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def search_similar(text: str) -> List[Dict]:
    if not text or len(text.strip()) < 4:
        return []
    url = f"{SIM_BASE}/api/search/similar?q={quote(text)}&limit={SIM_LIMIT}"
    try:
        req = urllib_request.Request(url, method="GET")
        with urllib_request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")).get("results", [])
    except Exception:
        return []


def analyze_similarity(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {"max_similarity": None, "mean_similarity": None, "plagiarism_sentences": 0,
                "high_similarity_count": 0, "plagiarism_flag": False, "originality_score": None,
                "similar_details": []}

    def qlen(r):
        qs = r.get("query_sentence") or r.get("text") or ""
        return sum(1 for ch in qs if "一" <= ch <= "鿿")

    scored = [r for r in results if qlen(r) > 4]
    all_scores = [r.get("score", 0) for r in scored] if scored else [0]
    max_sim = max(all_scores)
    mean_sim = sum(all_scores) / len(all_scores)

    plagiarism_sentences = high_sim_count = 0
    details = []
    for r in scored:
        score = r.get("score", 0)
        if score >= 0.8:
            high_sim_count += 1
        if score >= 1.0:
            if any(p.get("dynasty", "") not in ("", "当代", "现代", "近现代") for p in r.get("poems", [])):
                plagiarism_sentences += 1
        if score >= 0.6:
            top = r.get("poems", [{}])[0] if r.get("poems") else {}
            details.append({"query": r.get("query_sentence", ""), "match": r.get("text", ""),
                            "score": score, "author": top.get("author", ""), "title": top.get("title", "")})

    return {"max_similarity": round(max_sim, 3), "mean_similarity": round(mean_sim, 3),
            "plagiarism_sentences": plagiarism_sentences, "high_similarity_count": high_sim_count,
            "plagiarism_flag": plagiarism_sentences > 0, "originality_score": round(1 - mean_sim, 3),
            "similar_details": details[:10]}


def process_sim(record: Dict) -> Dict:
    text = record.get("text", "").strip()
    if not text or record.get("max_similarity") is not None:
        return {}
    results = search_similar(text)
    return analyze_similarity(results)


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def process_file(path: str, jobs: List[str]):
    model = path.split("/")[-2]
    cond = path.split("/")[-1].replace(".jsonl", "")
    records = [json.loads(l) for l in open(path) if l.strip()]
    if not records:
        return

    # Filter records needing work
    if "sim" in jobs:
        need_sim = [(i, r) for i, r in enumerate(records)
                    if r.get("text", "").strip() and r.get("max_similarity") is None]
    else:
        need_sim = []

    total_work = len(need_sim)
    if total_work == 0:
        return

    print(f"  {model}/{cond}: {total_work} to process...", end="", flush=True)
    done = 0

    def do_one(idx_rec):
        nonlocal done
        idx, rec = idx_rec
        updates = {}
        if "sim" in jobs:
            updates.update(process_sim(rec))
        if updates:
            with _write_lock:
                records[idx].update(updates)
                done += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(do_one, need_sim))

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" done ({done} updated)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", required=True, help="Run directory")
    parser.add_argument("--jobs", default="sim", help="Comma-separated: sim,quality")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    global CONCURRENCY
    CONCURRENCY = args.concurrency
    jobs = [j.strip() for j in args.jobs.split(",")]

    files = sorted(glob.glob(f"{args.all}/*/*.jsonl"))
    print(f"Processing {len(files)} files, jobs={jobs}, concurrency={CONCURRENCY}")
    for f in files:
        process_file(f, jobs)
    print("\nAll done.")


if __name__ == "__main__":
    main()
