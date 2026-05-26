#!/usr/bin/env python3
"""Batch LLM-as-Judge literary quality scoring.

Groups poems by prompt (same cipai+keyword), sends 12 poems per judge call
with model names anonymized. Uses 3 judge models for cross-validation.

Usage:
    python scripts/posthoc_quality.py \
        --run results/run_002 \
        --conditions B C D1 D2 \
        --judges configs/model.claude-opus-4-6.json configs/model.gpt-5_4.json configs/model.deepseek-v4-pro.json \
        --concurrency 4
"""

import argparse
import concurrent.futures
import hashlib
import json
import glob
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longci_bench.models.openai_compatible import OpenAICompatibleClient
from longci_bench.evaluation.literary_quality import LITERARY_QUALITY_SYSTEM_PROMPT


CONCURRENCY = 4


def _load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _build_judge_prompt(poems: List[Dict]) -> str:
    """Build a judge prompt with anonymized poems."""
    lines = []
    lines.append(f"请对以下 {len(poems)} 首诗词逐一评分。每首诗词标注了编号、词牌/诗体和主题关键词。\n")

    for i, p in enumerate(poems):
        label = chr(65 + i) if i < 26 else f"#{i+1}"
        cipai = p.get("cipai_name") or p.get("cipai", "")
        keyword = p.get("keyword", "")
        text = p.get("text", "").strip()
        lines.append(f"### 作品 {label}")
        lines.append(f"词牌/诗体：{cipai}")
        lines.append(f"主题：{keyword}")
        lines.append(f"正文：\n{text}\n")

    lines.append("请按以下 JSON 格式输出，不要输出其他内容：")
    lines.append("```json")
    lines.append("[")
    for i, p in enumerate(poems):
        label = chr(65 + i) if i < 26 else f"#{i+1}"
        comma = "," if i < len(poems) - 1 else ""
        lines.append(f'  {{"id": "{label}", "fluency": {{"analysis": "...", "score": N}}, "coherence": {{"analysis": "...", "score": N}}, "poetic_quality": {{"analysis": "...", "score": N}}}}{comma}')
    lines.append("]")
    lines.append("```")
    lines.append("其中 score 为 1-5 整数。analysis 为 1-2 句简评。")

    return "\n".join(lines)


def _parse_judge_response(text: str, n_poems: int) -> Optional[List[Dict]]:
    """Parse judge response JSON array."""
    stripped = text.strip()
    # Extract JSON from markdown code block
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                stripped = part
                break

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        arr = json.loads(stripped[start:end + 1])
        if isinstance(arr, list) and len(arr) == n_poems:
            return arr
    except json.JSONDecodeError:
        pass
    return None


def _extract_scores(parsed: List[Dict]) -> List[Dict]:
    """Extract normalized scores from parsed judge response."""
    results = []
    for item in parsed:
        scores = {}
        for dim in ["fluency", "coherence", "poetic_quality"]:
            d = item.get(dim, {})
            if isinstance(d, dict):
                score = d.get("score")
                analysis = d.get("analysis", "")
            elif isinstance(d, (int, float)):
                score = d
                analysis = ""
            else:
                score = None
                analysis = ""
            if score is not None:
                scores[f"{dim}_score"] = max(1, min(5, int(round(float(score)))))
                scores[f"{dim}_analysis"] = str(analysis)
        results.append(scores)
    return results


def judge_batch(client: OpenAICompatibleClient, poems: List[Dict]) -> Optional[List[Dict]]:
    """Send a batch of poems to a judge model, return per-poem scores."""
    prompt = _build_judge_prompt(poems)
    try:
        resp = client.generate(prompt)
        parsed = _parse_judge_response(resp.text, len(poems))
        if parsed:
            return _extract_scores(parsed)
    except Exception as exc:
        pass
    return None


def collect_poems_by_prompt(run_dir: str, conditions: List[str]) -> Dict[str, List[Dict]]:
    """Group poems by prompt ID across all models and conditions."""
    by_prompt: Dict[str, List[Dict]] = defaultdict(list)

    for cond in conditions:
        for f in sorted(glob.glob(f"{run_dir}/*/{cond}.jsonl")):
            model = f.split("/")[-2]
            for line in open(f):
                r = json.loads(line.strip())
                text = r.get("text", "").strip()
                if not text:
                    continue
                prompt_id = r.get("id", "")
                by_prompt[prompt_id].append({
                    "id": r.get("id"),
                    "model": model,
                    "condition": cond,
                    "cipai": r.get("cipai", ""),
                    "cipai_name": r.get("cipai_name", ""),
                    "keyword": r.get("keyword", ""),
                    "text": text,
                    "split": r.get("split", ""),
                })

    return by_prompt


def build_batches(by_prompt: Dict[str, List[Dict]], batch_size: int = 12) -> List[List[Dict]]:
    """Build batches of poems. Group by prompt, pad to batch_size."""
    # Flatten all poems, shuffle within each prompt group for anonymity
    all_poems = []
    for prompt_id, poems in by_prompt.items():
        random.shuffle(poems)
        all_poems.extend(poems)

    # Split into batches of batch_size
    batches = []
    for i in range(0, len(all_poems), batch_size):
        batches.append(all_poems[i:i + batch_size])
    return batches


def run_judging(
    run_dir: str,
    conditions: List[str],
    judge_configs: List[str],
    concurrency: int = 4,
    batch_size: int = 12,
):
    by_prompt = collect_poems_by_prompt(run_dir, conditions)
    total_poems = sum(len(v) for v in by_prompt.values())
    print(f"Collected {total_poems} poems across {len(by_prompt)} prompts")

    batches = build_batches(by_prompt, batch_size)
    print(f"Split into {len(batches)} batches of ≤{batch_size}")

    output_dir = Path(run_dir) / "quality_scores"
    output_dir.mkdir(exist_ok=True)

    for judge_cfg_path in judge_configs:
        judge_config = _load_json(judge_cfg_path)
        judge_name = judge_config.get("name", judge_config.get("model", "?"))
        output_file = output_dir / f"{judge_name}.jsonl"

        # Load existing scores
        existing = set()
        if output_file.exists():
            for line in open(output_file):
                try:
                    r = json.loads(line.strip())
                    key = f"{r.get('model')}_{r.get('condition')}_{r.get('id')}"
                    existing.add(key)
                except:
                    pass

        client = OpenAICompatibleClient(
            model=str(judge_config["model"]),
            endpoint=str(judge_config["endpoint"]),
            api_key=judge_config.get("api_key") or os.environ.get("OPENAI_COMPATIBLE_API_KEY"),
            api_keys=judge_config.get("api_keys"),
            temperature=0.0,
            max_tokens=8192,
            timeout=judge_config.get("timeout", 120),
            system_prompt=LITERARY_QUALITY_SYSTEM_PROMPT,
        )

        # Filter batches to only include poems not yet scored
        filtered_batches = []
        for batch in batches:
            needed = [p for p in batch if f"{p['model']}_{p['condition']}_{p['id']}" not in existing]
            if needed:
                filtered_batches.append(needed)

        remaining = sum(len(b) for b in filtered_batches)
        print(f"\n  Judge: {judge_name} — {remaining} poems in {len(filtered_batches)} batches (skip {total_poems - remaining} done)")

        done = 0
        lock = threading.Lock()

        def process_batch(batch):
            nonlocal done
            scores = judge_batch(client, batch)
            if scores and len(scores) == len(batch):
                with lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for poem, score in zip(batch, scores):
                            record = {
                                "id": poem["id"],
                                "model": poem["model"],
                                "condition": poem["condition"],
                                "cipai": poem["cipai"],
                                "cipai_name": poem["cipai_name"],
                                "judge": judge_name,
                                **score,
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done += len(batch)
                    if done % 48 == 0:
                        print(f"    {done}/{remaining}...", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(process_batch, filtered_batches))

        print(f"    Done: {done} scored → {output_file}")

    print("\nAll judges done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--conditions", nargs="+", default=["B", "C", "D1", "D2"])
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    run_judging(args.run, args.conditions, args.judges, args.concurrency, args.batch_size)


if __name__ == "__main__":
    main()
