#!/usr/bin/env python3
"""LLM-as-Judge ranking-based literary quality scoring.

For each prompt:
  Task 1 (leaderboard): Rank all models' Condition A poems → base scores
  Task 2 (cross-condition): Rank A/B/C per model → calibrated B/C scores
  Task 3 (ablation): Rank A/D1/D2 for gpt-5.4 and DS-V4 → calibrated D scores

Usage:
    python scripts/posthoc_ranking.py \
        --run results/run_003 --run2 results/run_002 \
        --judges configs/model.claude-opus-4-6.json configs/model.gpt-5_4.json configs/model.deepseek-v4-pro.json \
        --concurrency 8
"""

from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, glob, os, sys, threading, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from longci_bench.models.openai_compatible import OpenAICompatibleClient
from longci_bench.evaluation.literary_quality import LITERARY_QUALITY_SYSTEM_PROMPT

DIMENSIONS = ["fluency", "coherence", "poetic_quality"]
LEADERBOARD_SCORES = [100, 90, 80, 70, 60, 50, 40, 30]
CONDITION_SCORES = [100, 80, 60]
ABLATION_MODELS = {"gpt-5.4", "DeepSeek-V4-Pro"}

JUDGE_SYSTEM = (
    "你是中国古典诗词审美评鉴专家。你的任务是对同一题目下的多首作品进行比较排序。\n"
    "评价时只考虑文学审美，不考虑格律合规性。格律（平仄、押韵、字数）已由机器指标单独评判，请完全忽略格律问题，只关注文学品质。\n"
    "不要因为作品可能是机器生成的就给予宽容或苛刻的偏见。\n\n"
    "三个评分维度：\n"
    "1. 流畅度（fluency）：字词搭配是否精到，句法是否流畅自然\n"
    "2. 连贯性（coherence）：主题是否贯穿，上下片/起承转合是否层次分明\n"
    "3. 意境（poetic_quality）：艺术感染力，情景交融，是否有余味\n"
)


def _load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _shuffle_seed(prompt_id: str, judge_name: str) -> int:
    return int(hashlib.md5(f"{prompt_id}:{judge_name}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Collect poems
# ---------------------------------------------------------------------------

def collect_poems(run_a: str, run_bcd: str) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Returns: {prompt_id: {condition: {model: record}}}"""
    data: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: defaultdict(dict))

    # Condition A from run_003
    for f in glob.glob(f"{run_a}/*/A.jsonl"):
        model = f.split("/")[-2]
        for line in open(f):
            r = json.loads(line.strip())
            if r.get("text", "").strip() and not r.get("error_message"):
                data[r["id"]]["A"][model] = r

    # B/C/D from run_002
    for cond in ["B", "C", "D1", "D2"]:
        for f in glob.glob(f"{run_bcd}/*/{cond}.jsonl"):
            model = f.split("/")[-2]
            for line in open(f):
                r = json.loads(line.strip())
                if r.get("text", "").strip() and not r.get("error_message"):
                    pid = r["id"]
                    if pid not in data[pid][cond] or not data[pid][cond].get(model):
                        data[pid][cond][model] = r

    return data


# ---------------------------------------------------------------------------
# Build ranking tasks
# ---------------------------------------------------------------------------

def build_tasks(data: Dict, judge_name: str) -> List[Dict]:
    tasks = []

    for prompt_id, by_cond in data.items():
        a_poems = by_cond.get("A", {})
        if not a_poems:
            continue

        sample = list(a_poems.values())[0]
        cipai_name = sample.get("cipai_name", "")
        keyword = sample.get("keyword", "")

        # Task 1: Leaderboard (all models, condition A)
        if len(a_poems) >= 2:
            items = [(model, rec) for model, rec in a_poems.items()]
            seed = _shuffle_seed(prompt_id, judge_name)
            import random
            rng = random.Random(seed)
            rng.shuffle(items)
            labels = [chr(65 + i) for i in range(len(items))]
            tasks.append({
                "task_type": "leaderboard",
                "prompt_id": prompt_id,
                "cipai_name": cipai_name,
                "keyword": keyword,
                "candidates": [{"label": l, "model": m, "condition": "A", "text": r["text"]}
                               for l, (m, r) in zip(labels, items)],
                "label_to_model": {l: m for l, (m, _) in zip(labels, items)},
            })

        # Task 2: Cross-condition (per model, A vs B vs C)
        for model in a_poems:
            conds_available = {}
            for c in ["A", "B", "C"]:
                rec = by_cond.get(c, {}).get(model)
                if rec:
                    conds_available[c] = rec
            if len(conds_available) >= 2:
                items = list(conds_available.items())
                seed = _shuffle_seed(f"{prompt_id}:{model}", judge_name)
                rng = random.Random(seed)
                rng.shuffle(items)
                labels = [chr(65 + i) for i in range(len(items))]
                tasks.append({
                    "task_type": "cross_condition",
                    "prompt_id": prompt_id,
                    "cipai_name": cipai_name,
                    "keyword": keyword,
                    "model": model,
                    "candidates": [{"label": l, "condition": c, "text": r["text"]}
                                   for l, (c, r) in zip(labels, items)],
                    "label_to_condition": {l: c for l, (c, _) in zip(labels, items)},
                })

        # Task 3: Ablation (gpt-5.4 and DS-V4, A vs D1 vs D2)
        for model in ABLATION_MODELS:
            if model not in a_poems:
                continue
            conds_available = {}
            for c in ["A", "D1", "D2"]:
                rec = by_cond.get(c, {}).get(model)
                if rec:
                    conds_available[c] = rec
            if len(conds_available) >= 2:
                items = list(conds_available.items())
                seed = _shuffle_seed(f"{prompt_id}:{model}:ablation", judge_name)
                rng = random.Random(seed)
                rng.shuffle(items)
                labels = [chr(65 + i) for i in range(len(items))]
                tasks.append({
                    "task_type": "ablation",
                    "prompt_id": prompt_id,
                    "cipai_name": cipai_name,
                    "keyword": keyword,
                    "model": model,
                    "candidates": [{"label": l, "condition": c, "text": r["text"]}
                                   for l, (c, r) in zip(labels, items)],
                    "label_to_condition": {l: c for l, (c, _) in zip(labels, items)},
                })

    return tasks


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def build_judge_prompt(task: Dict) -> str:
    candidates = task["candidates"]
    n = len(candidates)
    lines = [
        f"请对以下 {n} 首作品，在三个维度上分别从高到低排序，并给出每个维度的排序理由。",
        f"",
        f"词牌/诗体：{task['cipai_name']}",
        f"主题：{task['keyword']}",
        "",
    ]
    for c in candidates:
        lines.append(f"### 作品 {c['label']}")
        lines.append(c["text"])
        lines.append("")

    valid_labels = [c["label"] for c in candidates]
    lines.append("请输出一个 JSON 对象（不要输出其他内容）：")
    lines.append("{")
    lines.append(f'  "fluency": {json.dumps(valid_labels)},')
    lines.append(f'  "fluency_reason": "分析每首作品在流畅度上的优劣...",')
    lines.append(f'  "coherence": {json.dumps(valid_labels)},')
    lines.append(f'  "coherence_reason": "分析每首作品在连贯性上的优劣...",')
    lines.append(f'  "poetic_quality": {json.dumps(valid_labels)},')
    lines.append(f'  "poetic_quality_reason": "分析每首作品在意境上的优劣..."')
    lines.append("}")
    lines.append(f"其中每个维度的数组从最好到最差排列，使用作品编号（{'/'.join(valid_labels)}）。")

    return "\n".join(lines)


def parse_judge_response(text: str, valid_labels: List[str]) -> Optional[Dict]:
    stripped = text.strip()
    if "```" in stripped:
        for part in stripped.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                stripped = part
                break
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None

    result = {}
    for dim in DIMENSIONS:
        ranking = d.get(dim)
        reason = d.get(f"{dim}_reason", "")
        if not isinstance(ranking, list):
            return None
        ranking_clean = [l for l in ranking if l in valid_labels]
        if len(ranking_clean) != len(valid_labels):
            return None
        result[dim] = ranking_clean
        result[f"{dim}_reason"] = str(reason)
    return result


def call_judge(client: OpenAICompatibleClient, task: Dict) -> Optional[Dict]:
    prompt = build_judge_prompt(task)
    valid_labels = [c["label"] for c in task["candidates"]]
    try:
        resp = client.generate(prompt)
        return parse_judge_response(resp.text, valid_labels)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def rank_to_score(rank: int, n: int, score_table: List[int]) -> int:
    if rank < len(score_table):
        return score_table[rank]
    return max(score_table[-1] - 10 * (rank - len(score_table) + 1), 10)


def compute_scores(tasks: List[Dict], rankings: List[Optional[Dict]]) -> List[Dict]:
    """Convert rankings to scored records."""
    records = []

    # First pass: collect leaderboard scores for calibration
    leaderboard_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for task, ranking in zip(tasks, rankings):
        if not ranking or task["task_type"] != "leaderboard":
            continue
        label_to_model = task["label_to_model"]
        n = len(task["candidates"])
        for dim in DIMENSIONS:
            order = ranking[dim]
            for rank_idx, label in enumerate(order):
                model = label_to_model[label]
                score = rank_to_score(rank_idx, n, LEADERBOARD_SCORES)
                leaderboard_scores[model][dim].append(score)

    # Compute per-model average leaderboard scores (for fallback)
    model_avg_scores: Dict[str, Dict[str, float]] = {}
    for model, dim_scores in leaderboard_scores.items():
        model_avg_scores[model] = {dim: sum(s) / len(s) for dim, s in dim_scores.items() if s}

    # Second pass: emit all records
    for task, ranking in zip(tasks, rankings):
        if not ranking:
            continue

        n = len(task["candidates"])
        prompt_id = task["prompt_id"]

        if task["task_type"] == "leaderboard":
            label_to_model = task["label_to_model"]
            for dim in DIMENSIONS:
                order = ranking[dim]
                for rank_idx, label in enumerate(order):
                    model = label_to_model[label]
                    score = rank_to_score(rank_idx, n, LEADERBOARD_SCORES)
                    records.append({
                        "prompt_id": prompt_id,
                        "model": model,
                        "condition": "A",
                        "task": "leaderboard",
                        "dimension": dim,
                        "rank": rank_idx + 1,
                        "score": score,
                        "n_candidates": n,
                        "reason": ranking.get(f"{dim}_reason", ""),
                    })

        elif task["task_type"] in ("cross_condition", "ablation"):
            label_to_cond = task["label_to_condition"]
            model = task["model"]
            score_table = CONDITION_SCORES

            for dim in DIMENSIONS:
                order = ranking[dim]
                # Get base score for calibration
                # Try per-prompt leaderboard score first
                base = None
                for t2, r2 in zip(tasks, rankings):
                    if r2 and t2["task_type"] == "leaderboard" and t2["prompt_id"] == prompt_id:
                        lb_model_map = t2["label_to_model"]
                        lb_order = r2[dim]
                        for ri, lb_label in enumerate(lb_order):
                            if lb_model_map[lb_label] == model:
                                base = rank_to_score(ri, len(lb_order), LEADERBOARD_SCORES)
                                break
                        break

                if base is None:
                    base = model_avg_scores.get(model, {}).get(dim, 70)

                for rank_idx, label in enumerate(order):
                    cond = label_to_cond[label]
                    factor = rank_to_score(rank_idx, n, score_table)
                    calibrated = round(min(100, factor * base / 100), 1)
                    records.append({
                        "prompt_id": prompt_id,
                        "model": model,
                        "condition": cond,
                        "task": task["task_type"],
                        "dimension": dim,
                        "rank": rank_idx + 1,
                        "factor": factor,
                        "base_score": base,
                        "score": calibrated,
                        "n_candidates": n,
                        "reason": ranking.get(f"{dim}_reason", ""),
                    })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ranking(run_a: str, run_bcd: str, judge_configs: List[str], concurrency: int = 8):
    data = collect_poems(run_a, run_bcd)
    print(f"Collected {len(data)} prompts")

    output_dir = Path(run_a) / "quality_rankings"
    output_dir.mkdir(exist_ok=True)

    for judge_cfg_path in judge_configs:
        run_single_judge(data, judge_cfg_path, output_dir, concurrency)

    print("\nAll judges done.")


def run_single_judge(data: Dict, judge_cfg_path: str, output_dir: Path, concurrency: int = 8):
    judge_config = _load_json(judge_cfg_path)
    judge_name = judge_config.get("name", judge_config.get("model", "?"))
    output_file = output_dir / f"{judge_name}.jsonl"

    tasks = build_tasks(data, judge_name)

    # Split into leaderboard first, then rest
    lb_tasks = [t for t in tasks if t["task_type"] == "leaderboard"]
    other_tasks = [t for t in tasks if t["task_type"] != "leaderboard"]

    # Load existing task keys to skip
    existing_task_keys = set()
    if output_file.exists():
        for line in open(output_file):
            try:
                r = json.loads(line.strip())
                existing_task_keys.add(f"{r.get('prompt_id')}:{r.get('task')}:{r.get('model','all')}:{r.get('dimension')}")
            except:
                pass

    client = OpenAICompatibleClient(
        model=str(judge_config["model"]),
        endpoint=str(judge_config["endpoint"]),
        api_key=judge_config.get("api_key"),
        api_keys=judge_config.get("api_keys"),
        temperature=0.0,
        max_tokens=4096,
        timeout=judge_config.get("timeout", 120),
        system_prompt=JUDGE_SYSTEM,
    )

    lock = threading.Lock()
    done = [0]

    # Phase 1: Leaderboard (need all results for calibration fallback)
    lb_needed = [t for t in lb_tasks if not _task_done(t, existing_task_keys)]
    print(f"\n  Judge: {judge_name} — Phase 1: {len(lb_needed)}/{len(lb_tasks)} leaderboard tasks")

    lb_rankings = {}  # prompt_id → ranking

    def process_lb(task):
        result = call_judge(client, task)
        if result:
            records = _leaderboard_to_records(task, result)
            with lock:
                lb_rankings[task["prompt_id"]] = (task, result)
                with open(output_file, "a", encoding="utf-8") as f:
                    for r in records:
                        r["judge"] = judge_name
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"    {done[0]}/{len(lb_needed)}...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(process_lb, lb_needed))

    # Also load previously completed leaderboard results
    if output_file.exists():
        for line in open(output_file):
            try:
                r = json.loads(line.strip())
                if r.get("task") == "leaderboard" and r.get("judge") == judge_name:
                    pid = r["prompt_id"]
                    # We only need scores for fallback, reconstruct minimally
                    pass
            except:
                pass

    # Compute per-model average leaderboard scores for fallback
    model_lb_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    if output_file.exists():
        for line in open(output_file):
            try:
                r = json.loads(line.strip())
                if r.get("task") == "leaderboard" and r.get("judge") == judge_name:
                    model_lb_scores[r["model"]][r["dimension"]].append(r["score"])
            except:
                pass
    model_avg = {m: {d: sum(s)/len(s) for d, s in ds.items()} for m, ds in model_lb_scores.items()}

    # Build per-prompt leaderboard lookup
    prompt_lb: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    if output_file.exists():
        for line in open(output_file):
            try:
                r = json.loads(line.strip())
                if r.get("task") == "leaderboard" and r.get("judge") == judge_name:
                    prompt_lb[r["prompt_id"]][r["model"]][r["dimension"]] = r["score"]
            except:
                pass

    # Phase 2: Cross-condition + ablation (stream each result)
    other_needed = [t for t in other_tasks if not _task_done(t, existing_task_keys)]
    print(f"  Judge: {judge_name} — Phase 2: {len(other_needed)}/{len(other_tasks)} condition/ablation tasks")
    done[0] = 0

    def process_other(task):
        result = call_judge(client, task)
        if result:
            records = _condition_to_records(task, result, prompt_lb, model_avg)
            with lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    for r in records:
                        r["judge"] = judge_name
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"    {done[0]}/{len(other_needed)}...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(process_other, other_needed))

    total_lines = sum(1 for _ in open(output_file)) if output_file.exists() else 0
    print(f"  Judge: {judge_name} — done. {total_lines} records in {output_file}")


def _task_done(task: Dict, existing_keys: set) -> bool:
    pid = task["prompt_id"]
    tt = task["task_type"]
    model = task.get("model", "all")
    for dim in DIMENSIONS:
        if f"{pid}:{tt}:{model}:{dim}" not in existing_keys:
            return False
    return True


def _leaderboard_to_records(task: Dict, ranking: Dict) -> List[Dict]:
    records = []
    label_to_model = task["label_to_model"]
    n = len(task["candidates"])
    for dim in DIMENSIONS:
        order = ranking[dim]
        for rank_idx, label in enumerate(order):
            model = label_to_model[label]
            score = rank_to_score(rank_idx, n, LEADERBOARD_SCORES)
            records.append({
                "prompt_id": task["prompt_id"],
                "model": model, "condition": "A", "task": "leaderboard",
                "dimension": dim, "rank": rank_idx + 1, "score": score,
                "n_candidates": n, "reason": ranking.get(f"{dim}_reason", ""),
            })
    return records


def _condition_to_records(task: Dict, ranking: Dict,
                          prompt_lb: Dict, model_avg: Dict) -> List[Dict]:
    records = []
    label_to_cond = task["label_to_condition"]
    model = task["model"]
    n = len(task["candidates"])
    for dim in DIMENSIONS:
        order = ranking[dim]
        base = (prompt_lb.get(task["prompt_id"], {}).get(model, {}).get(dim)
                or model_avg.get(model, {}).get(dim, 70))
        for rank_idx, label in enumerate(order):
            cond = label_to_cond[label]
            factor = rank_to_score(rank_idx, n, CONDITION_SCORES)
            calibrated = round(factor * base / 100, 1)
            records.append({
                "prompt_id": task["prompt_id"],
                "model": model, "condition": cond, "task": task["task_type"],
                "dimension": dim, "rank": rank_idx + 1,
                "factor": factor, "base_score": base, "score": calibrated,
                "n_candidates": n, "reason": ranking.get(f"{dim}_reason", ""),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run dir for Condition A (run_003)")
    parser.add_argument("--run2", required=True, help="Run dir for B/C/D (run_002)")
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    run_ranking(args.run, args.run2, args.judges, args.concurrency)


if __name__ == "__main__":
    main()
