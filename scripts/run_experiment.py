#!/usr/bin/env python3
"""Batch experiment runner with per-model concurrency and adaptive throttling.

Output layout:
  {outdir}/{model}/
    {condition}.jsonl        — slim summary records
    traces/{condition}/      — full tool traces (tool conditions only)

Usage:
    python scripts/run_experiment.py \
        --models configs/model.glm.json configs/model.qwen.json \
        --conditions A B C D1 D2 \
        --prompts data/prompts.jsonl \
        --outdir results/run_001 \
        --concurrency 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longci_bench.postprocess import normalize_poem_output, validate_output

CONDITION_TEMPLATES = {
    "A":  "configs/zero_shot/templates/condition_a_full_tool.json",
    "B":  "configs/zero_shot/templates/condition_b_description_only.json",
    "C":  "configs/zero_shot/templates/condition_c_baseline.json",
    "D1": "configs/zero_shot/templates/ablation_d1_no_template.json",
    "D2": "configs/zero_shot/templates/ablation_d2_no_validation.json",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_prompts(path: str) -> List[Dict[str, Any]]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts

def _model_name(model_config_path: str) -> str:
    config = _load_json(model_config_path)
    name = config.get("name") or config.get("model") or ""
    if name:
        return str(name).replace("/", "_").replace(" ", "_")
    return Path(model_config_path).stem.replace("model.", "")

def _safe_filename(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

# ---------------------------------------------------------------------------
# Token / prosody extraction (same as before)
# ---------------------------------------------------------------------------

def _extract_usage(metadata: Mapping[str, Any]) -> Dict[str, int]:
    usage = metadata.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }

def _extract_prosody_stats(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "prosody_valid": None, "error_count": None, "warning_count": None,
        "tone_errors": None, "rhyme_errors": None, "closest_rule": None,
    }
    fv = metadata.get("final_validation")
    if isinstance(fv, Mapping):
        _fill_prosody(stats, fv)
        return stats
    for round_item in reversed(metadata.get("tool_trace", [])):
        if not isinstance(round_item, Mapping):
            continue
        for obs in reversed(round_item.get("observations") or []):
            if not isinstance(obs, Mapping):
                continue
            call = obs.get("call", {})
            if isinstance(call, Mapping) and call.get("name") == "validate_meter":
                resp = obs.get("response", {})
                if isinstance(resp, Mapping):
                    _fill_prosody(stats, resp)
                    return stats
    return stats

def _etype(e: Mapping) -> str:
    return str(e.get("error_type") or e.get("type") or "")

def _fill_prosody(stats: Dict[str, Any], resp: Mapping) -> None:
    stats["prosody_valid"] = resp.get("passed") if "passed" in resp else resp.get("is_valid")
    issues = resp.get("issues", resp.get("errors", []))
    if isinstance(issues, list):
        stats["error_count"] = len(issues)
        stats["tone_errors"] = sum(1 for e in issues if isinstance(e, Mapping) and
                                   _etype(e) == "Tone")
        stats["rhyme_errors"] = sum(1 for e in issues if isinstance(e, Mapping) and
                                    _etype(e) == "Rhyme")
        stats["punctuation_errors"] = sum(1 for e in issues if isinstance(e, Mapping) and
                                          _etype(e) == "Punctuation")
        stats["structural_errors"] = (stats["error_count"] - stats["tone_errors"]
                                      - stats["rhyme_errors"] - stats["punctuation_errors"])
    metrics = resp.get("metrics", {})
    if isinstance(metrics, Mapping):
        stats.setdefault("warning_count", metrics.get("warning_count", 0))
        cr = metrics.get("closest_rule") or resp.get("closest_rule")
        if isinstance(cr, Mapping):
            stats["closest_rule"] = cr.get("name")
        elif isinstance(cr, str):
            stats["closest_rule"] = cr
    responses = resp.get("responses")
    if isinstance(responses, Mapping) and "passed" not in resp:
        for sub in responses.values():
            if isinstance(sub, Mapping):
                _fill_prosody(stats, sub)
                return

def _posthoc_validate(text: str, cipai: str, fangcun_api_config: str) -> Dict[str, Any]:
    from longci_bench.tools.fangcun_api import build_fangcun_api_tools
    from longci_bench.schemas import ToolRequest
    tools = build_fangcun_api_tools(fangcun_api_config)
    vm = next((t for t in tools if t.name == "validate_meter"), None)
    if not vm or not text.strip():
        return {"prosody_valid": None, "error_count": None, "warning_count": None,
                "tone_errors": None, "rhyme_errors": None, "closest_rule": None}
    is_ci = not any(cipai.startswith(p) for p in ("五绝", "七绝", "五律", "七律"))
    # Use cipai name prefix (e.g. "水调歌头") to allow any variant (格一/格二/格三)
    # but still constrain to the correct cipai
    cipai_prefix = cipai.split("_")[0] if "_" in cipai else cipai
    req = ToolRequest(tool_name="validate_meter", text=text, cipai="",
                      metadata={"genre": "Ci" if is_ci else "Shi",
                                "include_punctuation": is_ci,
                                "rule_name": cipai_prefix})
    try:
        resp = vm.run(req)
        stats: Dict[str, Any] = {}
        _fill_prosody(stats, resp.to_dict())
        stats.setdefault("warning_count", resp.metadata.get("warning_count", 0))
        return stats
    except Exception as exc:
        return {"prosody_valid": None, "error_count": None, "warning_count": None,
                "tone_errors": None, "rhyme_errors": None, "closest_rule": None,
                "validation_error": str(exc)[:200]}

# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------

def _generate_one(
    prompt_record: Dict[str, Any],
    template: Dict[str, Any],
    model_config_path: str,
    fangcun_api_config: str,
    trace_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from longci_bench.models import OpenAICompatibleClient
    from longci_bench.models.tool_calling import ToolCallingRunner
    from longci_bench.tools import ToolRegistry
    from longci_bench.tools.fangcun_api import build_fangcun_api_tools

    prompt_text = prompt_record["prompt"]
    cipai = prompt_record["cipai"]
    prompt_id = prompt_record["id"]
    condition = template.get("name", "")
    model_config = _load_json(model_config_path)
    model_name_str = model_config.get("name") or model_config.get("model") or ""

    # Template can override max_tokens (e.g. B=32768, C=8192)
    max_tokens = template.get("max_tokens_override") or model_config.get("max_tokens", 4096)

    client = OpenAICompatibleClient(
        model=str(model_config["model"]),
        endpoint=str(model_config["endpoint"]),
        api_key=model_config.get("api_key") or os.environ.get("OPENAI_COMPATIBLE_API_KEY"),
        api_keys=model_config.get("api_keys"),
        temperature=model_config.get("temperature", 0.7),
        max_tokens=max_tokens,
        timeout=model_config.get("timeout", 180),
        max_retries=3,
        retry_backoff=5.0,
    )

    registry = ToolRegistry()
    no_tools = template.get("no_tools", False)
    if not no_tools:
        api_tools = build_fangcun_api_tools(fangcun_api_config)
        toolset_path = template.get("toolset_config")
        if toolset_path:
            full_path = str((Path("configs/zero_shot/templates") / toolset_path).resolve())
            include = set(_load_json(full_path).get("include", []))
            api_tools = [t for t in api_tools if t.name in include]
        for tool in api_tools:
            registry.register(tool)

    system_prompt = template.get("system_prompt", "")
    max_rounds = template.get("max_tool_rounds", 0)

    t0 = time.time()
    if no_tools or max_rounds == 0:
        response = client.generate(
            f"{system_prompt}\n\n{prompt_text}" if system_prompt else prompt_text,
        )
    else:
        runner = ToolCallingRunner(
            model=client, registry=registry,
            max_tool_rounds=max_rounds,
            system_prompt=system_prompt,
            max_tool_calls_per_tool=template.get("max_tool_calls_per_tool"),
        )
        response = runner.run(prompt_text, cipai=cipai)
    elapsed = time.time() - t0

    raw_text = response.text
    text = normalize_poem_output(raw_text)
    output_valid, output_issues = validate_output(text)
    meta = response.metadata if hasattr(response, "metadata") else {}

    top_usage = _extract_usage(meta)
    trace = meta.get("tool_trace", [])

    tool_call_count = 0
    tool_rounds = 0
    per_tool_counts: Dict[str, int] = {}
    for ri in trace:
        if isinstance(ri, Mapping) and ri.get("observations"):
            tool_rounds += 1
            for obs in ri["observations"]:
                if isinstance(obs, Mapping):
                    cn = (obs.get("call") or {}).get("name", "")
                    if cn:
                        per_tool_counts[cn] = per_tool_counts.get(cn, 0) + 1
                        tool_call_count += 1

    prosody_stats = _extract_prosody_stats(meta) if trace else {}
    if prosody_stats.get("prosody_valid") is None:
        prosody_stats = _posthoc_validate(text, cipai, fangcun_api_config)

    trace_file_rel = None
    if trace and trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
        tp = trace_dir / (_safe_filename(prompt_id) + ".json")
        trace_file_rel = str(tp.relative_to(trace_dir.parents[1]))
        with tp.open("w", encoding="utf-8") as f:
            json.dump({"id": prompt_id, "cipai": cipai, "model": model_name_str,
                        "condition": condition, "elapsed_s": round(elapsed, 2),
                        "tool_trace": trace, "messages": meta.get("messages", []),
                        "raw_text": raw_text, "final_validation": meta.get("final_validation")},
                       f, ensure_ascii=False, indent=2)

    record: Dict[str, Any] = {
        "id": prompt_id, "split": prompt_record.get("split", ""),
        "cipai": cipai, "cipai_name": prompt_record.get("cipai_name", ""),
        "prompt": prompt_text, "text": text,
        "model": model_name_str, "condition": condition,
        "keyword": prompt_record.get("keyword", ""),
        "elapsed_s": round(elapsed, 2),
        "tool_rounds": tool_rounds, "tool_call_count": tool_call_count,
        "per_tool_counts": per_tool_counts or None,
        "finish_reason": meta.get("finish_reason", "completed"),
        "prompt_tokens": top_usage["prompt_tokens"],
        "completion_tokens": top_usage["completion_tokens"],
        "total_tokens": top_usage["total_tokens"],
        "output_valid": output_valid,
        "output_issues": output_issues or None,
        **prosody_stats,
    }
    if trace_file_rel:
        record["trace_file"] = trace_file_rel
    return record

# ---------------------------------------------------------------------------
# Concurrent model runner with adaptive throttling
# ---------------------------------------------------------------------------

class _ResultWriter:
    """Thread-safe JSONL appender."""
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

def _run_model_condition(
    model_config_path: str,
    cond_key: str,
    prompts: List[Dict[str, Any]],
    outdir: Path,
    fangcun_api_config: str,
    initial_concurrency: int = 16,
    resume: bool = True,
):
    model_name = _model_name(model_config_path)
    template_path = CONDITION_TEMPLATES.get(cond_key)
    if not template_path:
        print(f"  SKIP unknown condition: {cond_key}")
        return
    template = _load_json(template_path)
    has_tools = not template.get("no_tools", False)

    model_dir = outdir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    out_file = model_dir / f"{cond_key}.jsonl"
    trace_dir = model_dir / "traces" / cond_key if has_tools else None

    existing_ids: set = set()
    if resume and out_file.exists():
        keep_lines: list = []
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("text", "").strip() and rec.get("finish_reason") != "error":
                        existing_ids.add(rec["id"])
                        keep_lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    pass
        if keep_lines:
            with out_file.open("w", encoding="utf-8") as f:
                f.writelines(keep_lines)

    pending = [p for p in prompts if p["id"] not in existing_ids]
    if not pending:
        print(f"  {model_name}/{cond_key}: all {len(existing_ids)} done, skip")
        return
    if existing_ids:
        print(f"  {model_name}/{cond_key}: resume, {len(existing_ids)} done, {len(pending)} remaining")

    writer = _ResultWriter(out_file)
    concurrency = initial_concurrency
    completed = 0
    errors_429 = 0
    total = len(pending)
    lock = threading.Lock()

    def _do_one(prompt_record: Dict[str, Any]) -> None:
        nonlocal completed, errors_429, concurrency
        pid = prompt_record.get("cipai_name", prompt_record["id"][:20])
        try:
            record = _generate_one(
                prompt_record, template, model_config_path,
                fangcun_api_config, trace_dir=trace_dir,
            )
            writer.write(record)
            with lock:
                completed += 1
                errs = record.get("error_count")
                tok = record.get("total_tokens", 0)
                print(f"  [{completed}/{total}] {model_name}/{cond_key}/{pid}  "
                      f"{record['elapsed_s']}s {errs}err {tok}tok")
        except Exception as exc:
            is_429 = "429" in str(exc)
            with lock:
                if is_429:
                    errors_429 += 1
                completed += 1
            if is_429:
                raise  # will be retried
            error_record = {
                "id": prompt_record["id"], "split": prompt_record.get("split", ""),
                "cipai": prompt_record["cipai"], "cipai_name": prompt_record.get("cipai_name", ""),
                "prompt": prompt_record["prompt"], "text": "",
                "model": model_name, "condition": cond_key,
                "keyword": prompt_record.get("keyword", ""),
                "elapsed_s": 0, "tool_rounds": 0, "tool_call_count": 0,
                "finish_reason": "error",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prosody_valid": None, "error_count": None, "warning_count": None,
                "tone_errors": None, "rhyme_errors": None, "closest_rule": None,
                "error_message": str(exc)[:500],
            }
            writer.write(error_record)
            print(f"  [{completed}/{total}] {model_name}/{cond_key}/{pid}  ERROR: {str(exc)[:80]}")

    # Adaptive concurrency: start at initial, halve on 429 wave, retry failed
    retry_queue: List[Dict[str, Any]] = []

    while pending or retry_queue:
        batch = pending or retry_queue
        if retry_queue and not pending:
            pending = retry_queue
            retry_queue = []
            if errors_429 > 2:
                concurrency = max(1, concurrency // 2)
                print(f"  {model_name}/{cond_key}: throttling → concurrency={concurrency} (429 count={errors_429})")
                errors_429 = 0
                time.sleep(10)

        current_batch = pending[:concurrency * 4]  # feed the pool
        pending = pending[len(current_batch):]

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_do_one, p): p for p in current_batch}
            for f in concurrent.futures.as_completed(futures):
                prompt_rec = futures[f]
                exc = f.exception()
                if exc and "429" in str(exc):
                    retry_queue.append(prompt_rec)

    print(f"  {model_name}/{cond_key}: done ({completed} completed)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    model_configs: List[str],
    conditions: List[str],
    prompts_path: str,
    outdir: str,
    fangcun_api_config: str = "configs/tools.fangcun_api.json",
    concurrency: int = 16,
    resume: bool = True,
):
    prompts = _load_prompts(prompts_path)
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {len(model_configs)} models × {len(conditions)} conditions × {len(prompts)} prompts")
    print(f"Concurrency: {concurrency} per model, outdir: {outdir}\n")

    # Launch each model in parallel (each model has its own thread pool internally)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_configs)) as model_pool:
        model_futures = []
        for model_config_path in model_configs:
            for cond_key in conditions:
                f = model_pool.submit(
                    _run_model_condition,
                    model_config_path, cond_key, prompts, out_root,
                    fangcun_api_config, concurrency, resume,
                )
                model_futures.append((f, _model_name(model_config_path), cond_key))

        for f, mn, ck in model_futures:
            try:
                f.result()
            except Exception as exc:
                print(f"  {mn}/{ck} FATAL: {exc}")

    print(f"\nAll done. Results in {outdir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D1", "D2"],
                        choices=list(CONDITION_TEMPLATES.keys()))
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument("--outdir", default="results/run_001")
    parser.add_argument("--fangcun-api-config", default="configs/tools.fangcun_api.json")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_configs=args.models,
        conditions=args.conditions,
        prompts_path=args.prompts,
        outdir=args.outdir,
        fangcun_api_config=args.fangcun_api_config,
        concurrency=args.concurrency,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
