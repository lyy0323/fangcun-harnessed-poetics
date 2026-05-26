#!/usr/bin/env python3
"""Offline smoke test — verifies core functionality without network calls.

Usage:
    python scripts/smoke_test_local.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_schemas():
    print("\n=== Schemas ===")
    from longci_bench.schemas import ToolRequest, ToolResponse, ModelResponse, GenerationRecord
    tr = ToolRequest(tool_name="test", text="hello", cipai="念奴娇")
    check("ToolRequest", tr.tool_name == "test" and tr.cipai == "念奴娇")
    resp = ToolResponse(tool_name="test", passed=True)
    check("ToolResponse", resp.passed is True)
    mr = ModelResponse(text="output")
    check("ModelResponse", mr.text == "output")


def test_postprocess():
    print("\n=== Postprocess ===")
    from longci_bench.postprocess import extract_boxed, normalize_poem_output, validate_output, has_placeholder
    check("extract_boxed single", extract_boxed(r"\boxed{test}") == "test")
    check("extract_boxed double", extract_boxed("\\\\boxed{test}") == "test")
    check("extract_boxed none", extract_boxed("no box") == "no box")
    check("normalize think", "think" not in normalize_poem_output("<think>reasoning</think>\\boxed{poem}"))
    check("placeholder detect", has_placeholder("明月□时有"))
    check("placeholder clean", not has_placeholder("明月几时有"))
    ok, issues = validate_output("明月几时有，把酒问青天。")
    check("validate_output ok", ok)
    ok2, issues2 = validate_output("明月□时有")
    check("validate_output placeholder", not ok2 and "contains_placeholder" in issues2)
    ok3, issues3 = validate_output("")
    check("validate_output empty", not ok3)


def test_paper_metrics():
    print("\n=== Paper Metrics ===")
    from longci_bench.evaluation.paper_metrics import compute_paper_metrics, aggregate_paper_metrics, WARNING_EXEMPT_CIPAI

    # Clean pass
    m = compute_paper_metrics({"is_valid": True, "errors": [], "display_segments": [{"text_chars": list("X"*50)}],
                                "rhyme_positions": [9,19], "closest_rule": {"char_count": 50, "name": "test"}}, "X"*50)
    check("metrics clean", m["zer"] == 1.0 and m["cta"] == 1.0 and m["error_class"] == "clean")

    # Fatal (length)
    m2 = compute_paper_metrics({"errors": [{"error_type": "Length", "position": -1}],
                                 "display_segments": [{"text_chars": list("X"*45)}],
                                 "closest_rule": {"char_count": 50}}, "X"*45)
    check("metrics fatal length", m2["has_fatal"] and m2["cta"] == 0.0)

    # Fatal (>10 tone errors)
    m3 = compute_paper_metrics({"errors": [{"error_type": "Tone"} for _ in range(12)],
                                 "display_segments": [{"text_chars": list("X"*50)}],
                                 "closest_rule": {"char_count": 50}}, "X"*50)
    check("metrics fatal >10 errors", m3["has_fatal"])

    # Warning exempt
    check("warning exempt cipai", "调笑令_钦谱_格一" in WARNING_EXEMPT_CIPAI)

    # Aggregate
    agg = aggregate_paper_metrics([m, m2])
    check("aggregate excludes fatal from CTA", agg["n_non_fatal"] == 1)


def test_literary_quality():
    print("\n=== Literary Quality ===")
    from longci_bench.evaluation.literary_quality import LITERARY_QUALITY_SYSTEM_PROMPT, _literary_quality_messages
    check("system prompt exists", len(LITERARY_QUALITY_SYSTEM_PROMPT) > 500)
    msgs = _literary_quality_messages("明月几时有", cipai="水调歌头", keyword="中秋")
    check("message construction", len(msgs) == 2 and msgs[0]["role"] == "system")
    user = json.loads(msgs[1]["content"])
    check("user payload", user["task"] == "score_literary_quality" and user["theme_keyword"] == "中秋")


def test_tool_schema():
    print("\n=== Tool Schema ===")
    example_path = Path("configs/tools.fangcun_api.example.json")
    if not example_path.exists():
        check("example config exists", False, "configs/tools.fangcun_api.example.json not found")
        return
    d = json.loads(example_path.read_text())
    check("has tools array", isinstance(d.get("tools"), list) and len(d["tools"]) >= 10)
    check("has base_urls", isinstance(d.get("base_urls"), dict))
    names = {t["name"] for t in d["tools"]}
    check("validate_meter present", "validate_meter" in names)
    check("char_lookup present", "char_lookup" in names)
    check("rules_list present", "rules_list" in names)


def test_tool_calling_runner():
    print("\n=== Tool Calling Runner ===")
    from longci_bench.models.dummy import DummyModelClient
    from longci_bench.models.tool_calling import ToolCallingRunner
    from longci_bench.tools import ToolRegistry
    from longci_bench.tools.prosody import ProsodyTool

    registry = ToolRegistry()
    registry.register(ProsodyTool())
    client = DummyModelClient()
    runner = ToolCallingRunner(client, registry, max_tool_rounds=2)
    resp = runner.run("test prompt", cipai="test")
    check("runner produces response", resp.text is not None)


def test_data_files():
    print("\n=== Data Files ===")
    check("cipai_catalog.json", Path("data/cipai_catalog.json").exists())
    check("cipai_list.json", Path("data/cipai_list.json").exists())
    check("prompts.jsonl", Path("data/prompts.jsonl").exists())

    prompts = [json.loads(l) for l in open("data/prompts.jsonl") if l.strip()]
    check("144 prompts", len(prompts) == 144)
    check("prompt has id", all("id" in p for p in prompts))
    check("prompt has cipai", all("cipai" in p for p in prompts))

    catalog = json.loads(open("data/cipai_catalog.json").read())
    check("catalog has categories", len(catalog.get("categories", [])) >= 4)


def test_analysis_scripts():
    print("\n=== Analysis Scripts ===")
    # Create tiny synthetic prosody data
    with tempfile.TemporaryDirectory() as tmpdir:
        pdir = Path(tmpdir)
        (pdir / "A.jsonl").write_text(json.dumps({
            "model": "test", "condition": "A", "cipai": "念奴娇_龙谱_格一",
            "cipai_name": "念奴娇", "split": "changdiao_popular",
            "error_count": 0, "tone_errors": 0, "rhyme_errors": 0,
            "punctuation_errors": 0, "has_fatal": False, "warning_count": 0,
            "closest_rule": "念奴娇_龙谱_格一", "total_chars": 100,
        }, ensure_ascii=False) + "\n")

        # Test analyze_prosody imports
        sys.path.insert(0, "scripts")
        try:
            from analyze_prosody import load_all, compute_metrics
            recs = load_all(str(pdir))
            check("analyze_prosody loads", len(recs) == 1)
            m = compute_metrics(recs)
            check("analyze_prosody metrics", m["zer_pct"] == 100.0)
        except Exception as e:
            check("analyze_prosody", False, str(e))


def test_no_secrets():
    print("\n=== Secret Scan ===")
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "sk-[a-zA-Z0-9]", "--include=*.py", "--include=*.json", "--include=*.md",
         "src/", "scripts/", "configs/", "docs/"],
        capture_output=True, text=True, cwd="."
    )
    lines = [l for l in result.stdout.strip().split("\n") if l and "example" not in l.lower() and "YOUR" not in l]
    check("no API keys in source", len(lines) == 0, f"Found: {lines[:3]}")

    result2 = subprocess.run(
        ["grep", "-rn", "sjtuguoxue\\|xaminim\\|minimaxi", "--include=*.py",
         "src/", "scripts/"],
        capture_output=True, text=True, cwd="."
    )
    lines2 = [l for l in result2.stdout.strip().split("\n") if l and "smoke_test_local" not in l]
    check("no hardcoded URLs in Python", len(lines2) == 0, f"Found: {lines2[:3]}")


if __name__ == "__main__":
    os.chdir(str(Path(__file__).resolve().parents[1]))

    test_schemas()
    test_postprocess()
    test_paper_metrics()
    test_literary_quality()
    test_tool_schema()
    test_tool_calling_runner()
    test_data_files()
    test_analysis_scripts()
    test_no_secrets()

    print(f"\n{'=' * 40}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    print("All smoke tests passed.")
