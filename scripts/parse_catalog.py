#!/usr/bin/env python3
"""Parse cipai_catalog.json + exam docx into experiment-ready artifacts.

Reads the curated 144-cipai catalog and the ~297 内容 keywords from
the "Shi-Ci's Last Exam" docx, then produces:

  data/cipai_list.json   — flat list of all 144 entries with metadata
  data/cipai_sets.json   — split-name → list-of-rule-names mapping
  data/prompts.jsonl     — generation prompts (1 keyword per cipai, baseline)

Keyword assignment: deterministic pseudo-random (seed=42) so each of the
144 cipai gets exactly 1 keyword drawn from the 297-keyword pool without
replacement (remaining 153 keywords are unused).

Usage:
    python scripts/parse_catalog.py \
        [--catalog data/cipai_catalog.json] \
        [--exam '/path/to/2026-02-27 Shi-Ci'\''s Last Exam.docx']
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_TO_SPLIT = {
    "长调_热门": "changdiao_popular",
    "长调_冷门": "changdiao_rare",
    "中调_热门": "zhongdiao_popular",
    "中调_冷门": "zhongdiao_rare",
    "小令_热门": "xiaoling_popular",
    "小令_冷门": "xiaoling_rare",
    "诗体": "regulated",
    "难点": "special",
}

SPLIT_ORDER = [
    "changdiao_popular",
    "changdiao_rare",
    "zhongdiao_popular",
    "zhongdiao_rare",
    "xiaoling_popular",
    "xiaoling_rare",
    "regulated",
    "special",
]

REGULATED_SPLITS = {"regulated"}


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------

def _extract_entries(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for cat in categories:
        for list_key in ("top50", "bottom50", "top5", "bottom5", "entries"):
            items = cat.get(list_key)
            if not items:
                continue
            for item in items:
                raw_category = item.get("category", cat["name"])
                split = CATEGORY_TO_SPLIT.get(raw_category)
                if not split:
                    raise ValueError(
                        f"Unknown category '{raw_category}' for cipai "
                        f"'{item.get('name')}'. Add a mapping to CATEGORY_TO_SPLIT."
                    )
                examples = item.get("examples", [])
                entries.append({
                    "name": item["name"],
                    "rule": item["rule"],
                    "split": split,
                    "char_count": item.get("char_count"),
                    "poem_count": item.get("poem_count", 0),
                    "example_count": len(examples),
                    "examples": [
                        {
                            "title": ex.get("title", ""),
                            "author": ex.get("author", ""),
                            "dynasty": ex.get("dynasty", ""),
                            "content": ex.get("content", ""),
                        }
                        for ex in examples
                    ],
                })
    return entries


def _build_sets(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    sets: Dict[str, List[str]] = {s: [] for s in SPLIT_ORDER}
    for entry in entries:
        sets[entry["split"]].append(entry["rule"])
    return sets


# ---------------------------------------------------------------------------
# Exam docx keyword extraction
# ---------------------------------------------------------------------------

def _extract_content_keywords(docx_path: str) -> List[Dict[str, str]]:
    from docx import Document

    doc = Document(docx_path)
    all_text = [(i, p.text.strip()) for i, p in enumerate(doc.paragraphs) if p.text.strip()]

    keywords: List[Dict[str, str]] = []
    current_section: Optional[str] = None
    current_group: Optional[str] = None
    current_cat: Optional[str] = None

    for _idx, text in all_text:
        if re.match(r"^(基础题|进阶题|附加题)", text):
            current_section = text.split()[0] if " " in text else text.split("～")[0].strip()
            continue
        if current_section != "基础题":
            continue
        if re.match(r"^(内容|格式|审美)\s+\d+$", text):
            current_group = text.split()[0]
            continue
        m_cat = re.match(r"^(\S+)\s+(\d+)$", text)
        if m_cat:
            current_cat = m_cat.group(1)
            continue
        m_kw = re.match(r"^【(.+?)】(.*)$", text)
        if m_kw and current_group == "内容":
            keywords.append({
                "keyword": m_kw.group(1),
                "description": m_kw.group(2).strip(),
                "category": current_cat or "",
            })
    return keywords


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _assign_keywords(
    entries: List[Dict[str, Any]],
    keywords: List[Dict[str, str]],
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Pseudo-random 1:1 assignment of keywords to cipai entries."""
    rng = random.Random(seed)
    pool = list(keywords)
    rng.shuffle(pool)
    assigned = pool[: len(entries)]
    if len(assigned) < len(entries):
        raise ValueError(
            f"Not enough keywords ({len(keywords)}) for entries ({len(entries)})"
        )
    return assigned


def _build_prompt_text(entry: Dict[str, Any], keyword: str) -> str:
    name = entry["name"]
    is_shi = entry["split"] in REGULATED_SPLITS
    if is_shi:
        form_clause = f"作{name}一首"
    else:
        form_clause = f"以《{name}》为词牌名填词一首"
    return f"围绕【{keyword}】主题，{form_clause}，符合中国古典诗词的格律规范。"


def _build_prompts(
    entries: List[Dict[str, Any]],
    keywords: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    assigned = _assign_keywords(entries, keywords)
    prompts: List[Dict[str, Any]] = []
    for entry, kw_info in zip(entries, assigned):
        keyword = kw_info["keyword"]
        prompts.append({
            "id": f"{entry['rule']}__{keyword}",
            "split": entry["split"],
            "cipai": entry["rule"],
            "cipai_name": entry["name"],
            "keyword": keyword,
            "keyword_category": kw_info.get("category", ""),
            "prompt": _build_prompt_text(entry, keyword),
        })
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default="data/cipai_catalog.json",
        help="Path to cipai_catalog.json",
    )
    parser.add_argument(
        "--exam",
        default="/Users/lyy0323/Downloads/2026-02-27 Shi-Ci's Last Exam.docx",
        help="Path to Shi-Ci's Last Exam docx",
    )
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load catalog ---
    with Path(args.catalog).open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    entries = _extract_entries(catalog["categories"])
    sets = _build_sets(entries)

    # --- Load keywords ---
    keywords = _extract_content_keywords(args.exam)
    print(f"Extracted {len(keywords)} 内容 keywords from exam docx")

    # --- Build prompts ---
    prompts = _build_prompts(entries, keywords)

    # --- Write cipai_list.json ---
    list_path = outdir / "cipai_list.json"
    with list_path.open("w", encoding="utf-8") as f:
        json.dump({
            "version": catalog.get("generated", "unknown"),
            "total": len(entries),
            "splits": {s: len(rules) for s, rules in sets.items()},
            "entries": entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote {list_path}  ({len(entries)} entries)")

    # --- Write cipai_sets.json ---
    sets_path = outdir / "cipai_sets.json"
    with sets_path.open("w", encoding="utf-8") as f:
        json.dump({
            "version": catalog.get("generated", "unknown"),
            "total": len(entries),
            "splits": sets,
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote {sets_path}  ({len(sets)} splits)")

    # --- Write prompts.jsonl ---
    prompts_path = outdir / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {prompts_path}  ({len(prompts)} prompts)")

    # --- Summary ---
    print(f"\nSplit summary:")
    for s in SPLIT_ORDER:
        n = len(sets[s])
        print(f"  {s:24s} {n:3d} cipai")
    print(f"  {'TOTAL':24s} {len(entries):3d} cipai  {len(prompts):3d} prompts")

    print(f"\nSample prompts:")
    seen = set()
    for p in prompts:
        if p["split"] not in seen:
            seen.add(p["split"])
            print(f"  [{p['split']}] {p['prompt']}")


if __name__ == "__main__":
    main()
