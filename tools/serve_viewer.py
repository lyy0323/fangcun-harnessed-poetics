#!/usr/bin/env python3
"""Trace viewer server. Auto-discovers all runs/conditions and serves a viewer UI.

Usage:
    python tools/serve_viewer.py [--port 8765] [--results-root results]
"""

import argparse
import json
import glob
import os
import http.server
import urllib.parse
from pathlib import Path


RESULTS_ROOT = "results"
VIEWER_HTML = Path(__file__).parent / "trace_viewer.html"


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(VIEWER_HTML.read_bytes())
            return

        if path == "/api/runs":
            self._serve_json(self._list_runs())
            return

        if path == "/api/conditions":
            run = qs.get("run", [""])[0]
            self._serve_json(self._list_conditions(run))
            return

        if path == "/api/records":
            run = qs.get("run", [""])[0]
            condition = qs.get("condition", ["A"])[0]
            self._serve_json(self._load_records(run, condition))
            return

        if path == "/api/stats":
            run = qs.get("run", [""])[0]
            condition = qs.get("condition", ["A"])[0]
            self._serve_json(self._compute_stats(run, condition))
            return

        if path == "/api/trace":
            run = qs.get("run", [""])[0]
            condition = qs.get("condition", ["A"])[0]
            trace_id = qs.get("id", [""])[0]
            model = qs.get("model", [""])[0]
            self._serve_json(self._load_trace(run, condition, trace_id, model))
            return

        self.send_error(404)

    def _serve_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _list_runs(self):
        runs = []
        for d in sorted(glob.glob(f"{RESULTS_ROOT}/run_*")):
            if os.path.isdir(d):
                name = os.path.basename(d)
                conditions = set()
                for f in glob.glob(f"{d}/*/*.jsonl"):
                    cond = os.path.splitext(os.path.basename(f))[0]
                    conditions.add(cond)
                runs.append({"name": name, "conditions": sorted(conditions)})
        return runs

    def _list_conditions(self, run):
        run_dir = f"{RESULTS_ROOT}/{run}" if run else self._latest_run()
        conditions = {}
        for f in sorted(glob.glob(f"{run_dir}/*/*.jsonl")):
            cond = os.path.splitext(os.path.basename(f))[0]
            model = os.path.basename(os.path.dirname(f))
            if cond not in conditions:
                conditions[cond] = {"name": cond, "models": [], "total_records": 0}
            count = sum(1 for line in open(f) if line.strip())
            conditions[cond]["models"].append(model)
            conditions[cond]["total_records"] += count
        return list(conditions.values())

    def _load_records(self, run, condition):
        run_dir = f"{RESULTS_ROOT}/{run}" if run else self._latest_run()
        records = []
        for f in sorted(glob.glob(f"{run_dir}/*/{condition}.jsonl")):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return records

    def _load_trace(self, run, condition, trace_id, model):
        run_dir = f"{RESULTS_ROOT}/{run}" if run else self._latest_run()
        safe_id = trace_id.replace("/", "_").replace("\\", "_")
        patterns = [
            f"{run_dir}/{model}/traces/{condition}/{safe_id}.json",
            f"{run_dir}/*/traces/{condition}/{safe_id}.json",
        ]
        for pattern in patterns:
            for f in glob.glob(pattern):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
        return None

    def _latest_run(self):
        runs = sorted(glob.glob(f"{RESULTS_ROOT}/run_*"))
        return runs[-1] if runs else RESULTS_ROOT

    def _compute_stats(self, run, condition):
        records = self._load_records(run, condition)
        if not records:
            return {}

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

        # Per-model stats
        from collections import defaultdict
        by_model = defaultdict(list)
        for r in records:
            by_model[r.get("model", "?")].append(r)

        model_stats = {}
        for model, recs in sorted(by_model.items()):
            valid = [r for r in recs if r.get("text", "").strip() and not r.get("error_message")]
            n = len(valid)
            zer = sum(1 for r in valid if r.get("error_count", 999) == 0)
            avg_err = sum(r.get("error_count") or 0 for r in valid) / max(n, 1)
            avg_tok = sum(r.get("total_tokens") or 0 for r in valid) / max(n, 1)
            avg_time = sum(r.get("elapsed_s") or 0 for r in valid) / max(n, 1)
            fail = sum(1 for r in recs if r.get("error_message"))

            # By split group
            by_group = defaultdict(list)
            for r in valid:
                g = SPLIT_GROUPS.get(r.get("split", ""), "other")
                by_group[g].append(r)

            groups = {}
            for g in GROUP_ORDER:
                gr = by_group.get(g, [])
                if gr:
                    g_zer = sum(1 for r in gr if r.get("error_count", 999) == 0)
                    g_avg = sum(r.get("error_count") or 0 for r in gr) / len(gr)
                    groups[g] = {"n": len(gr), "zer": g_zer, "zer_pct": round(g_zer / len(gr) * 100, 1), "avg_err": round(g_avg, 1)}

            model_stats[model] = {
                "n": n, "fail": fail, "zer": zer,
                "zer_pct": round(zer / max(n, 1) * 100, 1),
                "avg_err": round(avg_err, 1),
                "avg_tokens": int(avg_tok),
                "avg_time": round(avg_time, 1),
                "groups": groups,
            }

        # Overall
        all_valid = [r for r in records if r.get("text", "").strip() and not r.get("error_message")]
        total_zer = sum(1 for r in all_valid if r.get("error_count", 999) == 0)
        return {
            "total_records": len(records),
            "total_valid": len(all_valid),
            "total_zer": total_zer,
            "total_zer_pct": round(total_zer / max(len(all_valid), 1) * 100, 1),
            "models": model_stats,
            "group_order": GROUP_ORDER,
        }

    def log_message(self, format, *args):
        pass


def main():
    global RESULTS_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    RESULTS_ROOT = args.results_root

    print(f"Trace Viewer: http://localhost:{args.port}")
    print(f"Results root: {RESULTS_ROOT}")
    server = http.server.HTTPServer(("127.0.0.1", args.port), ViewerHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
