#!/usr/bin/env python3
"""FANGCUN Poetry Toolchain CLI — local proxy for the FANGCUN API.

Runs as a local HTTP server that proxies requests to the FANGCUN API endpoints.
Designed to be compiled into a standalone binary with PyInstaller so that
API URLs are not exposed in source code.

Usage:
    ./fangcun-cli serve --port 8901
    ./fangcun-cli call validate_meter '{"poem_text":"...", "genre":"Ci"}'

Build:
    pip install pyinstaller
    pyinstaller --onefile tools/fangcun_cli.py -n fangcun-cli
"""

import argparse
import json
import sys
import http.server
import threading
from urllib import request as urllib_request
from urllib.parse import urlencode, urlparse, parse_qs

# Hardcoded API endpoints (compiled into binary)
BASE_URLS = {
    "write": "https://write.sjtuguoxue.space",
    "checker": "https://checker.sjtuguoxue.space",
    "shi": "https://shi.sjtuguoxue.space",
}

TOOL_ROUTES = {
    "validate_meter":      ("write",   "POST", "/api/validate_meter"),
    "validate_batch":      ("checker", "POST", "/api/validate_batch"),
    "char_lookup":         ("write",   "GET",  "/api/char/lookup"),
    "rhyme_lookup":        ("write",   "GET",  "/api/rhyme/lookup"),
    "rhyme_list":          ("write",   "GET",  "/api/rhyme/list"),
    "rules_list":          ("write",   "GET",  "/api/rules/list"),
    "dictionary_search":   ("write",   "GET",  "/api/dictionary/search"),
    "dictionary_allusion": ("write",   "GET",  "/api/dictionary/allusion"),
    "free_rhyme":          ("write",   "POST", "/api/free_rhyme"),
    "examples":            ("checker", "GET",  "/api/examples"),
    "search_text":         ("shi",     "GET",  "/api/search/text"),
    "search_similar":      ("shi",     "GET",  "/api/search/similar"),
}


def call_api(tool_name: str, params: dict) -> dict:
    if tool_name not in TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}"}

    base_key, method, path = TOOL_ROUTES[tool_name]
    base_url = BASE_URLS[base_key]

    if method == "GET":
        query = {k: str(v) if not isinstance(v, bool) else ("true" if v else "false")
                 for k, v in params.items() if v is not None}
        url = f"{base_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        req = urllib_request.Request(url, method="GET")
    else:
        url = f"{base_url}{path}"
        body = json.dumps(params, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")

    try:
        with urllib_request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP proxy: accepts requests at /api/{tool_name} and forwards to FANGCUN."""

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "api":
            tool_name = parts[1]
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            result = call_api(tool_name, params)
            self._respond(result)
        else:
            self._respond({"error": "Not found", "tools": list(TOOL_ROUTES.keys())}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "api":
            tool_name = parts[1]
            try:
                params = json.loads(body)
            except json.JSONDecodeError:
                params = {}
            result = call_api(tool_name, params)
            self._respond(result)
        else:
            self._respond({"error": "Not found"}, 404)

    def _respond(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def cmd_serve(args):
    print(f"FANGCUN proxy: http://localhost:{args.port}")
    print(f"Tools: {', '.join(TOOL_ROUTES.keys())}")
    print(f"Example: curl http://localhost:{args.port}/api/rules_list?genre=Ci&search=念奴娇")
    server = http.server.HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    server.serve_forever()


def cmd_call(args):
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {args.params}", file=sys.stderr)
        sys.exit(1)
    result = call_api(args.tool, params)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="FANGCUN Poetry Toolchain CLI")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Run as local HTTP proxy")
    serve_p.add_argument("--port", type=int, default=8901)

    call_p = sub.add_parser("call", help="Call a single tool")
    call_p.add_argument("tool", choices=list(TOOL_ROUTES.keys()))
    call_p.add_argument("params", help="JSON parameters")

    list_p = sub.add_parser("list", help="List available tools")

    args = parser.parse_args()
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "call":
        cmd_call(args)
    elif args.command == "list":
        for name, (base, method, path) in TOOL_ROUTES.items():
            print(f"  {method:4s} {name:24s} → {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
