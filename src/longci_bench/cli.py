"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

from .evaluation.runner import evaluate_jsonl
from .models import OpenAICompatibleClient, ToolCallingRunner
from .schemas import ToolRequest
from .tools import (
    LOCAL_POETICS_BASE_TOOL_NAMES,
    LOCAL_POETICS_EXTRA_TOOL_NAMES,
    LOCAL_POETICS_LEXICAL_TOOL_NAMES,
    REPEAT_REPAIR_TOOL_NAME,
    STYLE_TOOL_NAMES,
    build_default_registry,
)


LOCAL_TOOL_NAMES = list(
    dict.fromkeys(
        ["prosody", "repetition", "imagery"]
        + LOCAL_POETICS_BASE_TOOL_NAMES
        + LOCAL_POETICS_EXTRA_TOOL_NAMES
        + LOCAL_POETICS_LEXICAL_TOOL_NAMES
        + STYLE_TOOL_NAMES
        + [REPEAT_REPAIR_TOOL_NAME]
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="longci-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    tool = sub.add_parser("tool", help="Run one local tool.")
    tool.add_argument("tool_name")
    tool.add_argument("--text", required=True)
    tool.add_argument("--cipai")
    tool.add_argument("--metadata-json", help="Optional JSON object passed to ToolRequest.metadata.")
    tool.add_argument("--specs")
    tool.add_argument("--imagery")
    tool.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    _add_toolset_args(tool)
    _add_poetics_args(tool)
    _add_style_tool_args(tool)

    schema = sub.add_parser("tool-schema", help="Print model-facing tool schemas for the configured toolset.")
    schema.add_argument("--specs")
    schema.add_argument("--imagery")
    schema.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    schema.add_argument("--output")
    _add_toolset_args(schema)
    _add_poetics_args(schema)
    _add_style_tool_args(schema)

    contract = sub.add_parser("tool-contract", help="Print callable input/output contracts for the configured tools.")
    contract.add_argument("--specs")
    contract.add_argument("--imagery")
    contract.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    contract.add_argument("--output")
    _add_toolset_args(contract)
    _add_poetics_args(contract)
    _add_style_tool_args(contract)

    tool_call = sub.add_parser("tool-call", help="Run one local tool from a JSON callable payload.")
    tool_call.add_argument("--input-json", help="JSON object containing tool_name/name plus input arguments.")
    tool_call.add_argument("--input", help="JSON file containing one tool call payload, or '-' for stdin.")
    tool_call.add_argument("--tool-name", help="Tool name to use when the JSON payload only contains arguments.")
    tool_call.add_argument("--specs")
    tool_call.add_argument("--imagery")
    tool_call.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    tool_call.add_argument("--output")
    _add_toolset_args(tool_call)
    _add_poetics_args(tool_call)
    _add_style_tool_args(tool_call)

    tool_batch = sub.add_parser("tool-batch", help="Run multiple local tool calls from JSON.")
    tool_batch.add_argument("--input-json", help="JSON array, or object with a calls array.")
    tool_batch.add_argument("--input", help="JSON file containing calls, or '-' for stdin.")
    tool_batch.add_argument("--specs")
    tool_batch.add_argument("--imagery")
    tool_batch.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    tool_batch.add_argument("--output")
    _add_toolset_args(tool_batch)
    _add_poetics_args(tool_batch)
    _add_style_tool_args(tool_batch)

    generate = sub.add_parser("generate", help="Run zero-shot generation with an OpenAI-compatible API.")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--cipai")
    generate.add_argument("--output")
    generate.add_argument("--text-only", action="store_true")
    generate.add_argument("--max-tool-rounds", type=int)
    generate.add_argument("--max-tool-calls-per-tool", type=int)
    generate.add_argument(
        "--require-final-prosody-pass",
        action="store_true",
        help="Keep revising final text until local prosody validation passes or tool rounds are exhausted.",
    )
    generate.add_argument("--no-tools", action="store_true", help="Call the model without tool schemas for baseline comparison.")
    generate.add_argument("--dry-run", action="store_true", help="Print the first model request payload without calling the API.")
    generate.add_argument("--system-prompt")
    generate.add_argument("--specs")
    generate.add_argument("--imagery")
    generate.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    _add_model_args(generate)
    _add_toolset_args(generate)
    _add_poetics_args(generate)
    _add_style_tool_args(generate)

    suite = sub.add_parser("generate-suite", help="Run one prompt across multiple zero-shot templates.")
    suite.add_argument("--prompt", required=True)
    suite.add_argument("--cipai")
    suite.add_argument("--output")
    suite.add_argument("--text-only", action="store_true", help="Print only per-template final texts, not JSON metadata.")
    suite.add_argument("--dry-run", action="store_true", help="Print request payloads without calling the API.")
    suite.add_argument(
        "--template",
        action="append",
        dest="templates",
        help="Zero-shot template JSON path. Repeat to select multiple templates.",
    )
    suite.add_argument(
        "--template-dir",
        default="configs/zero_shot/templates",
        help="Directory of template JSON files used when --template is omitted.",
    )
    suite.add_argument("--max-tool-rounds", type=int)
    suite.add_argument("--max-tool-calls-per-tool", type=int)
    suite.add_argument("--require-final-prosody-pass", action="store_true")
    suite.add_argument("--system-prompt")
    suite.add_argument("--specs")
    suite.add_argument("--imagery")
    suite.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    _add_model_args(suite)
    _add_poetics_args(suite)
    _add_style_tool_args(suite)

    eval_parser = sub.add_parser("eval", help="Evaluate a JSONL generation file.")
    eval_parser.add_argument("--input", required=True)
    eval_parser.add_argument("--specs")
    eval_parser.add_argument("--imagery")
    eval_parser.add_argument("--tool-config", help="Optional external tool config. Matching names override local tools.")
    eval_parser.add_argument("--metric-config", help="JSON config for API-backed entity and fluency metrics.")
    eval_parser.add_argument(
        "--metric-model-config",
        help="OpenAI-compatible tool-fitting/evaluation model config used by metric API backends; can be passed without --metric-config for a quick API smoke test.",
    )
    _add_toolset_args(eval_parser)
    _add_poetics_args(eval_parser)
    _add_style_tool_args(eval_parser)
    eval_parser.add_argument("--output")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tool":
        registry = _registry_from_args(args)
        response = registry.run(
            ToolRequest(
                args.tool_name,
                args.text,
                cipai=args.cipai,
                metadata=_metadata_from_json(args.metadata_json),
            )
        )
        print_json(response.to_dict())
        return 0

    if args.command == "tool-schema":
        registry = _registry_from_args(args)
        payload = {"tools": registry.model_tool_schemas()}
        if args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "tool-contract":
        registry = _registry_from_args(args)
        payload = registry.contract()
        if args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "tool-call":
        registry = _registry_from_args(args)
        call_payload = _load_json_payload(args.input_json, args.input, "--input-json/--input")
        if not isinstance(call_payload, dict):
            raise SystemExit("tool-call input must be a JSON object")
        payload = registry.run_mapping(call_payload, tool_name=args.tool_name).to_dict()
        if args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "tool-batch":
        registry = _registry_from_args(args)
        calls = _load_tool_batch(args.input_json, args.input)
        payload = {"responses": registry.call_many(calls)}
        if args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "generate":
        if args.dry_run:
            payload = _dry_run_payload(args)
            if args.output:
                _write_json(args.output, payload)
            else:
                print_json(payload)
            return 0
        try:
            response = _generate_response_from_args(args)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        payload = response.to_dict()
        if args.text_only:
            sys.stdout.write(response.text)
            if response.text and not response.text.endswith("\n"):
                sys.stdout.write("\n")
        elif args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "generate-suite":
        if args.text_only and args.dry_run:
            raise SystemExit("--text-only cannot be combined with --dry-run for generate-suite")
        payload = _generate_suite(args)
        if args.text_only:
            text = _suite_text(payload)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                sys.stdout.write(text)
        elif args.output:
            _write_json(args.output, payload)
        else:
            print_json(payload)
        return 0

    if args.command == "eval":
        report = evaluate_jsonl(
            args.input,
            spec_path=args.specs,
            imagery_path=args.imagery,
            tool_config_path=args.tool_config,
            toolset_config_path=args.toolset_config,
            poetics_path=getattr(args, "poetics_path", None),
            poetics_config_dir=getattr(args, "poetics_config_dir", None),
            poetics_data_dir=getattr(args, "poetics_data_dir", None),
            poetics_genre=getattr(args, "poetics_genre", None),
            poetics_rhyme_book=getattr(args, "poetics_rhyme_book", None),
            poetics_ensure_longpu=False if getattr(args, "poetics_no_longpu", False) else None,
            fangcun_path=args.fangcun_path,
            fangcun_config_dir=args.fangcun_config_dir,
            fangcun_data_dir=args.fangcun_data_dir,
            fangcun_genre=args.fangcun_genre or "Ci",
            fangcun_rhyme_book=args.fangcun_rhyme_book or "Cilinzhengyun",
            fangcun_ensure_longpu=not args.fangcun_no_longpu,
            metric_config_path=args.metric_config,
            metric_model_config_path=args.metric_model_config,
            style_tool_config_path=args.style_tool_config,
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        else:
            print_json(report)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def print_json(payload) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _suite_text(payload: Dict) -> str:
    parts = []
    for result in payload.get("results", []):
        template_name = str(result.get("template") or "<template>")
        response = result.get("response") or {}
        text = str(response.get("text") or "")
        parts.append(f"### {template_name}\n{text}".rstrip())
    return "\n\n".join(parts).rstrip() + "\n"


def _generate_response_from_args(args: argparse.Namespace):
    model = _model_from_args(args)
    if getattr(args, "no_tools", False):
        return model.generate(args.prompt, tools=[], metadata={"cipai": args.cipai})
    registry = _registry_from_args(args)
    return ToolCallingRunner(
        model,
        registry,
        max_tool_rounds=_max_tool_rounds_from_args(args),
        system_prompt=model.system_prompt,
        max_tool_calls_per_tool=_max_tool_calls_per_tool_from_args(args),
        require_final_prosody_pass=_require_final_prosody_pass_from_args(args),
        final_validation_tool_names=_final_validation_tool_names_from_args(args),
    ).run(args.prompt, cipai=args.cipai)


def _generate_suite(args: argparse.Namespace) -> Dict:
    templates = _load_zero_shot_templates(args.templates, args.template_dir)
    results = []
    for template in templates:
        run_args = _args_for_template(args, template)
        if args.dry_run:
            result = {"template": template["name"], "description": template.get("description"), "dry_run": _dry_run_payload(run_args)}
        else:
            response = _generate_response_from_args(run_args)
            result = {"template": template["name"], "description": template.get("description"), "response": response.to_dict()}
        results.append(result)
    return {
        "prompt": args.prompt,
        "cipai": args.cipai,
        "template_count": len(results),
        "results": results,
    }


def _load_zero_shot_templates(paths: Optional[List[str]], template_dir: str) -> List[Dict]:
    selected_paths = [Path(path) for path in paths or []]
    if not selected_paths:
        selected_paths = sorted(Path(template_dir).glob("*.json"))
    if not selected_paths:
        raise SystemExit("No zero-shot templates found. Pass --template or set --template-dir.")
    return [_load_zero_shot_template(path) for path in selected_paths]


def _load_zero_shot_template(path: Path) -> Dict:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Zero-shot template must be a JSON object: {path}")
    template_dir = path.expanduser().resolve().parent
    template = dict(data)
    template["name"] = str(template.get("name") or path.stem)
    template["_path"] = str(path)
    if template.get("system_prompt_file"):
        prompt_path = _resolve_template_path(template_dir, str(template["system_prompt_file"]))
        template["system_prompt"] = prompt_path.read_text(encoding="utf-8")
    if template.get("toolset_config"):
        template["toolset_config"] = str(_resolve_template_path(template_dir, str(template["toolset_config"])))
    return template


def _resolve_template_path(template_dir: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidate = (template_dir / path).resolve()
    if candidate.exists():
        return candidate
    return Path(raw).resolve()


def _args_for_template(base_args: argparse.Namespace, template: Dict) -> argparse.Namespace:
    values = dict(vars(base_args))
    values["no_tools"] = bool(template.get("no_tools", False))
    values["toolset_config"] = None if values["no_tools"] else template.get("toolset_config")
    values["system_prompt"] = base_args.system_prompt or template.get("system_prompt")
    if base_args.max_tool_rounds is None:
        values["max_tool_rounds"] = _optional_int(template.get("max_tool_rounds"))
    if base_args.max_tool_calls_per_tool is None:
        values["max_tool_calls_per_tool"] = _optional_int(template.get("max_tool_calls_per_tool"))
    if not base_args.require_final_prosody_pass:
        values["require_final_prosody_pass"] = _optional_bool(template.get("require_final_prosody_pass"), False)
    final_validation = template.get("final_validation")
    final_validation_tool_names = _final_validation_tool_names_from_mapping(final_validation)
    if final_validation_tool_names:
        values["final_validation_tool_names"] = final_validation_tool_names
    return argparse.Namespace(**values)


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json_payload(input_json: Optional[str], input_path: Optional[str], label: str):
    if bool(input_json) == bool(input_path):
        raise SystemExit(f"Pass exactly one of {label}")
    if input_json:
        raw = input_json
    elif input_path == "-":
        raw = sys.stdin.read()
    else:
        with open(str(input_path), "r", encoding="utf-8") as handle:
            raw = handle.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON payload: {exc}") from exc


def _load_tool_batch(input_json: Optional[str], input_path: Optional[str]) -> List[Dict]:
    payload = _load_json_payload(input_json, input_path, "--input-json/--input")
    if isinstance(payload, list):
        calls = payload
    elif isinstance(payload, dict) and isinstance(payload.get("calls"), list):
        calls = payload["calls"]
    else:
        raise SystemExit("tool-batch input must be a JSON array or an object with a calls array")
    if not all(isinstance(call, dict) for call in calls):
        raise SystemExit("Every tool-batch call must be a JSON object")
    return calls


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-config", help="JSON config for the OpenAI-compatible writing API.")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--stream", action="store_true", help="Use streaming chat completions and show live text on stderr.")
    parser.add_argument(
        "--stream-reasoning",
        action="store_true",
        help="Also stream provider reasoning chunks on stderr when the API exposes them.",
    )


def _add_toolset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--toolset-config", help="Editable JSON config that selects and describes model-facing tools.")


def _add_style_tool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--style-tool-config",
        help="JSON config for style tool backend, style DB, schema versions, and tool-fitting API settings.",
    )


def _add_poetics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--poetics-path", help="Optional local source root for extracted poetics data.")
    parser.add_argument("--poetics-config-dir", help="Local poetics JSON config directory.")
    parser.add_argument("--poetics-data-dir", help="Local poetics data config directory, e.g. data/fangcun/config.")
    parser.add_argument("--poetics-genre", choices=["Shi", "Ci"])
    parser.add_argument("--poetics-rhyme-book")
    parser.add_argument("--poetics-no-longpu", action="store_true", help="Disable Longpu rule filtering.")
    parser.add_argument("--fangcun-path", help=argparse.SUPPRESS)
    parser.add_argument("--fangcun-config-dir", help=argparse.SUPPRESS)
    parser.add_argument("--fangcun-data-dir", help=argparse.SUPPRESS)
    parser.add_argument("--fangcun-genre", choices=["Shi", "Ci"], help=argparse.SUPPRESS)
    parser.add_argument("--fangcun-rhyme-book", help=argparse.SUPPRESS)
    parser.add_argument("--fangcun-no-longpu", action="store_true", help=argparse.SUPPRESS)


def _registry_from_args(args: argparse.Namespace):
    poetics_no_longpu = getattr(args, "poetics_no_longpu", False)
    legacy_no_longpu = getattr(args, "fangcun_no_longpu", False)
    poetics_ensure_longpu = False if poetics_no_longpu or legacy_no_longpu else None
    return build_default_registry(
        spec_path=args.specs,
        imagery_path=args.imagery,
        external_tool_config_path=args.tool_config,
        toolset_config_path=args.toolset_config,
        poetics_path=getattr(args, "poetics_path", None),
        poetics_config_dir=getattr(args, "poetics_config_dir", None),
        poetics_data_dir=getattr(args, "poetics_data_dir", None),
        poetics_genre=getattr(args, "poetics_genre", None),
        poetics_rhyme_book=getattr(args, "poetics_rhyme_book", None),
        poetics_ensure_longpu=poetics_ensure_longpu,
        fangcun_path=args.fangcun_path,
        fangcun_config_dir=args.fangcun_config_dir,
        fangcun_data_dir=args.fangcun_data_dir,
        fangcun_genre=args.fangcun_genre or "Ci",
        fangcun_rhyme_book=args.fangcun_rhyme_book or "Cilinzhengyun",
        fangcun_ensure_longpu=not legacy_no_longpu,
        style_tool_config_path=getattr(args, "style_tool_config", None),
    )


def _model_from_args(args: argparse.Namespace) -> OpenAICompatibleClient:
    settings = _model_settings_from_args(args)
    if not settings["model"]:
        raise SystemExit("--model is required or set OPENAI_COMPATIBLE_MODEL")
    if not settings["endpoint"]:
        raise SystemExit("--endpoint is required or set OPENAI_COMPATIBLE_CHAT_URL")
    return OpenAICompatibleClient(
        model=settings["model"],
        endpoint=settings["endpoint"],
        api_key=settings["api_key"],
        timeout=settings["timeout"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        system_prompt=settings["system_prompt"],
        stream=settings["stream"],
        stream_reasoning=settings["stream_reasoning"],
    )


def _dry_run_payload(args: argparse.Namespace) -> dict:
    settings = _model_settings_from_args(args)
    system_prompt = args.system_prompt or settings["system_prompt"] or (
        "You are generating Chinese ci. Use available tools when helpful, "
        "then return the final ci text only when you are satisfied."
    )
    payload = {
        "model": settings["model"] or "<model>",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.prompt},
        ],
    }
    if not args.no_tools:
        registry = _registry_from_args(args)
        payload["tools"] = registry.model_tool_schemas()
        payload["tool_choice"] = "auto"
    if settings["temperature"] is not None:
        payload["temperature"] = settings["temperature"]
    if settings["max_tokens"] is not None:
        payload["max_tokens"] = settings["max_tokens"]
    if settings["stream"]:
        payload["stream"] = True
    return payload


def _max_tool_rounds_from_args(args: argparse.Namespace) -> int:
    if args.max_tool_rounds is not None:
        return args.max_tool_rounds
    limits = _toolset_limits(args)
    parsed = _optional_int(limits.get("max_tool_rounds"))
    return 3 if parsed is None else parsed


def _max_tool_calls_per_tool_from_args(args: argparse.Namespace) -> Optional[int]:
    if args.max_tool_calls_per_tool is not None:
        return args.max_tool_calls_per_tool
    limits = _toolset_limits(args)
    return _optional_int(limits.get("max_tool_calls_per_tool"))


def _toolset_limits(args: argparse.Namespace) -> dict:
    path = getattr(args, "toolset_config", None)
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        return {}
    limits = config.get("limits", {})
    return dict(limits) if isinstance(limits, dict) else {}


def _require_final_prosody_pass_from_args(args: argparse.Namespace) -> bool:
    return bool(_final_validation_tool_names_from_args(args))


def _final_validation_tool_names_from_args(args: argparse.Namespace) -> List[str]:
    explicit = getattr(args, "final_validation_tool_names", None)
    if explicit is not None:
        return list(explicit)
    if getattr(args, "require_final_prosody_pass", False):
        return ["prosody"]
    path = getattr(args, "toolset_config", None)
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        return []
    final_validation = config.get("final_validation")
    if isinstance(final_validation, dict):
        tool_names = _final_validation_tool_names_from_mapping(final_validation)
        if tool_names:
            return tool_names
        if _optional_bool(final_validation.get("require_prosody_pass"), False):
            return ["prosody"]
    if _optional_bool(config.get("require_final_prosody_pass"), False):
        return ["prosody"]
    return []


def _final_validation_tool_names_from_mapping(value) -> List[str]:
    if not isinstance(value, dict):
        return []
    raw_names = value.get("tool_names") or value.get("tools") or value.get("require_tools_pass")
    if raw_names is None:
        return []
    if not isinstance(raw_names, list):
        raise SystemExit("final_validation.tool_names must be an array")
    return [str(item) for item in raw_names]


def _model_settings_from_args(args: argparse.Namespace) -> dict:
    config = _load_model_config(args.model_config) if getattr(args, "model_config", None) else {}
    api_key_env = str(config.get("api_key_env") or "OPENAI_COMPATIBLE_API_KEY")
    endpoint = (
        args.endpoint
        or config.get("endpoint")
        or config.get("chat_url")
        or config.get("url")
        or _endpoint_from_base_config(config)
        or os.environ.get("OPENAI_COMPATIBLE_CHAT_URL")
    )
    return {
        "model": args.model or config.get("model") or os.environ.get("OPENAI_COMPATIBLE_MODEL"),
        "endpoint": endpoint,
        "api_key": args.api_key or config.get("api_key") or os.environ.get(api_key_env),
        "timeout": _float_or_default(args.timeout, config.get("timeout"), 120.0),
        "temperature": _optional_float(args.temperature if args.temperature is not None else config.get("temperature")),
        "max_tokens": _optional_int(args.max_tokens if args.max_tokens is not None else config.get("max_tokens")),
        "system_prompt": args.system_prompt or config.get("system_prompt"),
        "stream": bool(args.stream or _optional_bool(config.get("stream"), False)),
        "stream_reasoning": bool(
            args.stream_reasoning or _optional_bool(config.get("stream_reasoning"), False)
        ),
    }


def _endpoint_from_base_config(config: dict) -> Optional[str]:
    base_url = config.get("base_url")
    if not base_url:
        return None
    chat_path = str(config.get("chat_path") or "/v1/chat/completions")
    return urljoin(str(base_url).rstrip("/") + "/", chat_path.lstrip("/"))


def _load_model_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("--model-config must point to a JSON object")
    provider = config.get("provider")
    if provider and provider != "openai-compatible":
        raise SystemExit("--model-config provider must be 'openai-compatible'")
    return config


def _metadata_from_json(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --metadata-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--metadata-json must be a JSON object")
    return parsed


def _optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Invalid integer value: {value}")


def _optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Invalid float value: {value}")


def _optional_bool(value, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"Invalid boolean value: {value}")


def _float_or_default(primary, secondary, default: float) -> float:
    parsed = _optional_float(primary if primary is not None else secondary)
    return default if parsed is None else parsed


if __name__ == "__main__":
    raise SystemExit(main())
