import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from longci_bench import cli
from longci_bench.models import ModelClient, OpenAICompatibleClient, ToolCallingRunner
from longci_bench.schemas import ModelResponse, ToolCall, ToolRequest, ToolResponse
from longci_bench.tools import CallableTool, ToolRegistry, apply_toolset_config, build_default_registry
from tests.test_fangcun_tool import _fake_fangcun_source


class ZeroShotToolingTests(unittest.TestCase):
    def test_toolset_config_can_select_alias_and_metadata_defaults(self):
        received = []

        def handler(request):
            received.append(request)
            return ToolResponse(request.tool_name, passed=True, metadata={"seen": request.metadata})

        registry = ToolRegistry([CallableTool("prosody", handler, description="original")])
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "toolset.json"
            config_path.write_text(
                json.dumps(
                    {
                        "include": ["meter_check"],
                        "tools": {
                            "meter_check": {
                                "source": "prosody",
                                "description": "experiment description",
                                "metadata_defaults": {"ensure_longpu": True},
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            configured = apply_toolset_config(registry, str(config_path))
            schema = configured.model_tool_schemas()[0]
            response = configured.run(ToolRequest("meter_check", "风月", metadata={"ensure_longpu": False}))

        self.assertEqual(configured.names(), ["meter_check"])
        self.assertEqual(schema["function"]["name"], "meter_check")
        self.assertEqual(schema["function"]["description"], "experiment description")
        self.assertEqual(received[0].tool_name, "prosody")
        self.assertEqual(received[0].metadata["ensure_longpu"], False)
        self.assertEqual(response.tool_name, "meter_check")

    def test_second_layer_toolset_configs_expose_expected_tool_tables(self):
        root = Path(__file__).resolve().parents[1]
        cases = {
            "configs/zero_shot/toolsets/base.judou_only.json": ["check_judou", "get_rule"],
            "configs/zero_shot/toolsets/base.strict_judou_then_meter.json": [
                "check_judou",
                "check_prosody",
                "get_rule",
            ],
            "configs/zero_shot/toolsets/base_tool.json": [
                "check_judou",
                "check_prosody",
                "get_rule",
            ],
            "configs/zero_shot/toolsets/base_style.json": [
                "check_judou",
                "check_prosody",
                "get_rule",
                "imagery_map",
                "style_contradiction_scan",
                "style_rewrite",
                "style_vectorize",
                "syntax_map",
            ],
            "configs/zero_shot/toolsets/base_repair.json": [
                "check_judou",
                "check_prosody",
                "form_repeat_check",
                "get_rule",
                "repeat_repair",
            ],
            "configs/zero_shot/toolsets/extra.validation.json": ["prosody", "rule_lookup"],
            "configs/zero_shot/toolsets/lexical.reference.json": [
                "allusion_search",
                "char_lookup",
                "phrase_suggest",
                "rhyme_list",
                "rhyme_lookup",
            ],
            "configs/zero_shot/toolsets/post_check.json": ["imagery", "repetition"],
            "configs/zero_shot/toolsets/style.operations.json": [
                "imagery_map",
                "style_contradiction_scan",
                "style_rewrite",
                "style_vectorize",
                "syntax_map",
            ],
            "configs/zero_shot/toolsets/repair.strict_repetition.json": [
                "check_judou",
                "check_prosody",
                "get_rule",
                "repeat_repair",
                "strict_repeat_check",
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            for relative_path, expected_names in cases.items():
                registry = build_default_registry(
                    fangcun_path=str(source),
                    toolset_config_path=str(root / relative_path),
                )
                schema_names = [schema["function"]["name"] for schema in registry.model_tool_schemas()]

                self.assertEqual(registry.names(), sorted(expected_names), relative_path)
                self.assertEqual(sorted(schema_names), sorted(expected_names), relative_path)

    def test_strict_repetition_toolset_alias_sets_single_char_validation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(
                fangcun_path=str(source),
                toolset_config_path=str(root / "configs/zero_shot/toolsets/repair.strict_repetition.json"),
            )

            response = registry.run(ToolRequest("strict_repeat_check", "风月，云月。"))

        self.assertEqual(response.tool_name, "strict_repeat_check")
        self.assertEqual(response.metadata["input"]["forbid_single_char_repetition"], True)
        self.assertEqual(response.metadata["input"]["min_ngram_size"], 1)
        self.assertEqual(response.metadata["input"]["validation_only"], True)
        self.assertEqual(response.metrics["final_repetition_min_ngram_size"], 1)
        self.assertEqual(response.metadata["repetitions"][0]["substring"], "月")
        self.assertFalse(response.passed)

    def test_curated_four_way_templates_expose_no_base_style_and_repair_tables(self):
        buffer = io.StringIO()
        root = Path(__file__).resolve().parents[1]
        templates = [
            "configs/zero_shot/templates/no_tool.json",
            "configs/zero_shot/templates/base_tool.json",
            "configs/zero_shot/templates/base_style.json",
            "configs/zero_shot/templates/base_repair.json",
        ]
        args = [
            "generate-suite",
            "--dry-run",
            "--model",
            "closed-model",
            "--poetics-data-dir",
            "data/fangcun/config",
            "--cipai",
            "暗香",
            "--prompt",
            "写一个性灵派风格的暗香，以江南初雪为题",
        ]
        for template in templates:
            args.extend(["--template", str(root / template)])

        with redirect_stdout(buffer):
            exit_code = cli.main(args)

        payload = json.loads(buffer.getvalue())
        tool_names_by_template = {}
        for result in payload["results"]:
            tools = result["dry_run"].get("tools") or []
            tool_names_by_template[result["template"]] = [tool["function"]["name"] for tool in tools]
        self.assertEqual(exit_code, 0)
        self.assertEqual(tool_names_by_template["no_tool"], [])
        self.assertEqual(tool_names_by_template["base_tool"], ["get_rule", "check_judou", "check_prosody"])
        self.assertEqual(
            sorted(tool_names_by_template["base_style"]),
            sorted(["get_rule", "check_judou", "check_prosody", "style_vectorize", "style_rewrite", "imagery_map", "syntax_map", "style_contradiction_scan"]),
        )
        self.assertEqual(
            tool_names_by_template["base_repair"],
            ["get_rule", "check_judou", "check_prosody", "repeat_repair", "form_repeat_check"],
        )

    def test_tool_calling_runner_uses_chat_tool_messages_when_available(self):
        model = _TwoRoundChatModel()

        def handler(request):
            return ToolResponse(request.tool_name, passed=True, metadata={"text": request.text})

        registry = ToolRegistry([CallableTool("prosody", handler)])
        response = ToolCallingRunner(model, registry, max_tool_rounds=2).run("写卜算子", cipai="卜算子")

        self.assertEqual(response.text, "最终词作")
        self.assertEqual(len(response.metadata["tool_trace"]), 1)
        second_round_messages = model.calls[1]["messages"]
        tool_msgs = [m for m in second_round_messages if m["role"] == "tool"]
        self.assertTrue(len(tool_msgs) >= 1)
        self.assertEqual(tool_msgs[-1]["tool_call_id"], "call-1")
        self.assertIn('"passed": true', tool_msgs[-1]["content"])

    def test_tool_calling_runner_returns_last_draft_if_model_never_finalizes(self):
        model = _NeverFinalizesChatModel()

        def handler(request):
            return ToolResponse(request.tool_name, passed=True)

        registry = ToolRegistry([CallableTool("prosody", handler)])
        response = ToolCallingRunner(model, registry, max_tool_rounds=0).run("写卜算子", cipai="卜算子")

        self.assertEqual(response.text, "草稿")
        self.assertEqual(response.metadata["finish_reason"], "max_tool_rounds_exhausted")
        self.assertEqual(response.metadata["fallback_text_source"], "last_draft_tool_call.arguments.text")

    def test_tool_calling_runner_continues_after_empty_visible_content(self):
        model = _EmptyThenFinalChatModel()
        registry = ToolRegistry([])

        response = ToolCallingRunner(model, registry, max_tool_rounds=1).run("写暗香", cipai="暗香")

        self.assertEqual(response.text, "最终正文")
        self.assertEqual(len(model.calls), 2)
        self.assertIn("No visible ci body", model.calls[1][-1]["content"])

    def test_tool_calling_runner_enforces_per_tool_call_limit(self):
        model = _RepeatedToolCallModel()
        seen = []

        def handler(request):
            seen.append(request)
            return ToolResponse(request.tool_name, passed=True)

        registry = ToolRegistry([CallableTool("prosody", handler)])
        response = ToolCallingRunner(model, registry, max_tool_rounds=1, max_tool_calls_per_tool=1).run(
            "写卜算子",
            cipai="卜算子",
        )

        self.assertEqual(len(seen), 1)
        second_response = response.metadata["tool_trace"][1]["observations"][0]["response"]
        self.assertEqual(second_response["issues"][0]["type"], "tool_call_limit")

    def test_tool_calling_runner_revises_until_final_prosody_passes(self):
        model = _FinalValidationModel()
        seen = []

        def handler(request):
            seen.append(request.text)
            return ToolResponse("prosody", passed=request.text == "合律", issues=[] if request.text == "合律" else [{"type": "tone"}])

        registry = ToolRegistry([CallableTool("prosody", handler)])
        response = ToolCallingRunner(
            model,
            registry,
            max_tool_rounds=2,
            require_final_prosody_pass=True,
        ).run("写卜算子", cipai="卜算子")

        self.assertEqual(response.text, "合律")
        self.assertEqual(seen, ["出律", "合律"])
        self.assertTrue(response.metadata["final_validation"]["passed"])
        self.assertIn("Final validation failed", model.calls[1][-1]["content"])

    def test_generate_dry_run_does_not_require_endpoint(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.main(["generate", "--dry-run", "--model", "closed-model", "--prompt", "写一首词"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["model"], "closed-model")
        self.assertEqual(payload["messages"][1]["content"], "写一首词")
        self.assertIn("tools", payload)

    def test_tool_contract_cli_exports_input_and_output_schemas(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.main(["tool-contract"])

        payload = json.loads(buffer.getvalue())
        tool_names = {tool["name"] for tool in payload["tools"]}
        prosody = next(tool for tool in payload["tools"] if tool["name"] == "prosody")
        self.assertEqual(exit_code, 0)
        self.assertIn("prosody", tool_names)
        self.assertIn("input_schema", prosody)
        self.assertIn("output_schema", prosody)

    def test_tool_call_cli_accepts_json_arguments(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli.main(
                [
                    "tool-call",
                    "--tool-name",
                    "repetition",
                    "--input-json",
                    '{"text":"风月风月"}',
                ]
            )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["tool_name"], "repetition")
        self.assertFalse(payload["passed"])

    def test_generate_dry_run_reads_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "model.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "openai-compatible",
                        "endpoint": "https://api.example.test/v1/chat/completions",
                        "api_key": "secret",
                        "model": "config-model",
                        "temperature": 0.25,
                        "max_tokens": 1234,
                        "stream": True,
                        "system_prompt": "系统提示",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.main(
                    [
                        "generate",
                        "--dry-run",
                        "--model-config",
                        str(config_path),
                        "--prompt",
                        "写一首词",
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["model"], "config-model")
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["stream"], True)
        self.assertEqual(payload["messages"][0]["content"], "系统提示")

    def test_model_config_can_build_endpoint_from_base_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "model.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "openai-compatible",
                        "base_url": "https://api.example.test/compatible-mode",
                        "chat_path": "/v1/chat/completions",
                        "model": "config-model",
                    }
                ),
                encoding="utf-8",
            )
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--dry-run",
                    "--model-config",
                    str(config_path),
                    "--prompt",
                    "写一首词",
                ]
            )
            settings = cli._model_settings_from_args(args)
            self.assertEqual(settings["endpoint"], "https://api.example.test/compatible-mode/v1/chat/completions")

    def test_toolset_config_supplies_runner_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "toolset.json"
            config_path.write_text(
                json.dumps({"limits": {"max_tool_rounds": 2, "max_tool_calls_per_tool": 1}}),
                encoding="utf-8",
            )
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--dry-run",
                    "--model",
                    "model",
                    "--toolset-config",
                    str(config_path),
                    "--prompt",
                    "写一首词",
                ]
            )
            self.assertEqual(cli._max_tool_rounds_from_args(args), 2)
            self.assertEqual(cli._max_tool_calls_per_tool_from_args(args), 1)

    def test_toolset_config_can_require_final_prosody_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "toolset.json"
            config_path.write_text(
                json.dumps({"final_validation": {"require_prosody_pass": True}}),
                encoding="utf-8",
            )
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--dry-run",
                    "--model",
                    "model",
                    "--toolset-config",
                    str(config_path),
                    "--prompt",
                    "写一首词",
                ]
            )
            self.assertTrue(cli._require_final_prosody_pass_from_args(args))

    def test_toolset_config_can_select_final_validation_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "toolset.json"
            config_path.write_text(
                json.dumps({"final_validation": {"tool_names": ["check_judou", "check_prosody"]}}),
                encoding="utf-8",
            )
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--dry-run",
                    "--model",
                    "model",
                    "--toolset-config",
                    str(config_path),
                    "--prompt",
                    "写一首词",
                ]
            )
            self.assertEqual(cli._final_validation_tool_names_from_args(args), ["check_judou", "check_prosody"])
            self.assertTrue(cli._require_final_prosody_pass_from_args(args))

    def test_generate_suite_dry_run_uses_multiple_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toolset_path = root / "toolset.json"
            toolset_path.write_text(json.dumps({"include": ["prosody"]}), encoding="utf-8")
            strict_path = root / "strict.json"
            strict_path.write_text(
                json.dumps(
                    {
                        "name": "strict",
                        "toolset_config": str(toolset_path),
                        "system_prompt": "严格模板",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            baseline_path = root / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "name": "baseline",
                        "no_tools": True,
                        "system_prompt": "基线模板",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.main(
                    [
                        "generate-suite",
                        "--dry-run",
                        "--model",
                        "closed-model",
                        "--prompt",
                        "以南京古城墙为主题，写作一首疏影。",
                        "--template",
                        str(strict_path),
                        "--template",
                        str(baseline_path),
                    ]
                )

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["template_count"], 2)
        self.assertEqual(payload["results"][0]["template"], "strict")
        self.assertEqual(payload["results"][0]["dry_run"]["messages"][0]["content"], "严格模板")
        self.assertIn("tools", payload["results"][0]["dry_run"])
        self.assertEqual(payload["results"][1]["template"], "baseline")
        self.assertNotIn("tools", payload["results"][1]["dry_run"])

    def test_generate_suite_text_only_formatter(self):
        text = cli._suite_text(
            {
                "results": [
                    {"template": "baseline", "response": {"text": "甲乙。"}},
                    {"template": "strict", "response": {"text": "丙丁。"}},
                ]
            }
        )

        self.assertEqual(text, "### baseline\n甲乙。\n\n### strict\n丙丁。\n")


    def test_openai_compatible_stream_parser_accumulates_content_and_tool_calls(self):
        chunks = []
        response = _FakeStreamingResponse(
            [
                _sse_line({"choices": [{"delta": {"content": "疏影"}}]}),
                _sse_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "prosody",
                                                "arguments": "{\"text\":\"草",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": "稿\"}",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )
        client = OpenAICompatibleClient(
            "stream-model",
            endpoint="https://api.example.test/v1/chat/completions",
            stream=True,
            stream_sink=chunks.append,
        )

        parsed = client._read_stream_response(response, metadata={"cipai": "疏影"})

        self.assertEqual(parsed.text, "疏影")
        self.assertEqual("".join(chunks), "疏影\n")
        self.assertEqual(parsed.tool_calls[0].name, "prosody")
        self.assertEqual(parsed.tool_calls[0].arguments["text"], "草稿")
        self.assertEqual(parsed.metadata["stream"], True)


class _TwoRoundChatModel(ModelClient):
    def __init__(self):
        self.calls = []

    def generate_from_messages(self, messages, tools=None, metadata=None):
        self.calls.append({"messages": list(messages), "tools": tools, "metadata": metadata})
        if len(self.calls) == 1:
            return ModelResponse(
                text="初稿",
                tool_calls=[ToolCall("prosody", {"text": "初稿"}, call_id="call-1")],
                metadata={
                    "assistant_message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "prosody", "arguments": '{"text":"初稿"}'},
                            }
                        ],
                    }
                },
            )
        return ModelResponse(
            text="最终词作",
            metadata={"assistant_message": {"role": "assistant", "content": "最终词作"}},
        )


class _NeverFinalizesChatModel(ModelClient):
    def generate_from_messages(self, messages, tools=None, metadata=None):
        return ModelResponse(
            text="",
            tool_calls=[ToolCall("prosody", {"text": "草稿"}, call_id="call-1")],
            metadata={
                "assistant_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "prosody", "arguments": '{"text":"草稿"}'},
                        }
                    ],
                }
            },
        )


class _EmptyThenFinalChatModel(ModelClient):
    def __init__(self):
        self.calls = []

    def generate_from_messages(self, messages, tools=None, metadata=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return ModelResponse(
                text="",
                metadata={"assistant_message": {"role": "assistant", "content": "", "reasoning_content": "草稿在推理里"}},
            )
        return ModelResponse(
            text="最终正文",
            metadata={"assistant_message": {"role": "assistant", "content": "最终正文"}},
        )


class _RepeatedToolCallModel(ModelClient):
    def __init__(self):
        self.calls = 0

    def generate_from_messages(self, messages, tools=None, metadata=None):
        self.calls += 1
        return ModelResponse(
            text="",
            tool_calls=[ToolCall("prosody", {"text": "草稿足够长，用来模拟一首词作正文。"}, call_id=f"call-{self.calls}")],
            metadata={
                "assistant_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{self.calls}",
                            "type": "function",
                            "function": {
                                "name": "prosody",
                                "arguments": '{"text":"草稿足够长，用来模拟一首词作正文。"}',
                            },
                        }
                    ],
                }
            },
        )


class _FinalValidationModel(ModelClient):
    def __init__(self):
        self.calls = []

    def generate_from_messages(self, messages, tools=None, metadata=None):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return ModelResponse(
                text="出律",
                metadata={"assistant_message": {"role": "assistant", "content": "出律"}},
            )
        return ModelResponse(
            text="合律",
            metadata={"assistant_message": {"role": "assistant", "content": "合律"}},
        )


class _FakeStreamingResponse:
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)


def _sse_line(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


if __name__ == "__main__":
    unittest.main()
