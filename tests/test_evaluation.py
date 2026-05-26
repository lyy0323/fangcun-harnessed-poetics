import unittest

from longci_bench.evaluation.api_metrics import ApiFluencyJudge, ApiPoeticEntityExtractor
from longci_bench.evaluation.config import build_metric_components
from longci_bench.evaluation.metrics import evaluate_records
from longci_bench.evaluation.fluency import FluencyResult, FluencyScorer
from longci_bench.schemas import GenerationRecord, ModelResponse, ToolResponse
from longci_bench.tools import CallableTool, ToolRegistry, build_default_registry


class EvaluationTests(unittest.TestCase):
    def test_evaluate_records_summary(self):
        registry = build_default_registry()
        registry.get("prosody").specs["sample_long_ci"] = {"line_lengths": [4, 4]}
        records = [
            GenerationRecord(
                id="ex-1",
                split="hot100",
                cipai="sample_long_ci",
                prompt="prompt",
                text="风月无边。\n江水归来。",
                model="dummy",
            )
        ]
        report = evaluate_records(records, registry)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["summary"]["prosody_pass_rate"], 1.0)
        self.assertIn("fluency_average", report["summary"])

    def test_evaluate_records_computes_requested_metric_groups(self):
        registry = _metric_registry()
        records = [
            GenerationRecord(
                id="ex-2",
                split="hot100",
                cipai="sample_long_ci",
                prompt="prompt",
                text="风月风月。",
                model="dummy",
                metadata={
                    "tool_trace": [
                        {
                            "round": 0,
                            "observations": [
                                {
                                    "call": {"name": "get_rule"},
                                    "response": {"tool_name": "get_rule", "passed": True},
                                },
                                {
                                    "call": {"name": "check_judou"},
                                    "response": {"tool_name": "check_judou", "passed": False},
                                },
                            ],
                        },
                        {
                            "round": 1,
                            "final_validation": {
                                "validations": [
                                    {"response": {"tool_name": "check_judou", "passed": True}},
                                    {"response": {"tool_name": "check_prosody", "passed": False}},
                                ]
                            },
                        },
                    ]
                },
            )
        ]

        report = evaluate_records(records, registry)
        row = report["records"][0]

        self.assertEqual(report["summary"]["judou_pass_rate"], 1.0)
        self.assertEqual(report["summary"]["prosody_pass_rate"], 0.0)
        self.assertEqual(report["summary"]["final_validation_pass_rate"], 0.0)
        self.assertEqual(row["prosody_violation_count"], 2.0)
        self.assertEqual(row["out_of_prosody_ratio"], 0.5)
        self.assertEqual(row["content_repetition_rate"], 0.5)
        self.assertEqual(row["entity_backend"], "heuristic")
        self.assertEqual(row["fluency_backend"], "heuristic")
        self.assertEqual(row["tool_calls"]["tool_call_count"], 2)
        self.assertEqual(row["tool_calls"]["failed_tool_response_count"], 1)
        self.assertEqual(row["tool_calls"]["final_validation_attempt_count"], 2)
        self.assertEqual(report["summary"]["tool_call_counts"], {"check_judou": 1, "get_rule": 1})

    def test_evaluate_records_accepts_custom_fluency_scorer(self):
        registry = _metric_registry()
        records = [
            GenerationRecord(
                id="ex-3",
                split="hot100",
                cipai="sample_long_ci",
                prompt="prompt",
                text="风月风月。",
                model="dummy",
            )
        ]

        report = evaluate_records(records, registry, fluency_scorer=_FixedFluencyScorer(), fluency_pass_threshold=0.8)
        row = report["records"][0]

        self.assertEqual(row["fluency"], 0.7)
        self.assertFalse(row["fluency_passed"])
        self.assertEqual(row["fluency_backend"], "fixed")
        self.assertEqual(row["fluency_dimensions"], {"syntax": 0.7})

    def test_api_metric_backends_parse_structured_json(self):
        client = _FakeMetricClient(
            [
                {
                    "entities": [
                        {"text": "明月", "category": "imagery", "evidence": "明月照归舟"},
                        {"text": "归舟", "category": "object", "evidence": "明月照归舟"},
                    ]
                },
                {
                    "score": 0.82,
                    "dimensions": {"syntax": 0.8, "semantic_coherence": 0.85},
                    "issues": [{"type": "register", "message": "minor modern phrase", "evidence": "x"}],
                },
            ]
        )

        entity_result = ApiPoeticEntityExtractor(client).extract_result("明月照归舟。", cipai="卜算子")
        fluency_result = ApiFluencyJudge(client).score_result("明月照归舟。", cipai="卜算子")

        self.assertEqual(entity_result.terms, ["明月", "归舟"])
        self.assertEqual(entity_result.items[0]["category"], "imagery")
        self.assertEqual(fluency_result.score, 0.82)
        self.assertEqual(fluency_result.dimensions["semantic_coherence"], 0.85)
        self.assertEqual(fluency_result.issues[0]["type"], "register")

    def test_metric_config_accepts_tool_fitting_model_config_directly(self):
        components = build_metric_components(
            metric_config_path="configs/model.tool_fitting.openai_compatible.example.json"
        )

        self.assertIsNotNone(components.entity_extractor)
        self.assertIsNotNone(components.fluency_scorer)
        self.assertEqual(components.fluency_pass_threshold, 0.6)


def _metric_registry():
    def check_judou(request):
        return ToolResponse("check_judou", passed=True, metrics={"violation_count": 0})

    def check_prosody(request):
        return ToolResponse(
            "check_prosody",
            passed=False,
            issues=[{"type": "tone"}, {"type": "rhyme"}],
            metrics={"violation_count": 2, "checked_chars": 4},
        )

    def repetition(request):
        return ToolResponse(
            "repetition",
            passed=False,
            issues=[{"type": "repeated_substring", "substring": "风月"}],
            metrics={
                "content_repetition_rate": 0.5,
                "repeated_substring_count": 1,
                "repeated_units": 2,
            },
        )

    def imagery(request):
        return ToolResponse(
            "imagery",
            passed=True,
            metrics={"repeated_imagery_category_count": 0},
        )

    return ToolRegistry(
        [
            CallableTool("check_judou", check_judou),
            CallableTool("check_prosody", check_prosody),
            CallableTool("repetition", repetition),
            CallableTool("imagery", imagery),
        ]
    )


class _FixedFluencyScorer(FluencyScorer):
    def score_result(self, text, cipai=None, prompt=None, metadata=None):
        return FluencyResult(
            score=0.7,
            dimensions={"syntax": 0.7},
            backend="fixed",
        )


class _FakeMetricClient:
    model = "fake-metric-model"

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def generate_from_messages(self, messages, tools=None, metadata=None):
        payload = self.payloads.pop(0)
        return ModelResponse(text=_json_text(payload))


def _json_text(payload):
    import json

    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
