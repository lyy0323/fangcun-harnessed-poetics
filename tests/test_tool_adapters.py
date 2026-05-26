import unittest

from longci_bench.schemas import ToolRequest, ToolResponse
from longci_bench.tools import CallableTool, ToolRegistry
from longci_bench.tools.external import external_http_tool_from_config


class ToolAdapterTests(unittest.TestCase):
    def test_tool_response_from_mapping(self):
        response = ToolResponse.from_mapping(
            {
                "tool_name": "prosody",
                "passed": True,
                "issues": [],
                "metrics": {"violation_count": 0},
                "metadata": {"source": "test"},
            }
        )
        self.assertTrue(response.passed)
        self.assertEqual(response.metrics["violation_count"], 0)

    def test_callable_tool_can_override_placeholder_name(self):
        def handler(request):
            return ToolResponse(
                tool_name=request.tool_name,
                passed=True,
                metrics={"source": "callable"},
            )

        registry = ToolRegistry()
        registry.register(CallableTool("prosody", handler))
        response = registry.run(ToolRequest("prosody", "text", cipai="cipai"))
        self.assertTrue(response.passed)
        self.assertEqual(response.metrics["source"], "callable")

    def test_tool_contract_and_direct_call_are_json_compatible(self):
        def handler(request):
            return ToolResponse(
                tool_name=request.tool_name,
                passed=bool(request.text),
                metadata={"cipai": request.cipai},
            )

        tool = CallableTool("meter_check", handler, description="check meter")
        contract = tool.contract()
        response = tool.call({"text": "风月", "cipai": "卜算子"})

        self.assertEqual(contract["name"], "meter_check")
        self.assertEqual(contract["input_schema"]["required"], ["text"])
        self.assertIn("output_schema", contract)
        self.assertTrue(response["passed"])
        self.assertEqual(response["metadata"]["cipai"], "卜算子")

    def test_registry_accepts_callable_payload_shapes(self):
        def handler(request):
            return ToolResponse(
                tool_name=request.tool_name,
                passed=True,
                metadata={"text": request.text},
            )

        registry = ToolRegistry([CallableTool("prosody", handler)])
        direct = registry.call("prosody", {"text": "草稿"})
        model_call = registry.call_many(
            [
                {
                    "name": "prosody",
                    "arguments": {"text": "模型草稿", "cipai": "卜算子"},
                }
            ]
        )

        self.assertEqual(direct["metadata"]["text"], "草稿")
        self.assertEqual(model_call[0]["metadata"]["text"], "模型草稿")

    def test_external_http_tool_from_config(self):
        tool = external_http_tool_from_config(
            {
                "name": "prosody",
                "endpoint": "http://127.0.0.1:8001/tools/prosody",
                "description": "real checker",
                "timeout": 7,
            }
        )
        self.assertEqual(tool.name, "prosody")
        self.assertEqual(tool.endpoint, "http://127.0.0.1:8001/tools/prosody")
        self.assertEqual(tool.timeout, 7.0)


if __name__ == "__main__":
    unittest.main()
