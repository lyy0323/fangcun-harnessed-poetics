import unittest

from longci_bench.schemas import ToolRequest
from longci_bench.tools import (
    ImageryMapTool,
    StyleContradictionScanTool,
    StyleRewriteTool,
    StyleVectorizeTool,
    SyntaxMapTool,
    build_default_registry,
)


class StyleToolTests(unittest.TestCase):
    def test_registry_exposes_style_tool_group(self):
        registry = build_default_registry()
        contract = registry.contract()

        self.assertIn("style_tool", contract["tool_groups"])
        self.assertEqual(
            set(contract["tool_groups"]["style_tool"]["tools"]),
            {
                "style_vectorize",
                "style_rewrite",
                "imagery_map",
                "syntax_map",
                "style_contradiction_scan",
            },
        )

    def test_style_vectorize_returns_nearest_style(self):
        response = StyleVectorizeTool().run(
            ToolRequest("style_vectorize", "淡月涵江，短笛在远。", metadata={"input_mode": "text"})
        )

        self.assertTrue(response.passed)
        self.assertEqual(response.metadata["operation"], "style_vectorize")
        self.assertEqual(response.metadata["nearest_styles"][0]["style_id"], "jiang_kui_qingkong")
        self.assertIn("imagery_cold_light", response.metadata["style_vector"]["dimensions"])
        self.assertGreater(response.metrics["top_similarity"], 0)

    def test_style_rewrite_respects_slot_and_char_count(self):
        response = StyleRewriteTool().run(
            ToolRequest(
                "style_rewrite",
                "月光照着江上的船",
                cipai="疏影",
                metadata={
                    "slot": "line_2_phrase",
                    "target_style": "jiang_kui_qingkong",
                    "prosody_constraints": {"char_count": 4, "ban_units": ["照着"]},
                },
            )
        )

        self.assertTrue(response.passed)
        self.assertGreaterEqual(response.metrics["candidate_count"], 1)
        top = response.metadata["candidates"][0]
        self.assertEqual(top["slot"], "line_2_phrase")
        self.assertEqual(len(top["text"]), 4)
        self.assertIn(top["text"], {"淡月涵江", "江月微明", "冷月横江", "淡月归舟", "江月孤舟"})

    def test_imagery_map_returns_style_compatible_imagery(self):
        response = ImageryMapTool().run(
            ToolRequest(
                "imagery_map",
                "月光",
                metadata={"target_style": "jiang_kui_qingkong", "semantic_role": "cold_light"},
            )
        )

        self.assertTrue(response.passed)
        targets = {item["target_imagery"] for item in response.metadata["results"]}
        self.assertIn("淡月", targets)
        self.assertEqual(response.metadata["results"][0]["semantic_role"], "cold_light")

    def test_syntax_map_returns_style_syntax_candidates(self):
        response = SyntaxMapTool().run(
            ToolRequest("syntax_map", "月光照着江上的船", metadata={"target_style": "jiang_kui_qingkong"})
        )

        self.assertTrue(response.passed)
        top = response.metadata["results"][0]
        self.assertIn("rewritten_syntax", top)
        self.assertIn("moonlight", top["kept_semantics"])
        self.assertGreater(top["style_score"], 0)

    def test_style_contradiction_scan_reports_spans(self):
        response = StyleContradictionScanTool().run(
            ToolRequest(
                "style_contradiction_scan",
                "开心漂亮的月光照着江上的船。",
                metadata={"target_style": "jiang_kui_qingkong"},
            )
        )

        self.assertFalse(response.passed)
        self.assertGreaterEqual(response.metrics["contradiction_count"], 2)
        issue_types = {item["type"] for item in response.issues}
        self.assertIn("style_contradiction", issue_types)
        contradiction_types = {item["type"] for item in response.metadata["contradictions"]}
        self.assertIn("register", contradiction_types)
        self.assertIsNotNone(response.issues[0]["span"]["line"])


if __name__ == "__main__":
    unittest.main()
