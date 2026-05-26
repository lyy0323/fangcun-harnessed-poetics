import unittest

from longci_bench.schemas import ToolRequest
from longci_bench.tools import ImageryRecallTool, ProsodyTool, RepetitionTool


class ToolTests(unittest.TestCase):
    def test_prosody_passes_example_spec(self):
        tool = ProsodyTool(specs={"sample_long_ci": {"line_lengths": [4, 4]}})
        response = tool.run(ToolRequest("prosody", "风月无边。\n江水归来。", cipai="sample_long_ci"))
        self.assertTrue(response.passed)
        self.assertEqual(response.metrics["violation_count"], 0)

    def test_prosody_reports_line_length(self):
        tool = ProsodyTool(specs={"sample_long_ci": {"line_lengths": [4, 4]}})
        response = tool.run(ToolRequest("prosody", "风月无边长。\n江水归来。", cipai="sample_long_ci"))
        self.assertFalse(response.passed)
        self.assertEqual(response.issues[0]["type"], "line_length")

    def test_repetition_reports_positions(self):
        response = RepetitionTool(min_ngram=2, max_ngram=2).run(
            ToolRequest("repetition", "风月无边，风月又来。")
        )
        self.assertFalse(response.passed)
        self.assertEqual(response.issues[0]["substring"], "风月")
        self.assertEqual(len(response.issues[0]["spans"]), 2)

    def test_imagery_reports_repeated_category(self):
        response = ImageryRecallTool().run(ToolRequest("imagery", "明月照江水，残月落花前。"))
        categories = {issue["category"] for issue in response.issues}
        self.assertIn("moon", categories)

    def test_imagery_does_not_double_count_nested_terms(self):
        response = ImageryRecallTool().run(ToolRequest("imagery", "明月照归舟。"))
        categories = {issue["category"] for issue in response.issues}
        self.assertNotIn("moon", categories)


if __name__ == "__main__":
    unittest.main()
