import tempfile
import unittest

from longci_bench.schemas import ToolRequest
from longci_bench.tools import RepeatRepairTool, build_default_registry

from tests.test_fangcun_tool import _fake_fangcun_source


class RepeatRepairToolTests(unittest.TestCase):
    def test_locates_repeated_substrings_and_returns_candidates(self):
        tool = RepeatRepairTool()

        response = tool.run(
            ToolRequest(
                "repeat_repair",
                "风月无边，风月又来。",
                metadata={"ngram_size": 2},
            )
        )

        self.assertTrue(response.passed)
        self.assertEqual(response.metrics["repetition_count"], 1)
        self.assertGreater(response.metrics["candidate_count"], 0)
        repetition = response.metadata["repetitions"][0]
        self.assertEqual(repetition["substring"], "风月")
        self.assertEqual(repetition["occurrence_count"], 2)
        self.assertEqual(repetition["occurrences"][0]["sentence_index"], 1)
        self.assertEqual(repetition["occurrences"][1]["sentence_index"], 2)
        self.assertEqual(repetition["occurrences"][0]["semantic_role"], "scene_imagery")

        best = response.metadata["best_candidate"]
        self.assertIsNotNone(best)
        self.assertNotEqual(best["replacement"], "风月")
        self.assertEqual(best["validation"]["repetition_passed"], True)
        self.assertIn("风月无边", best["patched_text"])
        self.assertNotIn("风月又来", best["patched_text"])

    def test_binds_repeat_targets_to_local_rule_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(fangcun_path=str(source))

            response = registry.run(
                ToolRequest(
                    "repeat_repair",
                    "风月，风月。",
                    cipai="卜算子",
                    metadata={"ngram_size": 2},
                )
            )
            contract = registry.contract()

        self.assertIn("repeat_repair", registry.names())
        self.assertIn("repeat_repair", contract["tool_groups"]["repair_tool"]["tools"])
        self.assertEqual(response.metadata["rule"]["sentence_char_counts"], [2, 2])
        target = response.metadata["repair_targets"][0]
        window = target["repair_window"]
        self.assertEqual(window["sentence_index"], 2)
        self.assertEqual(window["sentence_char_range_1based"], [3, 4])
        self.assertEqual(window["char_range_1based"], [3, 4])
        self.assertEqual(window["tone_pattern"], ["P", "Z"])
        self.assertEqual(window["forbidden_units"], ["风月"])

        best = response.metadata["best_candidate"]
        self.assertTrue(best["validation"]["repetition_passed"])
        self.assertTrue(best["validation"]["judou_passed"])

    def test_can_repair_single_character_repetition_for_strict_short_ci(self):
        tool = RepeatRepairTool()

        response = tool.run(
            ToolRequest(
                "repeat_repair",
                "风月，云月。",
                metadata={"forbid_single_char_repetition": True},
            )
        )

        self.assertTrue(response.passed)
        self.assertEqual(response.metadata["input"]["min_ngram_size"], 1)
        self.assertEqual(response.metrics["final_repetition_min_ngram_size"], 1)
        repetition = response.metadata["repetitions"][0]
        self.assertEqual(repetition["substring"], "月")
        self.assertEqual(repetition["ngram_size"], 1)
        self.assertEqual(repetition["occurrence_count"], 2)

        best = response.metadata["best_candidate"]
        self.assertIsNotNone(best)
        self.assertEqual(best["source"], "月")
        self.assertEqual(best["validation"]["final_repetition_min_ngram_size"], 1)
        self.assertTrue(best["validation"]["repetition_passed"])
        self.assertNotIn("云月", best["patched_text"])

    def test_auto_repetition_threshold_uses_single_char_only_for_short_forms(self):
        tool = RepeatRepairTool()

        short_response = tool.run(
            ToolRequest(
                "repeat_repair",
                "风月，云月。",
                metadata={
                    "auto_repetition_by_form_length": True,
                    "single_char_repetition_max_chars": 60,
                    "validation_only": True,
                },
            )
        )
        long_chars = [chr(0x4E00 + index) for index in range(70)]
        long_chars[10] = "月"
        long_chars[50] = "月"
        long_response = tool.run(
            ToolRequest(
                "repeat_repair",
                "".join(long_chars[:61]),
                metadata={
                    "auto_repetition_by_form_length": True,
                    "single_char_repetition_max_chars": 60,
                    "validation_only": True,
                },
            )
        )

        self.assertFalse(short_response.passed)
        self.assertEqual(short_response.metadata["input"]["form_char_count"], 4)
        self.assertEqual(short_response.metadata["input"]["min_ngram_size"], 1)
        self.assertEqual(short_response.metrics["final_repetition_min_ngram_size"], 1)
        self.assertEqual(short_response.metadata["repetitions"][0]["substring"], "月")
        self.assertTrue(long_response.passed)
        self.assertEqual(long_response.metadata["input"]["form_char_count"], 61)
        self.assertEqual(long_response.metadata["input"]["min_ngram_size"], 2)
        self.assertEqual(long_response.metrics["final_repetition_min_ngram_size"], 2)
        self.assertEqual(long_response.metadata["repetitions"], [])

    def test_no_repetition_returns_empty_results(self):
        tool = RepeatRepairTool()

        response = tool.run(ToolRequest("repeat_repair", "烟水无边，云影又来。", metadata={"ngram_size": 2}))

        self.assertTrue(response.passed)
        self.assertEqual(response.issues, [])
        self.assertEqual(response.metadata["repetitions"], [])
        self.assertEqual(response.metadata["candidates"], [])
        self.assertIsNone(response.metadata["best_candidate"])
        self.assertEqual(response.metrics["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
