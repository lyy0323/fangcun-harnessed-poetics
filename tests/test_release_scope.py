import argparse
import importlib.util
import json
from pathlib import Path
import unittest

from longci_bench.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


class ReleaseScopeTests(unittest.TestCase):
    def test_cli_does_not_expose_rl_commands(self):
        parser = build_parser()
        subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
        self.assertEqual(len(subparsers), 1)
        self.assertNotIn("rl-reward", subparsers[0].choices)

    def test_training_package_is_not_published(self):
        self.assertIsNone(importlib.util.find_spec("longci_bench.training"))

    def test_writing_and_tool_fitting_configs_are_separate(self):
        writing_config = ROOT / "configs/model.writing.openai_compatible.example.json"
        fitting_config = ROOT / "configs/model.tool_fitting.openai_compatible.example.json"
        style_config = json.loads((ROOT / "configs/style_tools.api.example.json").read_text(encoding="utf-8"))
        metric_config = json.loads(
            (ROOT / "configs/evaluation.metrics.api.example.json").read_text(encoding="utf-8")
        )

        self.assertTrue(writing_config.exists())
        self.assertTrue(fitting_config.exists())
        self.assertNotEqual(writing_config.name, fitting_config.name)
        self.assertEqual(style_config["model_config"], "configs/model.tool_fitting.openai_compatible.local.json")
        self.assertEqual(
            metric_config["api"]["model_config"],
            "configs/model.tool_fitting.openai_compatible.local.json",
        )


if __name__ == "__main__":
    unittest.main()
