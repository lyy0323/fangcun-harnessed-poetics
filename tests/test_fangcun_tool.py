import tempfile
import unittest
from pathlib import Path

from longci_bench.schemas import ToolRequest
from longci_bench.tools import FangcunDataClient, FangcunProsodyTool, build_default_registry


class FangcunProsodyToolTests(unittest.TestCase):
    def test_maps_fangcun_result_to_tool_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            tool = FangcunProsodyTool(source_path=str(source))

            response = tool.run(ToolRequest("prosody", "风月。\n风雨。", cipai="卜算子"))

        self.assertFalse(response.passed)
        self.assertEqual(response.metrics["violation_count"], 1)
        self.assertEqual(response.metrics["checked_chars"], 4)
        self.assertEqual(response.metadata["closest_rule"]["name"], "卜算子_格一")
        self.assertEqual(response.metadata["rule_name"], "卜算子")
        self.assertEqual(response.issues[0]["type"], "tone")
        self.assertEqual(response.issues[0]["span"]["line"], 1)
        self.assertEqual(response.issues[0]["span"]["column"], 2)
        self.assertEqual(response.issues[0]["actual"], "月")
        self.assertEqual(response.issues[0]["position_1based"], 2)
        self.assertEqual(response.issues[0]["line"], 1)
        self.assertEqual(response.issues[0]["column"], 2)
        self.assertEqual(response.issues[0]["text"], "月")
        self.assertEqual(response.metadata["issue_summary"][0]["position_1based"], 2)
        self.assertEqual(response.metadata["issue_summary"][0]["message"], "应为仄, 实为平")

    def test_registry_can_override_placeholder_prosody_with_fangcun(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(fangcun_path=str(source))

            response = registry.run(ToolRequest("prosody", "风月。", cipai="卜算子"))

        self.assertEqual(response.metadata["source"], "poetics_data")
        self.assertIn("phrase_suggest", registry.names())

    def test_data_client_char_lookup_uses_rhyme_book_and_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            client = FangcunDataClient(str(source))

            result = client.lookup_chars("花", book="Pingshuiyun", include_definitions=True)

        self.assertEqual(result[0]["char"], "花")
        self.assertEqual(result[0]["rhyme_categories"][0]["name"], "六麻")
        self.assertEqual(result[0]["definitions"][0]["py"], "hua")

    def test_auxiliary_tools_return_canonical_tool_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(fangcun_path=str(source))

            char_response = registry.run(
                ToolRequest("char_lookup", "花", metadata={"book": "Pingshuiyun"})
            )
            rhyme_response = registry.run(
                ToolRequest("rhyme_lookup", "六麻", metadata={"book": "Pingshuiyun", "include": ["neighbor"]})
            )
            rule_response = registry.run(
                ToolRequest("rule_lookup", "卜算子", metadata={"genre": "Ci"})
            )
            phrase_response = registry.run(
                ToolRequest("phrase_suggest", "明", metadata={"mode": "head", "length": "2"})
            )
            allusion_response = registry.run(
                ToolRequest("allusion_search", "月", metadata={"limit": 3})
            )

        self.assertTrue(char_response.passed)
        self.assertEqual(char_response.metadata["source"], "poetics_data")
        self.assertEqual(char_response.metadata["results"], char_response.metadata["characters"])
        self.assertEqual(char_response.metadata["characters"][0]["char"], "花")
        self.assertTrue(rhyme_response.passed)
        self.assertEqual(rhyme_response.metadata["results"][0], rhyme_response.metadata["result"])
        self.assertEqual(rhyme_response.metadata["result"]["primary"]["category_name"], "六麻")
        self.assertTrue(rule_response.passed)
        self.assertEqual(rule_response.metadata["results"], rule_response.metadata["rules"])
        self.assertEqual(rule_response.metadata["rules"][0]["cipai"], "卜算子")
        self.assertIn("constraints", rule_response.metadata["rules"][0])
        self.assertTrue(phrase_response.passed)
        self.assertEqual(phrase_response.metadata["results"], phrase_response.metadata["suggestions"])
        self.assertEqual(phrase_response.metadata["suggestions"][0]["phrase"], "明月")
        self.assertTrue(allusion_response.passed)
        self.assertEqual(allusion_response.metadata["results"], allusion_response.metadata["entries"])
        self.assertEqual(allusion_response.metadata["entries"][0]["w"], "月中桂")

    def test_base_tools_expose_lightweight_rule_judou_and_prosody(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(fangcun_path=str(source))

            rule_response = registry.run(ToolRequest("get_rule", "卜算子"))
            judou_pass = registry.run(ToolRequest("check_judou", "风月，风雨。", cipai="卜算子"))
            judou_fail = registry.run(ToolRequest("check_judou", "风月风，雨。", cipai="卜算子"))
            prosody_response = registry.run(ToolRequest("check_prosody", "风月。", cipai="卜算子"))

        self.assertIn("get_rule", registry.names())
        self.assertIn("check_judou", registry.names())
        self.assertIn("check_prosody", registry.names())
        self.assertTrue(rule_response.passed)
        self.assertEqual(rule_response.metadata["rule"]["sentence_char_counts"], [2, 2])
        self.assertEqual(rule_response.metadata["rule"]["internal_sentence_breaks_1based"], [2])
        self.assertTrue(judou_pass.passed)
        self.assertFalse(judou_fail.passed)
        self.assertEqual(judou_fail.issues[0]["type"], "missing_sentence_break")
        self.assertEqual(prosody_response.tool_name, "check_prosody")
        self.assertEqual(prosody_response.metadata["base_tool"], "check_prosody")

    def test_tool_contract_groups_current_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _fake_fangcun_source(tmp)
            registry = build_default_registry(poetics_path=str(source))
            contract = registry.contract()

        self.assertIn("base_tool", contract["tool_groups"])
        self.assertIn("get_rule", contract["tool_groups"]["base_tool"]["tools"])
        self.assertIn("rule_lookup", contract["tool_groups"]["extra_tool"]["tools"])
        self.assertIn("char_lookup", contract["tool_groups"]["lexical_tool"]["tools"])
        self.assertIn("repetition", contract["tool_groups"]["post_check_tool"]["tools"])
        check_judou = next(tool for tool in contract["tools"] if tool["name"] == "check_judou")
        self.assertEqual(check_judou["tool_group"], "base_tool")
        self.assertEqual(check_judou["model_exposure"], "base")


def _fake_fangcun_source(tmp: str) -> Path:
    source = Path(tmp)
    config = source / "static" / "config"
    config.mkdir(parents=True)
    _write_json(config / "t2s_map.json", {"華": "花"})
    _write_json(
        config / "char_dict.json",
        {
            "花": {
                "tones": ["平"],
                "rhymes": {"Pingshuiyun": ["六麻"], "Cilinzhengyun": ["第10部_平"]},
            },
            "风": {
                "tones": ["平"],
                "rhymes": {"Pingshuiyun": ["六麻"], "Cilinzhengyun": ["第10部_平"]},
            },
            "月": {
                "tones": ["平"],
                "rhymes": {"Pingshuiyun": ["六麻"], "Cilinzhengyun": ["第10部_平"]},
            },
            "雨": {
                "tones": ["入"],
                "rhymes": {"Pingshuiyun": ["六月"], "Cilinzhengyun": ["第18部_仄"]},
            },
        },
    )
    _write_json(
        config / "char_definitions.json",
        {"花": [{"py": "hua", "defs": [{"d": "flower"}]}]},
    )
    _write_json(
        config / "rhyme_books.json",
        {
            "Pingshuiyun": {
                "name": "Pingshuiyun",
                "categories": {
                    "六麻": {
                        "name": "六麻",
                        "tone_type": "P",
                        "characters": ["花", "家"],
                        "relations": {"neighbor": ["七阳"]},
                    },
                    "七阳": {
                        "name": "七阳",
                        "tone_type": "P",
                        "characters": ["香"],
                        "relations": {},
                    },
                },
            },
            "Cilinzhengyun": {
                "name": "Cilinzhengyun",
                "categories": {
                    "第10部_平": {
                        "name": "第10部_平",
                        "tone_type": "P",
                        "characters": ["花"],
                        "relations": {},
                    },
                    "第18部_仄": {
                        "name": "第18部_仄",
                        "tone_type": "Z",
                        "characters": ["月"],
                        "relations": {},
                    },
                },
            },
        },
    )
    _write_json(
        config / "ci_rules.json",
        [
            {
                "name": "卜算子_格一",
                "genre": "Ci",
                "cipai": "卜算子",
                "char_count": 4,
                "tone_pattern": [
                    {"tone": "P", "comment": None},
                    {"tone": "Z", "comment": "句"},
                    {"tone": "P", "comment": None},
                    {"tone": "Z", "comment": None}
                ],
                "rhyme_rule": {},
            }
        ],
    )
    _write_json(
        config / "shi_rules.json",
        [
            {
                "name": "五绝仄起",
                "genre": "Shi",
                "cipai": "Wujue",
                "char_count": 20,
                "tone_pattern": [],
                "rhyme_rule": {},
            }
        ],
    )
    _write_json(
        config / "phrase_head.json",
        {"明": {"2": {"P": [["明月", 10]], "Z": [["明灭", 2]]}}},
    )
    _write_json(
        config / "phrase_tail.json",
        {"月": {"2": {"P": [["明月", 10]], "Z": []}}},
    )
    _write_json(config / "phrase_pairs.json", {"月": [["风", 100]], "风": [["月", 100]]})
    _write_json(config / "phrase_tongwei.json", {"月": [["花", 7]]})
    _write_json(config / "allusion_index.json", {"月": [0]})
    _write_json(
        config / "allusion_entries.json",
        [
            {
                "id": 0,
                "w": "月中桂",
                "d": "月宫桂树。",
                "src": "test",
                "src_text": "test",
                "related": [],
                "examples": [],
                "rc": 1,
            }
        ],
    )
    return source


def _write_json(path: Path, payload) -> None:
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
