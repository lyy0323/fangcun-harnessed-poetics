"""LLM-as-Judge literary quality evaluation.

Three-dimension scoring (1–5 scale) following the paper's evaluation design:
  1. Fluency  (流畅度)  — grammatical correctness and natural phrasing
  2. Coherence (连贯性) — thematic consistency across lines and stanzas
  3. Poetic Quality (意境) — artistic effect and emotional resonance

Plugs into the same ApiFluencyJudge / JsonMetricCache infrastructure.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from .api_metrics import JsonMetricCache, MetricApiError, _cache_key, _call_json_metric, _client_model, _parse_json_object
from .fluency import FluencyResult, FluencyScorer


LITERARY_QUALITY_SYSTEM_PROMPT = """\
你是中国古典诗词审美评鉴专家，任务是从文学审美角度评价一首机器生成的诗词作品。

## 评分维度与标尺

### 1. 流畅度（Fluency）—— 语言是否通顺自然
| 分值 | 标准 |
|------|------|
| 5 | 字词搭配精到，句法流畅自然，全篇无一处生硬或病句 |
| 4 | 整体流畅，个别字词稍欠锤炼但不碍阅读 |
| 3 | 基本可读，但有两三处搭配不当、句法别扭或语体混杂（如夹杂现代口语） |
| 2 | 多处语句不通或用词明显不当，阅读障碍较大 |
| 1 | 大量语病，几乎无法作为通顺的古典诗词阅读 |

### 2. 连贯性（Coherence）—— 主题是否贯穿、层次是否分明
| 分值 | 标准 |
|------|------|
| 5 | 主题鲜明、首尾呼应、意脉贯通，上下片（或起承转合）层次清晰 |
| 4 | 主线清晰，偶有一两处过渡不够圆融，整体仍能把握主旨 |
| 3 | 有中心主题但部分句段游离，层次感不强或转折突兀 |
| 2 | 主题模糊，多处意象堆砌与主题无关，读后不知所云 |
| 1 | 无可辨识的主题或逻辑，句与句之间几乎无关联 |

### 3. 意境（Poetic Quality）—— 艺术感染力与审美意蕴
| 分值 | 标准 |
|------|------|
| 5 | 意境深远、情景交融、余味悠长，有独到的艺术创造（意象新颖、用典精巧、情感真挚等） |
| 4 | 有较好的意境营造，情景关系处理得当，读后有一定回味 |
| 3 | 意境平平，意象组合尚可但缺乏新意，情感表达较为概念化 |
| 2 | 意象陈旧堆砌、情感空洞，或过度依赖套语（如「明月清风」「春花秋月」的简单罗列） |
| 1 | 无意境可言，纯属凑字或机械拼接 |

## 评分要求

1. 先逐维度写出简短分析（2-3句），再给出该维度的整数分值（1-5）
2. 评价时只考虑文学审美，不考虑格律合规性（格律由专门的校验工具评判）
3. 不要因为作品是机器生成的就给予宽容或苛刻的偏见，以与人类作品相同的标准评判
4. 给定主题仅作为连贯性维度的参考（作品是否围绕该主题展开），不作为流畅度或意境的评分依据

## 输出格式

只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。"""


def _literary_quality_messages(
    text: str,
    cipai: Optional[str] = None,
    prompt: Optional[str] = None,
    keyword: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_payload = {
        "task": "score_literary_quality",
        "cipai": cipai or "",
        "theme_keyword": keyword or "",
        "generation_prompt": prompt or "",
        "text": text,
        "output_schema": {
            "fluency": {
                "analysis": "string: 2-3句分析",
                "score": "integer 1-5",
            },
            "coherence": {
                "analysis": "string: 2-3句分析",
                "score": "integer 1-5",
            },
            "poetic_quality": {
                "analysis": "string: 2-3句分析",
                "score": "integer 1-5",
            },
            "overall_comment": "string: 一句话总评（可选）",
        },
    }
    return [
        {"role": "system", "content": LITERARY_QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


class ApiLiteraryQualityJudge(FluencyScorer):
    """Score literary quality through a JSON-only API judge (3 dimensions, 1-5 scale)."""

    DIMENSION_NAMES = ("fluency", "coherence", "poetic_quality")

    def __init__(
        self,
        client: Any,
        cache: Optional[JsonMetricCache] = None,
        metric_version: str = "literary_quality.api_judge.v1",
        fallback: Optional[FluencyScorer] = None,
    ):
        self.client = client
        self.cache = cache
        self.metric_version = metric_version
        self.fallback = fallback

    def score_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FluencyResult:
        keyword = (metadata or {}).get("keyword") or ""
        messages = _literary_quality_messages(text, cipai=cipai, prompt=prompt, keyword=keyword)
        try:
            payload, api_metadata = _call_json_metric(
                self.client,
                messages,
                cache=self.cache,
                cache_key=_cache_key(self.metric_version, text, cipai, prompt),
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            result = self.fallback.score_result(text, cipai=cipai, prompt=prompt, metadata=metadata)
            result.metadata = {**result.metadata, "api_error": str(exc), "api_fallback": True}
            return result

        dimensions = {}
        analyses = {}
        for dim in self.DIMENSION_NAMES:
            dim_data = payload.get(dim, {})
            if isinstance(dim_data, Mapping):
                raw_score = dim_data.get("score")
                analysis = str(dim_data.get("analysis", ""))
            elif isinstance(dim_data, (int, float)):
                raw_score = dim_data
                analysis = ""
            else:
                raw_score = None
                analysis = ""
            if raw_score is not None:
                score_int = max(1, min(5, int(round(float(raw_score)))))
                dimensions[dim] = score_int / 5.0
                analyses[f"{dim}_analysis"] = analysis
                analyses[f"{dim}_raw_score"] = score_int

        if not dimensions:
            raise MetricApiError("Literary quality API response must include at least one scored dimension")

        composite = sum(dimensions.values()) / len(dimensions)
        overall_comment = str(payload.get("overall_comment", ""))

        return FluencyResult(
            score=composite,
            dimensions=dimensions,
            issues=[],
            backend="literary_quality_judge",
            metadata={
                **api_metadata,
                **analyses,
                "overall_comment": overall_comment,
                "metric_version": self.metric_version,
            },
        )


__all__ = ["ApiLiteraryQualityJudge", "LITERARY_QUALITY_SYSTEM_PROMPT"]
