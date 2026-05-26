"""Repeated imagery recall tool."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

from ..schemas import ToolRequest, ToolResponse
from ..text import find_literal_spans
from .base import Tool


DEFAULT_IMAGERY_LEXICON: Dict[str, List[str]] = {
    "moon": ["明月", "残月", "月"],
    "wind": ["东风", "西风", "风"],
    "water": ["江水", "江", "水", "溪"],
    "flower": ["梅", "柳", "花"],
}


class ImageryRecallTool(Tool):
    name = "imagery"
    description = "Post-generation check for repeated imagery terms or imagery categories."
    tool_group = "post_check_tool"
    usage_stage = "post_check"
    model_exposure = "post_check"
    tags = ("imagery", "after_generation")

    def __init__(self, lexicon: Optional[Mapping[str, Iterable[str]]] = None):
        source = lexicon or DEFAULT_IMAGERY_LEXICON
        self.lexicon = {category: sorted(set(terms), key=len, reverse=True) for category, terms in source.items()}

    def run(self, request: ToolRequest) -> ToolResponse:
        issues = []
        category_hits = {}

        for category, terms in self.lexicon.items():
            hits = []
            occupied = set()
            for term in terms:
                for span in find_literal_spans(request.text, term):
                    covered = set(range(span.start, span.end))
                    if covered & occupied:
                        continue
                    occupied.update(covered)
                    hits.append({"term": term, "span": span.to_dict()})
            if hits:
                category_hits[category] = hits

        for category, hits in category_hits.items():
            literal_terms = sorted({hit["term"] for hit in hits})
            if len(hits) > 1:
                issues.append(
                    {
                        "type": "repeated_imagery",
                        "message": "Repeated imagery category detected.",
                        "category": category,
                        "terms": literal_terms,
                        "occurrence_count": len(hits),
                        "occurrences": hits,
                    }
                )

        return ToolResponse(
            self.name,
            passed=not issues,
            issues=issues,
            metrics={
                "imagery_category_count": len(category_hits),
                "repeated_imagery_category_count": len(issues),
            },
        )
