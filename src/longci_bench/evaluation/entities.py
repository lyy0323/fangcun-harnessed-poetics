"""Entity richness backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional

from ..text import normalized_cjk


DEFAULT_ENTITY_TERMS = [
    "明月",
    "残月",
    "东风",
    "西风",
    "江水",
    "归舟",
    "故人",
    "天涯",
    "梅花",
    "杨柳",
    "春山",
    "秋夜",
]


@dataclass
class EntityExtractionResult:
    terms: List[str]
    items: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    def extract(self, text: str) -> List[str]:
        return self.extract_result(text).terms

    def extract_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        raise NotImplementedError


class HeuristicEntityExtractor(EntityExtractor):
    def __init__(self, terms: Optional[Iterable[str]] = None):
        self.terms = sorted(set(terms or DEFAULT_ENTITY_TERMS), key=len, reverse=True)

    def extract_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        entities = []
        for term in self.terms:
            if term in text:
                entities.append(term)

        compact = normalized_cjk(text)
        # Fallback candidate nouns: repeated or salient two-character chunks.
        for match in re.finditer(r"[\u4e00-\u9fff]{2}", compact):
            candidate = match.group(0)
            if candidate not in entities and not _looks_functional(candidate):
                entities.append(candidate)
        terms = sorted(set(entities))
        items = [{"text": term, "category": "heuristic_candidate", "source": "heuristic"} for term in terms]
        return EntityExtractionResult(
            terms=terms,
            items=items,
            backend="heuristic",
            metadata={"configured_term_count": len(self.terms)},
        )


class BertEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the BERT extra first: pip install -e '.[bert]'") from exc
        self.pipeline = pipeline("token-classification", model=model_name, aggregation_strategy="simple")

    def extract_result(
        self,
        text: str,
        cipai: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityExtractionResult:
        results = self.pipeline(text)
        items: List[Dict[str, Any]] = []
        for item in results:
            word = str(item.get("word", "")).replace(" ", "")
            if word:
                items.append(
                    {
                        "text": word,
                        "category": str(item.get("entity_group") or item.get("entity") or "entity"),
                        "score": item.get("score"),
                        "source": "bert",
                    }
                )
        terms = sorted({str(item["text"]) for item in items})
        return EntityExtractionResult(
            terms=terms,
            items=items,
            backend="bert",
            metadata={"model_name": self.pipeline.model.name_or_path},
        )


def _looks_functional(token: str) -> bool:
    functional = set("之乎者也兮矣焉哉而其所何不无有又更")
    return all(char in functional for char in token)
