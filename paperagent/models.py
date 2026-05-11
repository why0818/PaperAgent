from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .text_utils import query_terms


@dataclass
class PaperDocument:
    id: str
    title: str
    filename: str
    stored_path: str
    sha256: str
    pages: int
    chunks: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperDocument":
        return cls(**data)


@dataclass
class Chunk:
    id: str
    document_id: str
    paper_title: str
    page_start: int
    page_end: int
    text: str
    keywords: list[str] = field(default_factory=list)
    section: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        data = dict(data)
        data.setdefault("section", "")
        data.setdefault("chunk_index", 0)
        return cls(**data)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    keyword_score: float
    semantic_score: float
    bm25_score: float = 0.0

    def to_source_dict(self, snippet_chars: int = 900, query: str | None = None) -> dict[str, Any]:
        text = self.chunk.text.strip()
        snippet = make_snippet(text, query=query, snippet_chars=snippet_chars)
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "paper_title": self.chunk.paper_title,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "section": self.chunk.section,
            "chunk_index": self.chunk.chunk_index,
            "score": float(self.score),
            "keyword_score": float(self.keyword_score),
            "semantic_score": float(self.semantic_score),
            "bm25_score": float(self.bm25_score),
            "snippet": snippet,
            "text": text,
            "keywords": self.chunk.keywords,
        }


def make_snippet(text: str, query: str | None = None, snippet_chars: int = 900) -> str:
    text = " ".join(text.split())
    if len(text) <= snippet_chars:
        return text
    terms = query_terms(query or "")
    first_hit = -1
    lowered = text.lower()
    for term in terms:
        hit = lowered.find(term.lower())
        if hit >= 0 and (first_hit < 0 or hit < first_hit):
            first_hit = hit
    if first_hit < 0:
        return text[:snippet_chars].rstrip() + "..."
    start = max(0, first_hit - snippet_chars // 3)
    end = min(len(text), start + snippet_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
