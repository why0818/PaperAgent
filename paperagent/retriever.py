from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from .models import Chunk, SearchResult
from .text_utils import char_ngram_counter, cosine_counter, query_terms, term_counter


class HybridRetriever:
    """Offline advanced retriever for paper RAG.

    Ranking combines BM25 lexical matching, exact keyword hits, character
    n-gram semantic similarity, metadata boosts, and MMR diversification. It is
    intentionally pure Python so the app does not depend on fragile compiled
    numerical packages in the user's environment.
    """

    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self.texts = [chunk.text for chunk in self.chunks]
        self.term_vectors = [term_counter(text) for text in self.texts]
        self.semantic_vectors = [char_ngram_counter(text) for text in self.texts]
        self.doc_lengths = [sum(vector.values()) for vector in self.term_vectors]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )
        self.document_frequency = self._document_frequency(self.term_vectors)
        self.total_chunks = max(1, len(self.chunks))

    def search(
        self,
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []

        candidate_indexes = self._candidate_indexes(document_ids)
        if not candidate_indexes:
            return []

        bm25_scores = self._bm25_scores(query)
        exact_scores = self._exact_scores(query)
        semantic_scores = self._semantic_scores(query)
        keyword_scores = [
            max(bm25_scores[index], exact_scores[index])
            for index in range(len(self.chunks))
        ]

        if mode == "keyword":
            scores = keyword_scores
        elif mode == "semantic":
            scores = semantic_scores
        else:
            scores = [
                0.56 * bm25_scores[index]
                + 0.26 * semantic_scores[index]
                + 0.13 * exact_scores[index]
                + 0.05 * self._metadata_boost(query, self.chunks[index])
                for index in range(len(self.chunks))
            ]

        ranked_pool = sorted(
            candidate_indexes,
            key=lambda index: scores[index],
            reverse=True,
        )[: max(top_k * 6, 30)]
        selected = self._mmr_select(ranked_pool, scores, top_k=top_k)

        results: list[SearchResult] = []
        for index in selected:
            if scores[index] <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                    keyword_score=float(keyword_scores[index]),
                    semantic_score=float(semantic_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                )
            )
        return results

    def _candidate_indexes(self, document_ids: list[str] | None) -> list[int]:
        if not document_ids:
            return list(range(len(self.chunks)))
        allowed = set(document_ids)
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.document_id in allowed
        ]

    def _bm25_scores(self, query: str) -> list[float]:
        terms = query_terms(query)
        if not terms:
            return [0.0 for _ in self.chunks]
        query_counts = Counter(terms)
        k1 = 1.55
        b = 0.72
        scores: list[float] = []
        for index, vector in enumerate(self.term_vectors):
            score = 0.0
            doc_len = max(1, self.doc_lengths[index])
            for term, query_weight in query_counts.items():
                freq = vector.get(term.lower(), 0)
                if freq <= 0:
                    continue
                df = self.document_frequency.get(term.lower(), 0)
                idf = math.log(1 + (self.total_chunks - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / max(1.0, self.avgdl))
                score += query_weight * idf * (freq * (k1 + 1)) / denom
            scores.append(score)
        return self._normalize(scores)

    def _exact_scores(self, query: str) -> list[float]:
        scores = [self._exact_keyword_score(query, text) for text in self.texts]
        return self._normalize(scores)

    def _semantic_scores(self, query: str) -> list[float]:
        query_vector = char_ngram_counter(query)
        scores = [
            cosine_counter(query_vector, vector)
            for vector in self.semantic_vectors
        ]
        return self._normalize(scores)

    def _metadata_boost(self, query: str, chunk: Chunk) -> float:
        haystack = f"{chunk.paper_title} {chunk.section} {' '.join(chunk.keywords)}".lower()
        terms = query_terms(query)
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term.lower() in haystack)
        return min(1.0, hits / max(1, len(terms)))

    def _mmr_select(
        self,
        ranked_pool: list[int],
        scores: list[float],
        top_k: int,
        diversity: float = 0.26,
    ) -> list[int]:
        selected: list[int] = []
        remaining = list(ranked_pool)
        while remaining and len(selected) < top_k:
            best_index = remaining[0]
            best_value = -1.0
            for index in remaining:
                if not selected:
                    value = scores[index]
                else:
                    similarity = max(
                        cosine_counter(self.semantic_vectors[index], self.semantic_vectors[other])
                        for other in selected
                    )
                    value = (1 - diversity) * scores[index] - diversity * similarity
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(best_index)
            remaining.remove(best_index)
        return selected

    @staticmethod
    def _document_frequency(vectors: list[Counter[str]]) -> Counter[str]:
        df: Counter[str] = Counter()
        for vector in vectors:
            df.update(vector.keys())
        return df

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        max_score = max(scores)
        if max_score <= 0 or math.isnan(max_score):
            return [0.0 for _ in scores]
        return [score / max_score for score in scores]

    @staticmethod
    def _exact_keyword_score(query: str, text: str) -> float:
        lowered = text.lower()
        terms = query_terms(query)
        if not terms:
            return 0.0
        score = 0.0
        for term in terms:
            count = lowered.count(term.lower())
            if count:
                score += min(4.0, 1.0 + math.log(count + 1))
        return score / max(1.0, len(terms))
