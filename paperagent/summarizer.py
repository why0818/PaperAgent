from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .models import Chunk, SearchResult
from .text_utils import (
    char_ngram_counter,
    cosine_counter,
    normalize_text,
    query_terms,
    split_sentences,
    term_counter,
)


def summarize_chunks(
    chunks: Iterable[Chunk],
    query: str | None = None,
    max_sentences: int = 8,
) -> str:
    sentences: list[tuple[str, Chunk]] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            sentences.append((sentence, chunk))

    if not sentences:
        return "没有可汇总的文本。"

    selected_indexes = rank_sentences(
        [sentence for sentence, _ in sentences],
        query=query,
        limit=max_sentences,
    )

    bullets: list[str] = []
    for idx in selected_indexes:
        sentence, chunk = sentences[idx]
        citation = f"{chunk.paper_title}, p.{chunk.page_start}"
        bullets.append(f"- {normalize_text(sentence)} ({citation})")
    return "\n".join(bullets)


def summarize_results(
    results: Iterable[SearchResult],
    query: str,
    max_sentences: int = 8,
) -> str:
    return summarize_chunks([result.chunk for result in results], query=query, max_sentences=max_sentences)


def rank_sentences(sentences: list[str], query: str | None, limit: int) -> list[int]:
    if len(sentences) <= limit:
        return list(range(len(sentences)))

    if query:
        query_vector = char_ngram_counter(query)
        scored = [
            (cosine_counter(query_vector, char_ngram_counter(sentence)), idx)
            for idx, sentence in enumerate(sentences)
        ]
        ranked = sorted(scored, reverse=True)[:limit]
        return sorted(index for _, index in ranked)

    corpus_terms: Counter[str] = Counter()
    sentence_terms = []
    for sentence in sentences:
        counter = term_counter(sentence)
        sentence_terms.append(counter)
        corpus_terms.update(counter)

    terms = [term.lower() for term in query_terms(query or "")]
    scored: list[tuple[float, int]] = []
    for idx, (sentence, counter) in enumerate(zip(sentences, sentence_terms)):
        if terms:
            lowered = sentence.lower()
            score = sum(lowered.count(term) for term in terms)
        else:
            salience = sum(corpus_terms[term] * count for term, count in counter.items())
            length_penalty = max(1.0, sum(counter.values()) ** 0.5)
            position_bonus = 1.0 / (idx + 2)
            score = salience / length_penalty + position_bonus
        scored.append((score, idx))
    ranked = sorted(scored, reverse=True)[:limit]
    return sorted(index for _, index in ranked)
