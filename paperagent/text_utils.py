from __future__ import annotations

import re
from collections import Counter
from math import sqrt


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "may",
    "our",
    "their",
    "these",
    "those",
    "there",
    "where",
    "using",
    "used",
    "also",
    "such",
    "into",
    "over",
    "than",
    "between",
    "because",
    "paper",
    "study",
    "article",
    "figure",
    "table",
    "supplementary",
    "license",
    "published",
    "https",
    "http",
    "doi",
    "org",
    "wiley",
    "springer",
    "crossmark",
    "creative",
    "commons",
    "abstract",
    "introduction",
    "results",
    "discussion",
    "conclusion",
    "研究",
    "方法",
    "本文",
    "一个",
    "我们",
}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[。！？.!?])\s+|(?<=[。！？.!?])", text)
    sentences = [part.strip() for part in parts if len(part.strip()) >= 20]
    if sentences:
        return sentences
    return [text]


def extract_terms(text: str, limit: int = 12) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", lowered)
    cleaned: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) > 24 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            cleaned.extend(token[i : i + 4] for i in range(0, len(token) - 3, 4))
        else:
            cleaned.append(token)
    return [term for term, _ in Counter(cleaned).most_common(limit)]


def query_terms(query: str) -> list[str]:
    terms = extract_terms(query, limit=24)
    if terms:
        return terms
    return [part for part in re.split(r"\s+", query.lower()) if part]


def search_terms(text: str) -> list[str]:
    lowered = normalize_text(text).lower()
    raw_tokens = re.findall(r"[a-z][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", lowered)
    terms: list[str] = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 3:
            terms.extend(token[i : i + 2] for i in range(0, len(token) - 1))
    return terms


def term_counter(text: str) -> Counter[str]:
    return Counter(search_terms(text))


def char_ngram_counter(text: str, min_n: int = 3, max_n: int = 5) -> Counter[str]:
    compact = re.sub(r"\s+", " ", normalize_text(text).lower())
    counter: Counter[str] = Counter()
    if not compact:
        return counter
    for size in range(min_n, max_n + 1):
        if len(compact) < size:
            continue
        counter.update(compact[i : i + size] for i in range(0, len(compact) - size + 1))
    return counter


def cosine_counter(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    if dot <= 0:
        return 0.0
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)
