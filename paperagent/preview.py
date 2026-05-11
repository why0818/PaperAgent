from __future__ import annotations

import re

from .models import Chunk


REFERENCE_SECTION_RE = re.compile(
    r"\b(references?|bibliography|acknowledg(e)?ments?|data availability|supporting information|supplementary)\b",
    re.I,
)


def build_preview_chunks(chunks: list[Chunk]) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        if is_reference_chunk(chunk):
            continue
        text = clean_preview_text(chunk.text)
        text = strip_reference_tail(text)
        if len(text) < 80:
            continue
        if looks_like_reference_text(text):
            continue
        key = re.sub(r"\W+", "", text.lower())[:260]
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "paper_title": chunk.paper_title,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "keywords": chunk.keywords,
                "snippet": shorten(text, 760),
                "summary": summarize_preview_text(text, chunk.section),
            }
        )
    return items


def is_reference_chunk(chunk: Chunk) -> bool:
    section = chunk.section or ""
    text = chunk.text or ""
    if REFERENCE_SECTION_RE.search(section):
        return True
    numbered = re.match(r"^\s*(\d{1,3})[\).]\s+", section)
    if numbered and int(numbered.group(1)) > 8:
        return True
    cleaned = re.sub(r"^\[Section:\s*[^\]]+\]\s*", "", text, flags=re.I).strip()
    if re.match(r"^(references?|bibliography)\b", cleaned, re.I):
        return True
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    citation_like = sum(bool(re.match(r"^\[?\d{1,3}\]?[\).,]\s+", line)) for line in lines[:12])
    if len(lines) >= 6 and citation_like / len(lines[:12]) > 0.45:
        return True
    if looks_like_reference_text(cleaned[:1600]):
        return True
    return False


def clean_preview_text(text: str) -> str:
    text = re.sub(r"^\[Section:\s*[^\]]+\]\s*", "", text or "", flags=re.I).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\bReseaR\s+ch\b", "Research", text)
    text = re.sub(r"\baR\s+ticles\b", "articles", text)
    text = re.sub(r"\bReseaRch\b", "Research", text)
    text = re.sub(r"\bMethaNe\b", "Methane", text)
    text = re.sub(r"\beMissiONs\b", "emissions", text)
    text = re.sub(r"\beMissiON\b", "emission", text)
    text = re.sub(r"\s+([,.;:?!%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = " ".join(fix_mixed_case_token(token) for token in text.split())
    return text.strip()


def strip_reference_tail(text: str) -> str:
    patterns = [
        r"\bReferences?\b",
        r"\bBibliography\b",
        r"\bAcknowledg(e)?ments?\b",
        r"\bSupplementary Materials?\b",
        r"\bData Availability\b",
        r"\bFunding\b",
        r"\bCompeting interests?\b",
    ]
    cut = len(text)
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match and match.start() > 180:
            cut = min(cut, match.start())

    citation_run = re.search(
        r"(\b\d{1,3}[\).]\s+[A-Z][^。！？\n]{20,240}?\(\d{4}\)[^。！？\n]{0,220})"
        r"(\s+\d{1,3}[\).]\s+[A-Z][^。！？\n]{20,240}?\(\d{4}\))",
        text,
    )
    if citation_run and citation_run.start() > 120:
        cut = min(cut, citation_run.start())

    doi_run = re.search(r"\b\d{1,3}[\).]\s+[^。！？\n]{10,220}\bdoi\b", text, re.I)
    if doi_run and doi_run.start() > 120:
        cut = min(cut, doi_run.start())

    return text[:cut].strip()


def looks_like_reference_text(text: str) -> bool:
    sample = " ".join((text or "").split())[:1800]
    if not sample:
        return False
    numbered_refs = re.findall(r"\b\d{1,3}[\).]\s+[A-Z][^。！？]{10,180}?\(\d{4}\)", sample)
    et_al_refs = re.findall(r"\b\d{1,3}[\).]\s+[^。！？]{0,120}\bet al\.", sample, re.I)
    doi_refs = re.findall(r"\bdoi\b|https?://doi\.org", sample, re.I)
    starts_with_ref = bool(re.match(r"^\s*\d{1,3}[\).]\s+.+(\(\d{4}\)|et al\.|doi|https?://)", sample, re.I))
    return starts_with_ref or len(numbered_refs) >= 2 or len(et_al_refs) >= 2 or (len(doi_refs) >= 2 and len(et_al_refs) >= 1)


def fix_mixed_case_token(token: str) -> str:
    match = re.match(r"^([A-Za-z]{5,})([.,;:?!)]*)$", token)
    if not match:
        return token
    word, punct = match.groups()
    if word[:2].isupper():
        return token
    uppers = sum(char.isupper() for char in word)
    lowers = sum(char.islower() for char in word)
    if uppers >= 2 and lowers >= 2:
        if word[0].isupper():
            return word[:1].upper() + word[1:].lower() + punct
        return word.lower() + punct
    return token


def summarize_preview_text(text: str, section: str = "") -> str:
    lowered = f"{section} {text[:600]}".lower()
    if "abstract" in lowered or "front matter" in lowered:
        return "概括论文主题、研究对象、核心问题与主要发现。"
    if any(word in lowered for word in ["introduction", "background", "motivation"]):
        return "说明研究背景、已有工作的不足，以及本文要解决的问题。"
    if any(word in lowered for word in ["method", "materials", "model", "algorithm", "inversion"]):
        return "介绍数据来源、模型流程、实验设计或关键计算方法。"
    if any(word in lowered for word in ["result", "experiment", "evaluation", "performance"]):
        return "总结实验或观测结果，并说明这些结果支持了哪些结论。"
    if any(word in lowered for word in ["discussion", "limitation", "uncertainty"]):
        return "讨论结果含义、不确定性、局限性以及与已有研究的关系。"
    if any(word in lowered for word in ["conclusion", "summary"]):
        return "归纳论文结论、贡献和后续研究方向。"
    sentence = first_sentence(text)
    if sentence:
        return "本段主要说明：" + shorten(sentence, 86)
    return "概括该段涉及的论文内容与论证线索。"


def first_sentence(text: str) -> str:
    match = re.search(r"(.{45,220}?[.!?。！？])\s", text)
    if match:
        return match.group(1).strip()
    return text[:120].strip()


def shorten(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
