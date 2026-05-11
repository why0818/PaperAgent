from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Chunk
from .pdf_loader import PdfPage
from .text_utils import extract_terms, normalize_text


CHUNKING_VERSION = "section-paragraph-v2"


@dataclass
class Paragraph:
    page: int
    text: str
    section: str


def chunk_pdf_pages(
    pages: list[PdfPage],
    document_id: str,
    paper_title: str,
    chunk_size: int = 2600,
    overlap_paragraphs: int = 2,
) -> list[Chunk]:
    paragraphs = extract_paragraphs(pages)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer: list[Paragraph] = []
    buffer_chars = 0
    counter = 1

    for paragraph in paragraphs:
        paragraph_len = len(paragraph.text)
        same_section = not buffer or paragraph.section == buffer[-1].section
        should_flush = (
            buffer
            and buffer_chars + paragraph_len > chunk_size
            and (buffer_chars >= chunk_size * 0.62 or not same_section)
        )
        if should_flush:
            chunk = build_chunk(buffer, document_id, paper_title, counter)
            if chunk:
                chunks.append(chunk)
                counter += 1
            buffer = buffer[-overlap_paragraphs:] if overlap_paragraphs else []
            buffer_chars = sum(len(item.text) for item in buffer)

        buffer.append(paragraph)
        buffer_chars += paragraph_len

    chunk = build_chunk(buffer, document_id, paper_title, counter)
    if chunk:
        chunks.append(chunk)
    return chunks


def extract_paragraphs(pages: list[PdfPage]) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    current_section = "Front Matter"
    seen: set[tuple[int, str]] = set()

    for page in pages:
        page_paragraphs = paragraphs_from_page(page.text)
        for raw in page_paragraphs:
            text = cleanup_paragraph(raw)
            if not useful_paragraph(text):
                continue
            inline_section = split_inline_section_heading(text)
            if inline_section:
                current_section, text = inline_section
                text = cleanup_paragraph(text)
                if not useful_paragraph(text):
                    continue
            if is_section_heading(text):
                current_section = text[:120]
                continue
            key = (page.page_number, re.sub(r"\W+", "", text.lower())[:120])
            if key in seen:
                continue
            seen.add(key)
            paragraphs.append(
                Paragraph(page=page.page_number, text=text, section=current_section)
            )
    return paragraphs


def paragraphs_from_page(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(join_lines(buffer))
            buffer.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if should_skip_line(line):
            continue
        if is_section_heading(line):
            flush()
            paragraphs.append(line)
            continue
        if buffer and starts_new_paragraph(line, buffer[-1]):
            flush()
        buffer.append(line)
    flush()
    return paragraphs


def build_chunk(
    paragraphs: list[Paragraph],
    document_id: str,
    paper_title: str,
    counter: int,
) -> Chunk | None:
    paragraphs = [paragraph for paragraph in paragraphs if paragraph.text.strip()]
    if not paragraphs:
        return None
    section = dominant_section(paragraphs)
    body = "\n\n".join(paragraph.text for paragraph in paragraphs)
    if section and section not in {"Front Matter"}:
        text = f"[Section: {section}]\n\n{body}"
    else:
        text = body
    return Chunk(
        id=f"{document_id}-c{counter:05d}",
        document_id=document_id,
        paper_title=paper_title,
        page_start=min(paragraph.page for paragraph in paragraphs),
        page_end=max(paragraph.page for paragraph in paragraphs),
        text=text,
        keywords=extract_terms(text),
        section=section,
        chunk_index=counter,
    )


def dominant_section(paragraphs: list[Paragraph]) -> str:
    for paragraph in reversed(paragraphs):
        if paragraph.section:
            return paragraph.section
    return "Front Matter"


def cleanup_paragraph(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = text.replace("Š\\", "-")
    text = re.sub(r"\s+([,.;:?!%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line
            continue
        if text.endswith("-") and line and line[0].islower():
            text = text[:-1] + line
        elif re.search(r"[/:–—-]\s*$", text):
            text += " " + line
        elif line and line[0] in ",.;:%)]":
            text += line
        else:
            text += " " + line
    return text


def should_skip_line(line: str) -> bool:
    stripped = line.strip()
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if len(stripped) <= 2:
        return True
    lowered = stripped.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    return False


def starts_new_paragraph(line: str, previous: str) -> bool:
    if re.match(r"^([A-Z][a-z]+ ){2,}[A-Z]?[a-z]*$", line) and len(line) < 80:
        return True
    if re.match(r"^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)\b", line, re.I):
        return True
    if previous.endswith((".", "。", "!", "?", "；", ";")) and re.match(r"^[A-Z0-9]", line):
        return len(previous) > 80
    return False


def is_section_heading(text: str) -> bool:
    stripped = cleanup_paragraph(text)
    if len(stripped) > 130:
        return False
    if len(stripped.split()) > 14:
        return False
    if re.match(r"^(Abstract|Plain Language Summary|Introduction|Background|Related Work|Materials and Methods|Methods?|Methodology|Results?|Discussion|Conclusion|Conclusions|References|Acknowledg(e)?ments|Data Availability|Supporting Information)\s+.+", stripped, re.I):
        return False
    patterns = [
        r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z ,:/()-]+$",
        r"^(Abstract|Plain Language Summary|Introduction|Background|Related Work|Materials and Methods|Methods?|Methodology|Results?|Discussion|Conclusion|Conclusions|References|Acknowledg(e)?ments|Data Availability|Supporting Information)\.?$",
    ]
    return any(re.match(pattern, stripped) for pattern in patterns)


def split_inline_section_heading(text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(Abstract|Plain Language Summary|Introduction|Background|Related Work|Materials and Methods|Methods?|Methodology|Results?|Discussion|Conclusion|Conclusions)\.?\s+(.{35,})$",
        text,
        re.I,
    )
    if not match:
        return None
    heading = match.group(1)
    heading = heading[:1].upper() + heading[1:]
    return heading, match.group(2)


def useful_paragraph(text: str) -> bool:
    if len(text) < 45:
        return bool(re.search(r"\b(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion)\b", text, re.I))
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(1, len(text)) > 0.35
