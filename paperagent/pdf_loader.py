from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .text_utils import normalize_text


@dataclass
class PdfPage:
    page_number: int
    text: str


@dataclass
class PdfExtract:
    title: str | None
    pages: list[PdfPage]
    metadata: dict[str, Any]


def extract_pdf(path: Path) -> PdfExtract:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pypdf，无法解析 PDF。请运行: python -m pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError("该 PDF 已加密，无法用空密码打开。") from exc

    metadata_raw = reader.metadata or {}
    metadata = {str(key).lstrip("/"): str(value) for key, value in metadata_raw.items()}
    title = metadata.get("Title")

    pages: list[PdfPage] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PdfPage(page_number=page_index, text=normalize_text(text)))

    if not any(page.text for page in pages):
        raise RuntimeError("没有从 PDF 中抽取到文本；可能是扫描版 PDF，需要 OCR 后再导入。")

    return PdfExtract(title=title, pages=pages, metadata=metadata)

