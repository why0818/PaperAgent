from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import chat_completion, extract_json_object, llm_config, llm_last_error
from .models import Chunk, PaperDocument
from .summarizer import summarize_chunks
from .text_utils import extract_terms


MODULE_KEYS = [
    "motivation",
    "method",
    "result",
    "conclusion",
    "limitations",
    "novelty",
]


def analysis_path(data_dir: Path, document_id: str) -> Path:
    path = data_dir / "analyses"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{document_id}.json"


def load_analysis(data_dir: Path, document_id: str) -> dict | None:
    path = analysis_path(data_dir, document_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_analysis(data_dir: Path, document_id: str, analysis: dict) -> None:
    path = analysis_path(data_dir, document_id)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def delete_analysis(data_dir: Path, document_id: str) -> None:
    path = analysis_path(data_dir, document_id)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def analyze_document(
    document: PaperDocument,
    chunks: list[Chunk],
    force_llm: bool = True,
) -> dict:
    fallback = fallback_analysis(document, chunks)
    config = llm_config()
    if not force_llm or not config["enabled"] or not config.get("paper_content_allowed"):
        return fallback

    prompt = build_analysis_prompt(document, chunks, max_chars=12000, per_chunk_chars=1100)
    raw = chat_completion(
        prompt,
        temperature=0.15,
        max_tokens=4096,
        response_format={"type": "json_object"},
        timeout=180,
        role="analyze",
    )
    first_error = llm_last_error()
    parsed = extract_json_object(raw or "")
    if not parsed:
        compact_prompt = build_analysis_prompt(
            document,
            representative_chunks(chunks, limit=14),
            max_chars=8500,
            per_chunk_chars=900,
        )
        raw = chat_completion(
            compact_prompt,
            temperature=0.1,
            max_tokens=4096,
            response_format=None,
            timeout=180,
            role="analyze",
        )
        second_error = llm_last_error()
        parsed = extract_json_object(raw or "")
    if not parsed:
        fallback["llm_error"] = second_error if "second_error" in locals() and second_error else first_error
        fallback["llm_raw_preview"] = (raw or "")[:1200]
        return fallback
    return normalize_analysis(document, chunks, parsed, raw)


def fallback_analysis(document: PaperDocument, chunks: list[Chunk]) -> dict:
    terms = extract_terms(" ".join(chunk.text[:700] for chunk in chunks), limit=8)
    sections = group_chunks_by_module(chunks)
    modules = {}
    for key in MODULE_KEYS:
        selected = sections.get(key) or chunks[: min(4, len(chunks))]
        modules[key] = {
            "title": module_title(key),
            "summary": summarize_chunks(selected, max_sentences=3),
            "english_evidence": [
                source_line(chunk, limit=260)
                for chunk in selected[:2]
            ],
            "chinese_explanation": summarize_chunks(selected, max_sentences=2),
            "bullets": [
                source_line(chunk)
                for chunk in selected[:3]
            ],
            "citations": citations(selected[:3]),
        }

    return {
        "document_id": document.id,
        "title_cn": document.title,
        "title_en": document.title,
        "evidence": first_meaningful_sentence(chunks),
        "tldr": summarize_chunks(chunks[: min(8, len(chunks))], max_sentences=4),
        "authors": parse_author(document),
        "source": parse_source(document),
        "venue": parse_source(document),
        "venue_type": parse_venue_type(parse_source(document)),
        "date": parse_date(document),
        "tags": terms[:5],
        "score": 7.0,
        "modules": modules,
        "reading_plan": [
            "先读 TLDR 和 Evidence，把握论文主张。",
            "再按 Motivation、Method、Result、Conclusion 四个模块快速扫读。",
            "需要核验时点击证据来源返回 PDF 页。",
        ],
        "questions": [
            "这篇论文的核心贡献是什么？",
            "方法相比已有工作有什么不同？",
            "结果是否足以支持结论？",
        ],
        "status": "fallback",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model": "local-extractive",
    }


def build_analysis_prompt(
    document: PaperDocument,
    chunks: list[Chunk],
    max_chars: int = 12000,
    per_chunk_chars: int = 1100,
) -> list[dict[str, str]]:
    evidence = build_evidence_digest(
        chunks,
        max_chars=max_chars,
        per_chunk_chars=per_chunk_chars,
    )
    schema = {
        "title_cn": "中文标题",
        "title_en": "English title",
        "evidence": "一句话说明为什么这篇论文值得读，中文，不超过80字",
        "tldr": "中文 TLDR，120-220字，必须提炼而非照抄",
        "authors": ["作者1", "作者2"],
        "source": "期刊或会议名称，例如 Science / Nature / CVPR / NeurIPS；未知则填未知",
        "venue_type": "journal 或 conference 或 unknown",
        "date": "YYYY-MM-DD 或未知",
        "tags": ["tag1", "tag2"],
        "score": 8.0,
        "modules": {
            key: {
                "title": module_title(key),
                "summary": "该模块中文总结，80-160字",
                "english_evidence": ["从论文原文中提炼出的英文证据句，不超过2条，每条保留原文意思"],
                "chinese_explanation": "对英文证据的中文解释，不是逐字翻译，要说明它在论文中的作用",
                "bullets": ["要点1", "要点2", "要点3"],
                "citations": [{"chunk_id": "paper-xxx-c00001", "page": "1-2"}],
            }
            for key in MODULE_KEYS
        },
        "reading_plan": ["阅读建议1", "阅读建议2"],
        "questions": ["可继续追问的问题1", "可继续追问的问题2"],
    }
    system = (
        "你是一个前沿论文精读 Agent，风格参考 Daily Paper Reader。"
        "你的任务是把论文组织为模块化阅读卡片，而不是翻译或照搬原文。"
        "必须基于给定证据，输出严格 JSON，不要 Markdown。"
        "每个模块都要概括研究含义，并给出可回溯 citation。"
        "如果论文是英文，每个模块必须提供 english_evidence 与 chinese_explanation，形成中英对照阅读。"
        "english_evidence 要短而准，chinese_explanation 要解释意义，不要机械翻译。"
    )
    user = (
        f"论文标题：{document.title}\n"
        f"文件名：{document.filename}\n"
        f"元数据：{json.dumps(document.metadata, ensure_ascii=False)[:1200]}\n\n"
        f"请按以下 JSON schema 输出：\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"论文证据块：\n{evidence}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_evidence_digest(
    chunks: list[Chunk],
    max_chars: int = 12000,
    per_chunk_chars: int = 1100,
) -> str:
    priority = sorted(
        chunks,
        key=lambda chunk: (
            module_priority(chunk.section),
            chunk.page_start,
            chunk.chunk_index,
        ),
    )
    selected: list[str] = []
    used_chars = 0
    for chunk in priority:
        text = " ".join(chunk.text.split())
        item = (
            f"[{chunk.id}] section={chunk.section or 'Unknown'} "
            f"pages={chunk.page_start}-{chunk.page_end}\n{text[:per_chunk_chars]}"
        )
        if used_chars + len(item) > max_chars:
            continue
        selected.append(item)
        used_chars += len(item)
        if used_chars >= max_chars * 0.92:
            break
    return "\n\n".join(selected)


def representative_chunks(chunks: list[Chunk], limit: int = 14) -> list[Chunk]:
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            module_priority(chunk.section),
            chunk.page_start,
            chunk.chunk_index,
        ),
    )
    selected: list[Chunk] = []
    seen_sections: set[str] = set()
    for chunk in ranked:
        section = (chunk.section or "Unknown").lower()
        if section not in seen_sections or len(selected) < 6:
            selected.append(chunk)
            seen_sections.add(section)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for chunk in ranked:
            if chunk not in selected:
                selected.append(chunk)
            if len(selected) >= limit:
                break
    return selected


def normalize_analysis(
    document: PaperDocument,
    chunks: list[Chunk],
    parsed: dict[str, Any],
    raw: str | None,
) -> dict:
    fallback = fallback_analysis(document, chunks)
    analysis = {**fallback, **{k: v for k, v in parsed.items() if v}}
    modules = analysis.get("modules") if isinstance(analysis.get("modules"), dict) else {}
    normalized_modules = {}
    for key in MODULE_KEYS:
        value = modules.get(key) if isinstance(modules, dict) else None
        if not isinstance(value, dict):
            value = fallback["modules"][key]
        value.setdefault("title", module_title(key))
        value.setdefault("summary", fallback["modules"][key]["summary"])
        value.setdefault("english_evidence", fallback["modules"][key]["english_evidence"])
        value.setdefault("chinese_explanation", fallback["modules"][key]["chinese_explanation"])
        value.setdefault("bullets", fallback["modules"][key]["bullets"])
        value.setdefault("citations", fallback["modules"][key]["citations"])
        normalized_modules[key] = value
    analysis["modules"] = normalized_modules
    analysis["document_id"] = document.id
    source = str(analysis.get("source") or analysis.get("venue") or fallback.get("source") or "").strip()
    analysis["source"] = source or "未知"
    analysis["venue"] = analysis["source"]
    analysis["venue_type"] = str(analysis.get("venue_type") or parse_venue_type(analysis["source"]) or "unknown")
    analysis["status"] = "llm"
    analysis["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    analysis["model"] = llm_config().get("analyze_model") or "unknown"
    if raw:
        analysis["_raw_preview"] = raw[:600]
    return analysis


def group_chunks_by_module(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    groups = {key: [] for key in MODULE_KEYS}
    for chunk in chunks:
        section = (chunk.section or "").lower()
        if any(word in section for word in ["abstract", "introduction", "background"]):
            groups["motivation"].append(chunk)
        elif any(word in section for word in ["method", "material", "approach"]):
            groups["method"].append(chunk)
        elif "result" in section or "experiment" in section:
            groups["result"].append(chunk)
        elif "discussion" in section or "conclusion" in section:
            groups["conclusion"].append(chunk)
            groups["limitations"].append(chunk)
        else:
            groups["novelty"].append(chunk)
    return groups


def module_priority(section: str) -> int:
    lowered = (section or "").lower()
    order = [
        "abstract",
        "introduction",
        "method",
        "result",
        "discussion",
        "conclusion",
    ]
    for index, name in enumerate(order):
        if name in lowered:
            return index
    return 10


def module_title(key: str) -> str:
    return {
        "motivation": "Motivation",
        "method": "Method",
        "result": "Result",
        "conclusion": "Conclusion",
        "limitations": "Limitations",
        "novelty": "Novelty",
    }[key]


def source_line(chunk: Chunk, limit: int = 180) -> str:
    text = " ".join(chunk.text.split())
    return f"{text[:limit]}... (p.{chunk.page_start}-{chunk.page_end})"


def citations(chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {"chunk_id": chunk.id, "page": f"{chunk.page_start}-{chunk.page_end}"}
        for chunk in chunks
    ]


def first_meaningful_sentence(chunks: list[Chunk]) -> str:
    for chunk in chunks:
        text = " ".join(chunk.text.split())
        if len(text) > 80:
            return text[:180]
    return "暂无足够文本。"


def parse_author(document: PaperDocument) -> list[str]:
    author = str((document.metadata or {}).get("Author", "")).strip()
    if not author:
        return []
    return [part.strip() for part in author.replace(" and ", ",").split(",") if part.strip()][:8]


def parse_source(document: PaperDocument) -> str:
    metadata = document.metadata or {}
    subject = str(metadata.get("Subject", "")).strip()
    if "IEEE" in subject and "Conference" in subject:
        return subject.split(";")[0].strip()
    if "," in subject:
        return subject.split(",", 1)[0].strip()
    if ":" in subject and not subject.lower().startswith(("http://", "https://")):
        return subject.split(":", 1)[0].strip()
    for key in ("Journal", "journal", "PublicationTitle", "Publisher"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    return "未知"


def parse_venue_type(source: str) -> str:
    lowered = (source or "").lower()
    if any(word in lowered for word in ["conference", "cvpr", "iccv", "eccv", "neurips", "iclr", "icml", "symposium"]):
        return "conference"
    if source and source != "未知":
        return "journal"
    return "unknown"


def parse_date(document: PaperDocument) -> str:
    for key in ("date", "Date", "CreationDate", "ModDate"):
        value = str((document.metadata or {}).get(key, "")).strip()
        if value:
            return value[:16]
    return "未知"
