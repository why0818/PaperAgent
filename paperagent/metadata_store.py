from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Chunk, PaperDocument
from .paper_analysis import fallback_analysis, load_analysis, save_analysis


def venues_path(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "venues.json"


def load_venues(data_dir: Path) -> dict:
    path = venues_path(data_dir)
    if not path.exists():
        return {"version": 1, "venues": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "venues": []}
    venues = payload.get("venues", [])
    if not isinstance(venues, list):
        venues = []
    return {"version": 1, "venues": venues}


def save_venues(data_dir: Path, payload: dict) -> None:
    path = venues_path(data_dir)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def upsert_venue(data_dir: Path, name: str, venue_type: str = "") -> None:
    name = clean_field(name)
    if not name:
        return
    venue_type = clean_field(venue_type) or infer_venue_type(name)
    payload = load_venues(data_dir)
    venues = payload["venues"]
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    for venue in venues:
        if str(venue.get("name", "")).casefold() == name.casefold():
            venue["name"] = name
            venue["type"] = venue_type or venue.get("type", "")
            venue["count"] = int(venue.get("count", 0)) + 1
            venue["updated_at"] = now
            save_venues(data_dir, payload)
            return
    venues.append({"name": name, "type": venue_type, "count": 1, "updated_at": now})
    venues.sort(key=lambda item: str(item.get("name", "")).casefold())
    save_venues(data_dir, payload)


def editable_metadata(document: PaperDocument, analysis: dict | None) -> dict:
    analysis = analysis or {}
    source = clean_field(analysis.get("source") or analysis.get("venue") or infer_source(document))
    return {
        "authors": normalize_authors(analysis.get("authors") or infer_authors(document)),
        "source": source,
        "venue_type": clean_field(analysis.get("venue_type") or infer_venue_type(source)),
        "date": clean_field(analysis.get("date") or infer_date(document)),
        "score": normalize_score(analysis.get("score", 7.0)),
        "title_cn": clean_field(analysis.get("title_cn") or document.title),
        "title_en": clean_field(analysis.get("title_en") or document.title),
    }


def update_document_metadata(
    data_dir: Path,
    document: PaperDocument,
    chunks: list[Chunk],
    payload: dict[str, Any],
) -> dict:
    analysis = load_analysis(data_dir, document.id) or fallback_analysis(document, chunks)
    authors = normalize_authors(payload.get("authors", analysis.get("authors", [])))
    source = clean_field(payload.get("source", analysis.get("source", "")))
    venue_type = clean_field(payload.get("venue_type", analysis.get("venue_type", ""))) or infer_venue_type(source)
    date = clean_field(payload.get("date", analysis.get("date", ""))) or "未知"
    score = normalize_score(payload.get("score", analysis.get("score", 7.0)))
    title_cn = clean_field(payload.get("title_cn", analysis.get("title_cn", ""))) or analysis.get("title_cn") or document.title
    title_en = clean_field(payload.get("title_en", analysis.get("title_en", ""))) or analysis.get("title_en") or document.title

    analysis.update(
        {
            "authors": authors,
            "source": source,
            "venue": source,
            "venue_type": venue_type,
            "date": date,
            "score": score,
            "title_cn": title_cn,
            "title_en": title_en,
            "metadata_updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
    )
    save_analysis(data_dir, document.id, analysis)
    upsert_venue(data_dir, source, venue_type)
    return analysis


def infer_source(document: PaperDocument) -> str:
    metadata = document.metadata or {}
    subject = clean_field(metadata.get("Subject", ""))
    if "IEEE" in subject and "Conference" in subject:
        return subject.split(";")[0].strip()
    if "," in subject:
        return subject.split(",", 1)[0].strip()
    if ":" in subject and not subject.lower().startswith(("http://", "https://")):
        return subject.split(":", 1)[0].strip()
    for key in ("Journal", "journal", "PublicationTitle", "Publisher"):
        value = clean_field(metadata.get(key, ""))
        if value:
            return value
    return ""


def infer_venue_type(name: str) -> str:
    lowered = (name or "").lower()
    if any(word in lowered for word in ["conference", "cvpr", "iccv", "eccv", "neurips", "iclr", "icml", "symposium"]):
        return "conference"
    if name:
        return "journal"
    return ""


def infer_authors(document: PaperDocument) -> list[str]:
    author = clean_field((document.metadata or {}).get("Author", ""))
    if not author:
        return []
    return normalize_authors(author)


def infer_date(document: PaperDocument) -> str:
    metadata = document.metadata or {}
    for key in ("date", "Date", "CreationDate", "ModDate", "Meeting Starting Date"):
        value = clean_field(metadata.get(key, ""))
        if value:
            match = re.search(r"(20\d{2}|19\d{2})[-/:]?(\d{2})?[-/:]?(\d{2})?", value)
            if match:
                year, month, day = match.groups()
                if month and day:
                    return f"{year}-{month}-{day}"
                return year
            return value[:32]
    return "未知"


def normalize_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value or "").replace(" and ", ",")
        parts = [part.strip() for part in re.split(r"[,;\n]+", text)]
    return [part for part in parts if part][:20]


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 7.0
    return round(max(0.0, min(10.0, score)), 1)


def clean_field(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()
