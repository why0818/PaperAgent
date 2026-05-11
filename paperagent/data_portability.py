from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


PACKAGE_VERSION = 1


def export_data_package(data_dir: Path) -> bytes:
    data_dir = Path(data_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": PACKAGE_VERSION,
                    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    "excluded": ["config.local.json", "*.tmp"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        add_file(zf, data_dir / "library.json", "library.json")
        add_file(zf, data_dir / "venues.json", "venues.json")
        add_tree(zf, data_dir / "uploads", "uploads")
        add_tree(zf, data_dir / "analyses", "analyses", suffix=".json")
        add_tree(zf, data_dir / "chats", "chats", suffix=".json")
    return buffer.getvalue()


def import_data_package(data_dir: Path, content: bytes) -> dict[str, Any]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(exist_ok=True)
    (data_dir / "analyses").mkdir(exist_ok=True)
    (data_dir / "chats").mkdir(exist_ok=True)

    imported_docs = 0
    imported_chunks = 0
    imported_uploads = 0
    imported_analyses = 0
    imported_chats = 0

    with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
        names = [name for name in zf.namelist() if safe_archive_name(name)]
        library_payload = read_json_member(zf, "library.json") or {"version": 1, "documents": [], "chunks": []}

        uploads = {name for name in names if name.startswith("uploads/") and not name.endswith("/")}
        for name in sorted(uploads):
            target = data_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            imported_uploads += 1

        for folder, counter_name in (("analyses", "analyses"), ("chats", "chats")):
            for name in sorted(item for item in names if item.startswith(f"{folder}/") and item.endswith(".json")):
                target = data_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
                if counter_name == "analyses":
                    imported_analyses += 1
                else:
                    imported_chats += 1

        existing_library = load_local_library(data_dir / "library.json")
        documents_by_id = {str(doc.get("id")): doc for doc in existing_library.get("documents", []) if doc.get("id")}
        chunks_by_id = {str(chunk.get("id")): chunk for chunk in existing_library.get("chunks", []) if chunk.get("id")}

        for doc in library_payload.get("documents", []):
            if not isinstance(doc, dict) or not doc.get("id"):
                continue
            doc = dict(doc)
            doc["stored_path"] = local_stored_path(data_dir, doc, uploads)
            documents_by_id[str(doc["id"])] = doc
            imported_docs += 1

        for chunk in library_payload.get("chunks", []):
            if not isinstance(chunk, dict) or not chunk.get("id"):
                continue
            chunks_by_id[str(chunk["id"])] = dict(chunk)
            imported_chunks += 1

        save_json(
            data_dir / "library.json",
            {
                "version": 1,
                "documents": list(documents_by_id.values()),
                "chunks": list(chunks_by_id.values()),
            },
        )

        imported_venues = merge_venues(data_dir, read_json_member(zf, "venues.json"))

    return {
        "documents": imported_docs,
        "chunks": imported_chunks,
        "uploads": imported_uploads,
        "analyses": imported_analyses,
        "chats": imported_chats,
        "venues": imported_venues,
    }


def add_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.exists() and path.is_file():
        zf.write(path, arcname)


def add_tree(zf: zipfile.ZipFile, path: Path, arc_prefix: str, suffix: str | None = None) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if suffix and item.suffix.lower() != suffix:
            continue
        if item.suffix.lower() == ".tmp":
            continue
        zf.write(item, PurePosixPath(arc_prefix, item.relative_to(path).as_posix()).as_posix())


def safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute():
        return False
    if any(part in {"", ".", ".."} for part in path.parts):
        return False
    return True


def read_json_member(zf: zipfile.ZipFile, name: str) -> dict | None:
    if name not in zf.namelist():
        return None
    try:
        return json.loads(zf.read(name).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def load_local_library(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "documents": [], "chunks": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "documents": [], "chunks": []}
    payload.setdefault("documents", [])
    payload.setdefault("chunks", [])
    return payload


def local_stored_path(data_dir: Path, doc: dict, uploads: set[str]) -> str:
    original = str(doc.get("stored_path", "")).replace("\\", "/")
    basename = Path(original).name
    document_id = str(doc.get("id", ""))
    if not basename or f"uploads/{basename}" not in uploads:
        match = next((name for name in uploads if Path(name).name.startswith(f"{document_id}-")), "")
        basename = Path(match).name if match else basename
    if not basename:
        filename = str(doc.get("filename", "")).strip() or f"{document_id}.pdf"
        basename = f"{document_id}-{filename}"
    return str((data_dir / "uploads" / basename).resolve())


def merge_venues(data_dir: Path, incoming: dict | None) -> int:
    if not incoming:
        return 0
    target = data_dir / "venues.json"
    existing = read_json_file(target) or {"version": 1, "venues": []}
    venues_by_name = {
        str(item.get("name", "")).casefold(): dict(item)
        for item in existing.get("venues", [])
        if str(item.get("name", "")).strip()
    }
    imported = 0
    for item in incoming.get("venues", []):
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        key = str(item["name"]).casefold()
        current = venues_by_name.get(key, {})
        merged = {**current, **item}
        merged["count"] = max(int(current.get("count", 0) or 0), int(item.get("count", 0) or 0))
        venues_by_name[key] = merged
        imported += 1
    save_json(target, {"version": 1, "venues": sorted(venues_by_name.values(), key=lambda value: str(value.get("name", "")).casefold())})
    return imported


def read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)
