from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def chats_dir(data_dir: Path) -> Path:
    path = data_dir / "chats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chat_path(data_dir: Path, document_id: str) -> Path:
    return chats_dir(data_dir) / f"{document_id}.json"


def load_chat(data_dir: Path, document_id: str) -> dict:
    path = chat_path(data_dir, document_id)
    if not path.exists():
        return {"document_id": document_id, "sessions": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"document_id": document_id, "sessions": []}


def save_chat(data_dir: Path, document_id: str, payload: dict) -> None:
    path = chat_path(data_dir, document_id)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def get_or_create_session(data_dir: Path, document_id: str, session_id: str | None = None) -> tuple[dict, dict]:
    payload = load_chat(data_dir, document_id)
    if session_id:
        for session in payload["sessions"]:
            if session.get("id") == session_id:
                return payload, session
    now = now_iso()
    session = {
        "id": session_id or f"chat-{uuid4().hex[:10]}",
        "title": "新的论文追问",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    payload["sessions"].insert(0, session)
    return payload, session


def touch_session(data_dir: Path, document_id: str, session_id: str | None = None) -> dict:
    payload, session = get_or_create_session(data_dir, document_id, session_id)
    save_chat(data_dir, document_id, payload)
    return session


def history_for_session(data_dir: Path, document_id: str, session_id: str | None) -> dict:
    payload = load_chat(data_dir, document_id)
    if not session_id:
        return payload
    sessions = payload.get("sessions", [])
    selected = [session for session in sessions if session.get("id") == session_id]
    return {"document_id": document_id, "sessions": selected}


def append_turn(
    data_dir: Path,
    document_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    session_id: str | None = None,
    llm: dict | None = None,
) -> dict:
    payload, session = get_or_create_session(data_dir, document_id, session_id)
    now = now_iso()
    if session.get("title") == "新的论文追问":
        session["title"] = question[:40]
    session["messages"].append({"role": "user", "content": question, "created_at": now})
    session["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "created_at": now,
            "sources": sources,
            "llm": llm or {},
        }
    )
    session["updated_at"] = now
    save_chat(data_dir, document_id, payload)
    return session


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
