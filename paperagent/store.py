from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Chunk, PaperDocument
from .paths import DATA_DIR, ensure_data_dirs


class DocumentStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        ensure_data_dirs(self.data_dir)
        self.library_path = self.data_dir / "library.json"
        self._documents: dict[str, PaperDocument] = {}
        self._chunks: dict[str, Chunk] = {}
        self.load()

    def load(self) -> None:
        if not self.library_path.exists():
            self._documents = {}
            self._chunks = {}
            return
        with self.library_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self._documents = {
            item["id"]: PaperDocument.from_dict(item)
            for item in payload.get("documents", [])
        }
        self._chunks = {
            item["id"]: Chunk.from_dict(item)
            for item in payload.get("chunks", [])
        }

    def save(self) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "documents": [doc.to_dict() for doc in self.list_documents()],
            "chunks": [chunk.to_dict() for chunk in self.list_chunks()],
        }
        temp_path = self.library_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.library_path)

    def add_document(self, document: PaperDocument, chunks: list[Chunk]) -> None:
        self._documents[document.id] = document
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self.save()

    def update_document(self, document: PaperDocument) -> None:
        self._documents[document.id] = document
        self.save()

    def replace_document_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id != document_id
        }
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        document = self._documents.get(document_id)
        if document:
            document.chunks = len(chunks)
        self.save()

    def remove_document(self, document_id: str, remove_file: bool = True) -> bool:
        document = self._documents.pop(document_id, None)
        if not document:
            return False

        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id != document_id
        }
        if remove_file and document.stored_path:
            path = Path(document.stored_path)
            try:
                resolved = path.resolve()
                if resolved.is_relative_to(self.data_dir.resolve()) and resolved.exists():
                    resolved.unlink()
            except OSError:
                pass
        self.save()
        return True

    def find_by_sha256(self, sha256: str) -> PaperDocument | None:
        for document in self._documents.values():
            if document.sha256 == sha256:
                return document
        return None

    def get_document(self, document_id: str) -> PaperDocument | None:
        return self._documents.get(document_id)

    def list_documents(self) -> list[PaperDocument]:
        return sorted(self._documents.values(), key=lambda doc: doc.created_at, reverse=True)

    def list_chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda chunk: chunk.id)

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        return [chunk for chunk in self.list_chunks() if chunk.document_id == document_id]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def neighbor_chunks(self, chunk_id: str, radius: int = 1) -> list[Chunk]:
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return []
        chunks = self.chunks_for_document(chunk.document_id)
        index = next((idx for idx, item in enumerate(chunks) if item.id == chunk_id), -1)
        if index < 0:
            return [chunk]
        start = max(0, index - radius)
        end = min(len(chunks), index + radius + 1)
        return chunks[start:end]
