from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .agent import PaperAgent
from .chat_store import append_turn, history_for_session, load_chat, touch_session
from .chunking import CHUNKING_VERSION, chunk_pdf_pages
from .knowledge_graph import build_knowledge_graph
from .models import PaperDocument, SearchResult
from .paper_analysis import (
    analyze_document,
    delete_analysis,
    load_analysis,
    save_analysis,
)
from .paths import DATA_DIR, ensure_data_dirs
from .pdf_loader import extract_pdf
from .retriever import HybridRetriever
from .store import DocumentStore


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        ensure_data_dirs(self.data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.store = DocumentStore(self.data_dir)
        self.rebuild_outdated_documents()
        self.refresh()

    def refresh(self) -> None:
        self.retriever = HybridRetriever(self.store.list_chunks())
        self.agent = PaperAgent(self.store, self.retriever)

    def ingest_pdf_bytes(self, filename: str, content: bytes) -> dict:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("只支持 PDF 文件。")
        sha256 = hashlib.sha256(content).hexdigest()
        existing = self.store.find_by_sha256(sha256)
        if existing:
            return {"status": "duplicate", "document": existing}

        document_id = f"paper-{sha256[:12]}"
        safe_name = safe_filename(filename)
        stored_path = self.upload_dir / f"{document_id}-{safe_name}"
        stored_path.write_bytes(content)

        extract = extract_pdf(stored_path)
        title = clean_title(extract.title) or Path(filename).stem
        chunks = chunk_pdf_pages(extract.pages, document_id=document_id, paper_title=title)
        metadata = dict(extract.metadata)
        metadata["chunking_version"] = CHUNKING_VERSION
        document = PaperDocument(
            id=document_id,
            title=title,
            filename=filename,
            stored_path=str(stored_path.resolve()),
            sha256=sha256,
            pages=len(extract.pages),
            chunks=len(chunks),
            created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            metadata=metadata,
        )
        self.store.add_document(document, chunks)
        self.refresh()
        return {"status": "created", "document": document}

    def delete_document(self, document_id: str) -> bool:
        removed = self.store.remove_document(document_id)
        if removed:
            delete_analysis(self.data_dir, document_id)
        self.refresh()
        return removed

    def rebuild_document(self, document_id: str) -> PaperDocument | None:
        document = self.store.get_document(document_id)
        if not document:
            return None
        pdf_path = Path(document.stored_path)
        if not pdf_path.exists():
            return document

        extract = extract_pdf(pdf_path)
        title = document.title or clean_title(extract.title) or Path(document.filename).stem
        chunks = chunk_pdf_pages(extract.pages, document_id=document.id, paper_title=title)
        document.title = title
        document.pages = len(extract.pages)
        document.chunks = len(chunks)
        metadata = dict(document.metadata or {})
        metadata.update(extract.metadata)
        metadata["chunking_version"] = CHUNKING_VERSION
        metadata["rebuilt_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        document.metadata = metadata
        self.store.update_document(document)
        self.store.replace_document_chunks(document_id, chunks)
        self.refresh()
        return document

    def rebuild_all(self) -> list[PaperDocument]:
        rebuilt: list[PaperDocument] = []
        for document in self.store.list_documents():
            updated = self.rebuild_document(document.id)
            if updated:
                rebuilt.append(updated)
        return rebuilt

    def rebuild_outdated_documents(self) -> None:
        changed = False
        for document in self.store.list_documents():
            if (document.metadata or {}).get("chunking_version") == CHUNKING_VERSION:
                continue
            pdf_path = Path(document.stored_path)
            if not pdf_path.exists():
                continue
            extract = extract_pdf(pdf_path)
            chunks = chunk_pdf_pages(extract.pages, document_id=document.id, paper_title=document.title)
            document.pages = len(extract.pages)
            document.chunks = len(chunks)
            metadata = dict(document.metadata or {})
            metadata["chunking_version"] = CHUNKING_VERSION
            metadata["rebuilt_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
            document.metadata = metadata
            self.store.update_document(document)
            self.store.replace_document_chunks(document.id, chunks)
            changed = True
        if changed:
            self.store.load()

    def document_pdf_path(self, document_id: str) -> Path | None:
        document = self.store.get_document(document_id)
        if not document or not document.stored_path:
            return None
        path = Path(document.stored_path)
        try:
            resolved = path.resolve()
            if resolved.exists() and resolved.is_relative_to(self.data_dir.resolve()):
                return resolved
        except OSError:
            return None
        return None

    def search(
        self,
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        return self.retriever.search(
            query=query,
            top_k=top_k,
            mode=mode,
            document_ids=document_ids,
        )

    def list_documents(self) -> list[PaperDocument]:
        return self.store.list_documents()

    def graph(self) -> dict:
        return build_knowledge_graph(
            documents=self.store.list_documents(),
            chunks=self.store.list_chunks(),
        )

    def get_analysis(self, document_id: str) -> dict | None:
        return load_analysis(self.data_dir, document_id)

    def ensure_analysis(self, document_id: str, force: bool = False) -> dict | None:
        document = self.store.get_document(document_id)
        if not document:
            return None
        cached = None if force else self.get_analysis(document_id)
        if cached:
            return cached
        analysis = analyze_document(
            document=document,
            chunks=self.store.chunks_for_document(document_id),
            force_llm=True,
        )
        save_analysis(self.data_dir, document_id, analysis)
        return analysis

    def list_analyses(self, generate_missing: bool = False) -> list[dict]:
        analyses: list[dict] = []
        for document in self.store.list_documents():
            analysis = self.get_analysis(document.id)
            if not analysis and generate_missing:
                analysis = self.ensure_analysis(document.id)
            if analysis:
                analyses.append(analysis)
        return analyses

    def chat_history(self, document_id: str) -> dict:
        return load_chat(self.data_dir, document_id)

    def ask_document(
        self,
        document_id: str,
        question: str,
        session_id: str | None = None,
        top_k: int = 6,
    ) -> dict | None:
        document = self.store.get_document(document_id)
        if not document:
            return None
        response = self.agent.ask_document(
            document=document,
            question=question,
            analysis=self.ensure_analysis(document_id, force=False),
            history=history_for_session(self.data_dir, document_id, session_id),
            top_k=top_k,
        )
        session = append_turn(
            self.data_dir,
            document_id=document_id,
            question=question,
            answer=response["answer"],
            sources=response["sources"],
            session_id=session_id,
            llm=response.get("llm"),
        )
        response["session"] = session
        return response

    def ask_document_stream(
        self,
        document_id: str,
        question: str,
        session_id: str | None = None,
        top_k: int = 6,
    ):
        """Streaming version of ask_document."""
        document = self.store.get_document(document_id)
        if not document:
            return
        
        analysis = self.ensure_analysis(document_id, force=False)
        session = touch_session(self.data_dir, document_id, session_id)
        history = history_for_session(self.data_dir, document_id, session["id"])
        
        full_answer = ""
        sources = []
        llm_meta = {}
        saved_session = None
        saved = False
        yield {"session": {"id": session["id"], "title": session.get("title", "新的论文追问")}}
        try:
            for chunk in self.agent.ask_document_stream(
                document=document,
                question=question,
                analysis=analysis,
                history=history,
                top_k=top_k,
            ):
                if "sources" in chunk:
                    sources = chunk["sources"]
                if "llm" in chunk:
                    llm_meta = chunk["llm"] or {}
                if "content" in chunk:
                    full_answer += chunk["content"]
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                
                yield chunk
        finally:
            if full_answer and not saved:
                saved_session = append_turn(
                    self.data_dir,
                    document_id=document_id,
                    question=question,
                    answer=full_answer,
                    sources=sources,
                    session_id=session["id"],
                    llm=llm_meta,
                )
                saved = True

        if saved_session:
            yield {"session": saved_session}


def safe_filename(filename: str) -> str:
    filename = filename.strip().replace("\\", "_").replace("/", "_")
    filename = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", filename)
    return filename or "paper.pdf"


def clean_title(title: str | None) -> str | None:
    if not title:
        return None
    title = re.sub(r"\s+", " ", title).strip()
    if not title or title.lower() in {"untitled", "none"}:
        return None
    return title[:240]
