from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message="'cgi' is deprecated.*",
    category=DeprecationWarning,
)

import cgi
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from paperagent.knowledge_base import KnowledgeBase
from paperagent.llm import llm_config, save_llm_models
from paperagent.metadata_store import editable_metadata, load_venues, update_document_metadata, upsert_venue
from paperagent.paths import DATA_DIR
from paperagent.preview import build_preview_chunks


KB = KnowledgeBase(DATA_DIR)
KB_LOCK = threading.RLock()


def document_payload(doc) -> dict:
    analysis = KB.get_analysis(doc.id)
    meta = editable_metadata(doc, analysis)
    return {
        "id": doc.id,
        "title": doc.title,
        "filename": doc.filename,
        "pages": doc.pages,
        "chunks": doc.chunks,
        "created_at": doc.created_at,
        "sha256": doc.sha256,
        "analysis_status": analysis.get("status") if analysis else "missing",
        "score": meta.get("score"),
        "tags": analysis.get("tags", []) if analysis else [],
        "title_cn": meta.get("title_cn") or "",
        "authors": meta.get("authors", []),
        "source": meta.get("source", ""),
        "venue_type": meta.get("venue_type", ""),
        "date": meta.get("date", ""),
    }


def chunk_payload(chunk, preview_chars: int = 700) -> dict:
    text = " ".join(chunk.text.split())
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "paper_title": chunk.paper_title,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "keywords": chunk.keywords,
        "snippet": text[:preview_chars] + ("..." if len(text) > preview_chars else ""),
        "text": chunk.text,
    }


def venues_payload() -> dict:
    payload = load_venues(DATA_DIR)
    venues_by_name = {
        str(item.get("name", "")).casefold(): dict(item)
        for item in payload.get("venues", [])
        if str(item.get("name", "")).strip()
    }
    for document in KB.list_documents():
        analysis = KB.get_analysis(document.id)
        meta = editable_metadata(document, analysis)
        name = str(meta.get("source", "")).strip()
        if not name or name == "未知":
            continue
        key = name.casefold()
        if key not in venues_by_name:
            venues_by_name[key] = {
                "name": name,
                "type": meta.get("venue_type", "") or "unknown",
                "count": 0,
                "updated_at": document.created_at,
            }
        venues_by_name[key]["count"] = int(venues_by_name[key].get("count", 0)) + 1
    return {"version": 1, "venues": sorted(venues_by_name.values(), key=lambda item: str(item.get("name", "")).casefold())}


def source_payload(result, query: str) -> dict:
    payload = result.to_source_dict(query=query)
    payload["pdf_url"] = f"/api/documents/{payload['document_id']}/pdf#page={payload['page_start']}"
    return payload


def clamp_int(value: str | None, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


class PaperAgentHandler(BaseHTTPRequestHandler):
    server_version = "PaperAgentWeb/0.2"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/api/documents":
            self.handle_documents()
            return
        if parsed.path == "/api/config":
            self.send_json({"llm": llm_config()})
            return
        if parsed.path == "/api/venues":
            with KB_LOCK:
                self.send_json(venues_payload())
            return
        if parsed.path == "/api/graph":
            with KB_LOCK:
                graph = KB.graph()
            self.send_json(graph)
            return
        if parsed.path == "/api/analyses":
            with KB_LOCK:
                analyses = KB.list_analyses(generate_missing=False)
            self.send_json({"analyses": analyses})
            return
        chat_match = re.fullmatch(r"/api/documents/([^/]+)/chats", parsed.path)
        if chat_match:
            document_id = unquote(chat_match.group(1))
            with KB_LOCK:
                document = KB.store.get_document(document_id)
                if not document:
                    self.send_error_json(404, "没有找到该文档。")
                    return
                history = KB.chat_history(document_id)
            self.send_json(history)
            return
        document_match = re.fullmatch(r"/api/documents/([^/]+)", parsed.path)
        if document_match:
            self.handle_document_detail(unquote(document_match.group(1)))
            return
        pdf_match = re.fullmatch(r"/api/documents/([^/]+)/pdf", parsed.path)
        if pdf_match:
            self.handle_document_pdf(unquote(pdf_match.group(1)))
            return
        if parsed.path == "/api/search":
            self.handle_search(parsed.query)
            return
        if parsed.path == "/api/summary":
            self.handle_summary(parsed.query)
            return
        self.send_error_json(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self.handle_upload()
            return
        if parsed.path == "/api/ask":
            self.handle_ask()
            return
        if parsed.path == "/api/ask/stream":
            self.handle_ask_stream()
            return
        if parsed.path == "/api/analyze":
            self.handle_analyze()
            return
        if parsed.path == "/api/config/models":
            self.handle_save_models()
            return
        metadata_match = re.fullmatch(r"/api/documents/([^/]+)/metadata", parsed.path)
        if metadata_match:
            self.handle_update_document_metadata(unquote(metadata_match.group(1)))
            return
        analysis_match = re.fullmatch(r"/api/documents/([^/]+)/analysis", parsed.path)
        if analysis_match:
            self.handle_analyze(unquote(analysis_match.group(1)))
            return
        doc_chat_stream_match = re.fullmatch(r"/api/documents/([^/]+)/chat/stream", parsed.path)
        if doc_chat_stream_match:
            self.handle_document_chat_stream(unquote(doc_chat_stream_match.group(1)))
            return
        doc_chat_match = re.fullmatch(r"/api/documents/([^/]+)/chat", parsed.path)
        if doc_chat_match:
            self.handle_document_chat(unquote(doc_chat_match.group(1)))
            return
        if parsed.path == "/api/rebuild":
            self.handle_rebuild()
            return
        rebuild_match = re.fullmatch(r"/api/documents/([^/]+)/rebuild", parsed.path)
        if rebuild_match:
            self.handle_rebuild(unquote(rebuild_match.group(1)))
            return
        self.send_error_json(404, "Not found")

    def handle_ask_stream(self) -> None:
        payload = self.read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json(400, "问题不能为空。")
            return
        top_k = clamp_int(str(payload.get("top_k", "6")), default=6, low=3, high=20)
        document_ids = payload.get("document_ids")
        if not isinstance(document_ids, list):
            document_ids = None
        else:
            document_ids = [str(item) for item in document_ids if str(item).strip()]
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        print(f"[ask_stream] Processing question: {question[:50]}...", flush=True)
        disconnected = False
        with KB_LOCK:
            for chunk in KB.agent.ask_stream(question, top_k=top_k, document_ids=document_ids):
                if "sources" in chunk:
                    for source in chunk["sources"]:
                        source["pdf_url"] = f"/api/documents/{source['document_id']}/pdf#page={source['page_start']}"
                try:
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    print("[ask_stream] client disconnected", flush=True)
                    disconnected = True
                    break
        if not disconnected:
            try:
                self.wfile.write(f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("[ask_stream] client disconnected before done", flush=True)
        self.close_connection = True
        print(f"[ask_stream] Done", flush=True)

    def handle_document_chat_stream(self, document_id: str) -> None:
        payload = self.read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json(400, "问题不能为空。")
            return
        session_id = str(payload.get("session_id", "")).strip() or None
        top_k = clamp_int(str(payload.get("top_k", "6")), default=6, low=3, high=16)
        
        print(f"[document_chat_stream] document_id={document_id}, question={question[:50]}, session_id={session_id}", flush=True)
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        stream = KB.ask_document_stream(
            document_id=document_id,
            question=question,
            session_id=session_id,
            top_k=top_k,
        )
        disconnected = False
        for chunk in stream:
            if "sources" in chunk:
                for source in chunk["sources"]:
                    source["pdf_url"] = f"/api/documents/{source['document_id']}/pdf#page={source['page_start']}"
            try:
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("[document_chat_stream] client disconnected", flush=True)
                close_stream = getattr(stream, "close", None)
                if close_stream:
                    close_stream()
                disconnected = True
                break
        if not disconnected:
            try:
                self.wfile.write(f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("[document_chat_stream] client disconnected before done", flush=True)
        self.close_connection = True
        print(f"[document_chat_stream] Done", flush=True)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/documents/"):
            document_id = unquote(parsed.path.rsplit("/", 1)[-1])
            with KB_LOCK:
                removed = KB.delete_document(document_id)
            self.send_json({"ok": removed})
            return
        self.send_error_json(404, "Not found")

    def handle_documents(self) -> None:
        with KB_LOCK:
            docs = KB.list_documents()
            chunk_count = len(KB.store.list_chunks())
        self.send_json(
            {
                "documents": [document_payload(doc) for doc in docs],
                "stats": {
                    "documents": len(docs),
                    "chunks": chunk_count,
                    "pages": sum(doc.pages for doc in docs),
                },
            }
        )

    def handle_document_detail(self, document_id: str) -> None:
        with KB_LOCK:
            document = KB.store.get_document(document_id)
            if not document:
                self.send_error_json(404, "没有找到该文档。")
                return
            chunks = KB.store.chunks_for_document(document_id)
            analysis = KB.get_analysis(document_id)
        self.send_json(
            {
                "document": document_payload(document),
                "chunks": [chunk_payload(chunk) for chunk in chunks],
                "preview_chunks": build_preview_chunks(chunks),
                "analysis": analysis,
                "editable_metadata": editable_metadata(document, analysis),
                "pdf_url": f"/api/documents/{document_id}/pdf",
            }
        )

    def handle_document_pdf(self, document_id: str) -> None:
        with KB_LOCK:
            document = KB.store.get_document(document_id)
            path = KB.document_pdf_path(document_id)
        if not document or not path:
            self.send_error_json(404, "没有找到原始 PDF 文件。")
            return
        self.send_pdf(path, document.filename)

    def handle_search(self, query_string: str) -> None:
        params = parse_qs(query_string)
        query = (params.get("q") or [""])[0].strip()
        if not query:
            self.send_error_json(400, "检索内容不能为空。")
            return
        mode = (params.get("mode") or ["hybrid"])[0]
        top_k = clamp_int((params.get("top_k") or [None])[0], default=8, low=1, high=30)
        document_ids = selected_document_ids(params)
        with KB_LOCK:
            results = KB.search(query, top_k=top_k, mode=mode, document_ids=document_ids)
        self.send_json({"results": [source_payload(result, query) for result in results]})

    def handle_summary(self, query_string: str) -> None:
        params = parse_qs(query_string)
        document_id = (params.get("document_id") or ["__all__"])[0]
        focus = (params.get("focus") or [""])[0].strip() or None
        max_sentences = clamp_int(
            (params.get("max_sentences") or [None])[0],
            default=8,
            low=3,
            high=18,
        )
        with KB_LOCK:
            if document_id == "__all__":
                summary = KB.agent.summarize_library(focus, max_sentences=max_sentences)
            else:
                summary = KB.agent.summarize_document(
                    document_id,
                    query=focus,
                    max_sentences=max_sentences,
                )
        self.send_json({"summary": summary})

    def handle_ask(self) -> None:
        payload = self.read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json(400, "问题不能为空。")
            return
        top_k = clamp_int(str(payload.get("top_k", "6")), default=6, low=3, high=20)
        with KB_LOCK:
            response = KB.agent.ask(question, top_k=top_k)
            for source in response.get("sources", []):
                source["pdf_url"] = f"/api/documents/{source['document_id']}/pdf#page={source['page_start']}"
        self.send_json(response)

    def handle_rebuild(self, document_id: str | None = None) -> None:
        with KB_LOCK:
            if document_id:
                document = KB.rebuild_document(document_id)
                if not document:
                    self.send_error_json(404, "没有找到该文档。")
                    return
                payload = {"documents": [document_payload(document)]}
            else:
                documents = KB.rebuild_all()
                payload = {"documents": [document_payload(document) for document in documents]}
        self.send_json(payload)

    def handle_analyze(self, document_id: str | None = None) -> None:
        payload = self.read_json()
        force = bool(payload.get("force", True)) if payload else True
        with KB_LOCK:
            if document_id:
                analysis = KB.ensure_analysis(document_id, force=force)
                if not analysis:
                    self.send_error_json(404, "没有找到该文档。")
                    return
                self.send_json({"analysis": analysis})
                return
            analyses = []
            for document in KB.list_documents():
                analysis = KB.ensure_analysis(document.id, force=force)
                if analysis:
                    analyses.append(analysis)
            self.send_json({"analyses": analyses})

    def handle_save_models(self) -> None:
        payload = self.read_json()
        chat_model = str(payload.get("chat_model", "")).strip()
        analyze_model = str(payload.get("analyze_model", "")).strip()
        
        if not chat_model or not analyze_model:
            self.send_error_json(400, "chat_model 和 analyze_model 都不能为空。")
            return
            
        save_llm_models(chat_model, analyze_model)
        self.send_json({"success": True, "llm": llm_config()})

    def handle_update_document_metadata(self, document_id: str) -> None:
        payload = self.read_json()
        with KB_LOCK:
            document = KB.store.get_document(document_id)
            if not document:
                self.send_error_json(404, "没有找到该文档。")
                return
            chunks = KB.store.chunks_for_document(document_id)
            analysis = update_document_metadata(DATA_DIR, document, chunks, payload)
            if analysis.get("source"):
                upsert_venue(DATA_DIR, analysis["source"], analysis.get("venue_type", ""))
            venues = venues_payload().get("venues", [])
        self.send_json(
            {
                "success": True,
                "analysis": analysis,
                "document": document_payload(document),
                "editable_metadata": editable_metadata(document, analysis),
                "venues": venues,
            }
        )

    def handle_document_chat(self, document_id: str) -> None:
        payload = self.read_json()
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_error_json(400, "问题不能为空。")
            return
        session_id = str(payload.get("session_id", "")).strip() or None
        top_k = clamp_int(str(payload.get("top_k", "6")), default=6, low=3, high=16)
        with KB_LOCK:
            response = KB.ask_document(
                document_id=document_id,
                question=question,
                session_id=session_id,
                top_k=top_k,
            )
            if not response:
                self.send_error_json(404, "没有找到该文档。")
                return
            for source in response.get("sources", []):
                source["pdf_url"] = f"/api/documents/{source['document_id']}/pdf#page={source['page_start']}"
        self.send_json(response)

    def handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error_json(400, "请使用 multipart/form-data 上传 PDF。")
            return

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )
        uploads = []
        for field_name in ("files", "file"):
            if field_name not in form:
                continue
            field = form[field_name]
            uploads.extend(field if isinstance(field, list) else [field])

        if not uploads:
            self.send_error_json(400, "没有收到 PDF 文件。")
            return

        results = []
        with KB_LOCK:
            for item in uploads:
                filename = Path(item.filename or "").name
                if not filename:
                    continue
                try:
                    content = item.file.read()
                    result = KB.ingest_pdf_bytes(filename, content)
                    doc = result["document"]
                    results.append(
                        {
                            "filename": filename,
                            "status": result["status"],
                            "document": document_payload(doc),
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "filename": filename,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
        self.send_json({"results": results})

    def read_json(self) -> dict:
        length = clamp_int(self.headers.get("Content-Length"), default=0, low=0, high=20_000_000)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def send_pdf(self, path: Path, filename: str) -> None:
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        status = 200
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                status = 206

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(1024 * 256, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)


def selected_document_ids(params: dict[str, list[str]]) -> list[str] | None:
    values: list[str] = []
    for raw in params.get("docs", []):
        values.extend(part for part in raw.split(",") if part)
    return values or None


def run(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), PaperAgentHandler)
    print(f"PaperAgent Web is running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPaperAgent Web stopped.")
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PaperAgent - 论文智能助手</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <style>
    :root {
      --primary: #FF8A3D;
      --secondary: #48C6F0;
      --bg: #FFF0D6;
      --panel: #ffffff;
      --text: #2c3e50;
      --muted: #7f8c8d;
      --line: #e6dac8;
      --success: #27ae60;
      --danger: #e74c3c;
      --shadow: 0 10px 30px rgba(0,0,0,0.08);
      --grad: linear-gradient(135deg, #FF8A3D 0%, #48C6F0 100%);
      font-size: 16px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
      overflow: hidden;
      font-size: 1rem;
    }

    button, input, textarea, select { font: inherit; }

    .app {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100vh;
    }

    /* Sidebar */
    .sidebar {
      background: #fff;
      display: flex;
      flex-direction: column;
      border-right: 2px solid var(--line);
      box-shadow: 4px 0 15px rgba(0,0,0,0.03);
      z-index: 10;
    }

    .brand {
      padding: 30px 24px;
      background: var(--grad);
      color: #fff;
      text-align: center;
    }

    .brand h1 { margin: 0; font-size: 28px; font-weight: 900; letter-spacing: 1px; }

    .sidebar-content { flex: 1; overflow-y: auto; padding: 24px; }
    .sidebar-section { margin-bottom: 30px; }
    .section-title { font-size: 14px; font-weight: 700; color: var(--primary); text-transform: uppercase; margin-bottom: 15px; border-bottom: 2px solid var(--line); padding-bottom: 5px; }

    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
    .stat { background: #fff; border: 1.5px solid var(--line); padding: 12px 5px; border-radius: 12px; text-align: center; }
    .stat strong { display: block; font-size: 22px; color: var(--primary); }
    .stat span { font-size: 13px; color: var(--muted); }

    .upload-btn { background: #fff; border: 2.5px dashed var(--primary); color: var(--primary); padding: 15px; border-radius: 12px; text-align: center; cursor: pointer; transition: all 0.2s; font-weight: 700; }
    .upload-btn:hover { background: rgba(255, 138, 61, 0.05); }

    .doc-list { display: flex; flex-direction: column; gap: 12px; }
    .doc-item { background: #fff; border: 2px solid var(--line); border-radius: 14px; padding: 18px; cursor: pointer; transition: all 0.2s; position: relative; }
    .doc-item:hover { border-color: var(--primary); transform: translateX(8px); }
    .doc-item.active { border-color: var(--primary); background: rgba(255, 138, 61, 0.08); box-shadow: 0 5px 20px rgba(255, 138, 61, 0.15); border-width: 3px; }
    .doc-item.hit { border-color: var(--secondary); background: rgba(72, 198, 240, 0.1); animation: pulse 2s infinite; border-width: 3px; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(72, 198, 240, 0.4); } 70% { box-shadow: 0 0 0 12px rgba(72, 198, 240, 0); } 100% { box-shadow: 0 0 0 0 rgba(72, 198, 240, 0); } }

    .doc-item h3 { margin: 0; font-size: 16px; font-weight: 800; line-height: 1.45; color: var(--text); overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .doc-item .meta { font-size: 13px; color: var(--muted); margin-top: 10px; display: flex; justify-content: space-between; align-items: flex-end; }
    
    .delete-btn { position: absolute; top: 10px; right: 10px; background: #fff; border: 1.5px solid var(--line); color: var(--danger); border-radius: 50%; width: 24px; height: 24px; display: none; align-items: center; justify-content: center; font-size: 12px; transition: all 0.2s; z-index: 5; }
    .doc-item:hover .delete-btn { display: flex; }
    .delete-btn:hover { background: var(--danger); color: #fff; border-color: var(--danger); transform: scale(1.1); }

    /* Main Area */
    .main { display: flex; flex-direction: column; height: 100vh; overflow: hidden; position: relative; }
    .top-nav { height: 80px; background: #fff; border-bottom: 3px solid var(--line); display: flex; align-items: center; padding: 0 35px; gap: 40px; flex-shrink: 0; }
    .nav-tabs { display: flex; gap: 35px; height: 100%; }
    .nav-tab { height: 100%; display: flex; align-items: center; color: var(--muted); font-size: 18px; font-weight: 800; cursor: pointer; border: 0; background: transparent; position: relative; }
    .nav-tab.active { color: var(--primary); }
    .nav-tab.active::after { content: ''; position: absolute; bottom: -3px; left: 0; right: 0; height: 5px; background: var(--primary); border-radius: 5px 5px 0 0; }
    .icon-btn { width: 46px; height: 46px; border-radius: 14px; border: 2px solid var(--line); background: #fff; color: var(--text); display: inline-flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; }
    .icon-btn:hover { border-color: var(--primary); color: var(--primary); transform: translateY(-2px); box-shadow: var(--shadow); }

    .content-area { flex: 1; overflow-y: auto; padding: 35px; position: relative; scroll-behavior: smooth; }
    .panel { display: none; max-width: 1400px; margin: 0 auto; width: 100%; height: 100%; }
    .panel.active { display: flex; flex-direction: column; }

    /* Cards */
    .card { background: #fff; border-radius: 25px; border: 3px solid var(--line); padding: 30px; margin-bottom: 30px; box-shadow: var(--shadow); position: relative; }
    .card-header { margin-bottom: 25px; border-bottom: 3px solid var(--line); padding-bottom: 18px; display: flex; justify-content: space-between; align-items: center; }
    .card-header h2 { margin: 0; font-size: 24px; font-weight: 900; color: var(--primary); }

    /* Chat UI inside Ask Panel */
    .chat-container { display: flex; gap: 20px; height: 100%; width: 100%; position: relative; overflow: hidden; }
    .chat-sidebar { width: 380px; display: flex; flex-direction: column; gap: 20px; flex-shrink: 0; }
    .chat-main { flex: 1; display: flex; flex-direction: column; background: #fff; border: 3px solid var(--line); border-radius: 25px; position: relative; box-shadow: var(--shadow); overflow: hidden; }
    
    .chat-history { flex: 1; overflow-y: auto; padding: 30px; padding-bottom: 130px; background: #fff; scroll-behavior: smooth; }
    .message { margin-bottom: 35px; display: flex; flex-direction: column; }
    .message.user { align-items: flex-end; }
    .message-bubble { max-width: 88%; padding: 20px 28px; border-radius: 25px; font-size: 17px; line-height: 1.8; position: relative; }
    .user .message-bubble { background: var(--primary); color: #fff; border-bottom-right-radius: 5px; box-shadow: 0 10px 25px rgba(255, 138, 61, 0.2); }
    .bot .message-bubble { background: #fdfdfd; border: 3px solid var(--line); color: var(--text); border-bottom-left-radius: 5px; box-shadow: var(--shadow); }
    .thought-bubble { background: #fdf2e9; border-left: 6px solid var(--primary); padding: 18px; margin-bottom: 20px; font-size: 15px; color: #a35200; border-radius: 10px; font-style: italic; }
    .runtime-badge { margin-bottom: 12px; color: var(--muted); font-size: 13px; display: inline-flex; align-items: center; gap: 6px; background: #f7f7f7; border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; max-width: 100%; }
    .source-list { margin-top:15px; display:flex; gap:10px; flex-wrap:wrap; }
    .source-chip { background:rgba(72,198,240,0.1); border:1px solid var(--secondary); cursor:pointer; max-width: 100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

    /* Bottom Chat Bar */
    .chat-composer {
      position: absolute;
      bottom: 25px;
      left: 25px;
      right: 25px;
      background: #fff;
      border: 4px solid var(--primary);
      border-radius: 30px;
      box-shadow: 0 15px 35px rgba(255, 138, 61, 0.2);
      padding: 10px 14px;
      z-index: 100;
    }
    .composer-context {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 4px 8px;
      border-bottom: 1px solid rgba(255, 138, 61, 0.18);
      margin-bottom: 8px;
    }
    .selected-docs { display:flex; gap:8px; flex-wrap:wrap; align-items:center; min-width:0; flex:1; }
    .doc-selector {
      display:none;
      position:absolute;
      bottom:calc(100% + 12px);
      left:0;
      right:0;
      background:#fff;
      border:2px solid var(--line);
      border-radius:15px;
      padding:15px;
      max-height:300px;
      overflow-y:auto;
      box-shadow:var(--shadow);
      z-index:120;
    }
    .bottom-chat-bar {
      position: relative;
      display: flex;
      align-items: center;
      gap: 15px;
    }
    .bottom-chat-bar textarea { flex: 1; border: 0; background: transparent; padding: 12px 0; font-size: 18px; resize: none; height: 48px; max-height: 150px; outline: none; font-weight: 500; }
    .send-btn { background: var(--primary); color: #fff; border: 0; width: 50px; height: 48px; border-radius: 20px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; }
    .send-btn:hover { transform: scale(1.08); background: #ff751a; }
    .send-btn.is-stop { background: #d32f2f; }
    .send-btn.is-stop:hover { background: #b71c1c; }

    .modal-backdrop { position: fixed; inset: 0; background: rgba(44, 62, 80, 0.34); display: none; align-items: center; justify-content: center; z-index: 500; padding: 24px; }
    .modal-backdrop.open { display: flex; }
    .settings-modal { width: min(560px, 100%); background: #fff; border: 3px solid var(--line); border-radius: 24px; box-shadow: 0 24px 70px rgba(0,0,0,0.18); padding: 26px; }
    .settings-header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding-bottom:16px; border-bottom:2px solid var(--line); margin-bottom:20px; }
    .settings-header h2 { margin:0; color:var(--primary); font-size:22px; }
    .field { margin-bottom:16px; }
    .field label { display:block; font-size:14px; font-weight:800; color:var(--text); margin-bottom:8px; }
    .field select { width:100%; padding:12px 14px; border:2px solid var(--line); border-radius:12px; background:#fff; cursor:pointer; }
    .settings-actions { display:flex; justify-content:flex-end; gap:12px; padding-top:8px; }
    .field input, .field textarea { width:100%; padding:11px 13px; border:2px solid var(--line); border-radius:12px; background:#fff; outline:none; }
    .field textarea { resize:vertical; min-height:78px; line-height:1.5; }
    .field input:focus, .field textarea:focus, .field select:focus { border-color:var(--primary); box-shadow:0 0 0 3px rgba(255, 138, 61, 0.12); }

    /* PDF Viewer */
    .viewer-layout { display: grid; grid-template-columns: 420px 1fr; gap: 30px; height: calc(100vh - 160px); }
    .pdf-pane { background: #fff; border: 3.5px solid var(--line); border-radius: 25px; overflow: hidden; box-shadow: var(--shadow); }
    .chunk-pane { overflow-y: auto; padding-right: 12px; }
    .chunk-card { background: #fff; border: 2px solid var(--line); border-radius: 18px; padding: 22px; margin-bottom: 20px; cursor: pointer; transition: all 0.2s; }
    .chunk-card:hover { border-color: var(--secondary); background: rgba(72, 198, 240, 0.05); }
    .chunk-card.active { border-color: var(--secondary); background: rgba(72, 198, 240, 0.1); border-width: 3.5px; }
    .chunk-card .chunk-page { color:var(--primary); font-weight:900; margin-bottom:8px; }
    .chunk-card .chunk-summary { font-weight:800; line-height:1.55; margin-bottom:10px; color:#25384a; }
    .chunk-card .chunk-snippet { color:#536372; font-size:14px; line-height:1.55; }

    /* Graph Page */
    .graph-page-wrapper { width: 100%; height: 100%; min-height: calc(100vh - 150px); position: relative; background: radial-gradient(circle at 50% 45%, #ffffff 0%, #fffaf2 46%, #f7f9fb 100%); border-radius: 25px; border: 3px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
    #graphCanvas { width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; }
    .graph-page-wrapper::before { content: ''; position: absolute; inset: 30px; border: 1px solid rgba(72, 198, 240, 0.14); border-radius: 50%; pointer-events: none; }
    .graph-page-wrapper::after { content: ''; position: absolute; inset: 12%; border: 1px dashed rgba(255, 138, 61, 0.16); border-radius: 50%; pointer-events: none; }

    /* AI 精读模块 */
    .reader-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 25px; padding-bottom:40px; }
    .module-card { background: #fff; border: 2.5px solid var(--line); border-radius: 22px; padding: 28px; box-shadow: var(--shadow); }
    .module-card h3 { margin: 0 0 18px 0; font-size: 20px; color: var(--primary); display: flex; align-items: center; gap: 12px; font-weight: 900; }
    .module-card p { font-size: 17px; line-height: 1.7; color: #34495e; }
    .reader-hero { display:grid; gap:18px; }
    .reader-tldr { font-size:18px; line-height:1.8; margin:0; max-width:1120px; }
    .paper-info-panel { background:var(--bg); padding:14px; border-radius:15px; font-size:14px; border:2px solid var(--line); display:grid; grid-template-columns: minmax(240px, 2fr) minmax(180px, 1.2fr) 130px 95px 120px auto; gap:10px; align-items:end; }
    .paper-info-panel .field { margin:0; min-width:0; }
    .paper-info-panel .field label { margin-bottom:5px; font-size:12px; color:#6f5d45; }
    .paper-info-panel input, .paper-info-panel select { height:40px; padding:8px 10px; border-width:1.5px; border-radius:10px; overflow:hidden; text-overflow:ellipsis; }
    .paper-info-actions { display:flex; flex-direction:column; gap:6px; align-items:stretch; }
    .paper-info-actions .btn { height:40px; padding:0 14px; white-space:nowrap; }
    .paper-info-status { color:var(--muted); font-size:12px; white-space:nowrap; text-align:right; }
    @media (max-width: 1200px) { .paper-info-panel { grid-template-columns: 1fr 1fr 120px 90px; } .paper-info-actions { grid-column: 1 / -1; flex-direction:row; justify-content:flex-end; align-items:center; } }

    /* Markdown styling */
    .markdown-body h1, .markdown-body h2 { font-size: 1.3em; margin: 1.5rem 0 1rem; border-bottom: 2px solid var(--line); padding-bottom: 0.5rem; }
    .markdown-body ul, .markdown-body ol { padding-left: 1.5rem; margin-bottom: 1rem; }
    .markdown-body li { margin-bottom: 0.5rem; }
    .markdown-body p { margin-bottom: 1rem; }
    .markdown-body code { background: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 5px; font-family: monospace; font-size: 0.9em; }
    .markdown-body .katex-display { overflow-x:auto; overflow-y:hidden; padding:0.35rem 0; }

    .btn { padding: 14px 30px; border-radius: 18px; font-weight: 800; cursor: pointer; border: 2.5px solid var(--line); background: #fff; transition: all 0.2s; font-size: 16px; }
    .btn-primary { background: var(--primary); color: #fff; border: 0; box-shadow: 0 5px 15px rgba(255, 138, 61, 0.3); }
    .btn-primary:hover { opacity: 0.9; transform: translateY(-3px); box-shadow: 0 8px 20px rgba(255, 138, 61, 0.4); }
    .btn-small { padding: 8px 16px; font-size: 14px; border-radius: 12px; }

    /* SSE Thinking */
    .thinking-dots { display: inline-flex; gap: 6px; }
    .thinking-dots span { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
    @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }

    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #dcdde1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand"><h1>PaperAgent</h1></div>
      <div class="sidebar-content">
        <div class="stats">
          <div class="stat"><strong id="statDocs">0</strong><span>论文</span></div>
          <div class="stat"><strong id="statChunks">0</strong><span>切片</span></div>
          <div class="stat"><strong id="statPages">0</strong><span>页数</span></div>
        </div>
        <div class="upload-btn" id="uploadZone">点击或拖入 PDF 论文<input id="fileInput" type="file" multiple accept=".pdf" style="display:none" /></div>
        
        <!-- 书架视图 -->
        <div id="shelfView">
          <div class="section-title">我的书架</div>
          <div class="doc-list" id="docList"></div>
        </div>
        
        <!-- 会话历史视图 -->
        <div id="chatHistoryView" style="display:none;">
          <div class="section-title" style="display:flex; justify-content:space-between; align-items:center;">
            历史会话
            <button class="btn btn-small" id="newSessionBtn" style="padding:4px 12px; font-size:12px;">+ 新会话</button>
          </div>
          <div class="doc-list" id="sessionList"></div>
        </div>
      </div>
    </aside>

    <main class="main">
      <nav class="top-nav">
        <div class="nav-tabs">
          <button class="nav-tab active" data-panel="readerPanel">AI 精读</button>
          <button class="nav-tab" data-panel="askPanel">Agent 问答</button>
          <button class="nav-tab" data-panel="viewerPanel">原文预览</button>
          <button class="nav-tab" data-panel="graphPanel">知识图谱</button>
        </div>
        <div style="flex:1"></div>
        <button class="btn btn-small" id="refreshBtn">同步状态</button>
        <button class="icon-btn" id="settingsBtn" title="模型配置" aria-label="模型配置">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"></path>
            <path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.05.05a2.1 2.1 0 0 1-2.97 2.97l-.05-.05a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.09 1.65v.07a2.1 2.1 0 0 1-4.2 0v-.07a1.8 1.8 0 0 0-1.09-1.65 1.8 1.8 0 0 0-1.98.36l-.05.05a2.1 2.1 0 1 1-2.97-2.97l.05-.05A1.8 1.8 0 0 0 3.85 15a1.8 1.8 0 0 0-1.65-1.09H2.1a2.1 2.1 0 0 1 0-4.2h.1a1.8 1.8 0 0 0 1.65-1.09 1.8 1.8 0 0 0-.36-1.98l-.05-.05a2.1 2.1 0 1 1 2.97-2.97l.05.05a1.8 1.8 0 0 0 1.98.36 1.8 1.8 0 0 0 1.09-1.65V2.1a2.1 2.1 0 0 1 4.2 0v.27a1.8 1.8 0 0 0 1.09 1.65 1.8 1.8 0 0 0 1.98-.36l.05-.05a2.1 2.1 0 0 1 2.97 2.97l-.05.05a1.8 1.8 0 0 0-.36 1.98 1.8 1.8 0 0 0 1.65 1.09h.27a2.1 2.1 0 0 1 0 4.2h-.27A1.8 1.8 0 0 0 19.4 15Z"></path>
          </svg>
        </button>
      </nav>

      <div class="content-area" id="mainContent">
        <!-- AI Reader Panel -->
        <section class="panel active" id="readerPanel" style="display:block">
          <div id="readerContent">
             <div style="text-align:center; padding:100px; color:var(--muted)">
                <h2>请在书架选择论文</h2>
                <p>AI 将为您生成模块化精读报告。</p>
             </div>
          </div>
        </section>

        <!-- Agent Q&A Panel -->
        <section class="panel" id="askPanel">
          <div class="chat-container" style="width:100%;">
            <div class="chat-main" style="width:100%;">
              <div class="chat-history" id="askHistory">
                 <div style="text-align:center; padding: 60px; color:var(--muted)">
                    <h2>向 Agent 提问</h2>
                    <p>开始对话，Agent 会根据论文证据为您解答。</p>
                 </div>
              </div>
              <div class="chat-composer">
                <div class="composer-context">
                  <button class="btn btn-small" id="addDocBtn" title="限定问答范围" style="display:flex; align-items:center; gap:5px;">
                    <span>+</span>
                    <span>引用文档</span>
                  </button>
                  <div id="selectedDocs" class="selected-docs"></div>
                </div>
                <div id="docSelector" class="doc-selector">
                  <div style="font-weight:800; margin-bottom:10px; color:var(--primary);">选择要引用的文档</div>
                  <div id="docSelectorList"></div>
                </div>
                <div class="bottom-chat-bar">
                  <textarea id="chatInput" placeholder="在此输入您的问题 (Shift+Enter 换行)..."></textarea>
                  <button class="send-btn" id="sendBtn"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg></button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- PDF Viewer Panel -->
        <section class="panel" id="viewerPanel">
          <div class="viewer-layout">
            <div class="chunk-pane" id="chunkList"></div>
            <div class="pdf-pane"><iframe id="pdfFrame" src="about:blank" style="width:100%; height:100%; border:0;"></iframe></div>
          </div>
        </section>

        <!-- Graph Panel -->
        <section class="panel" id="graphPanel">
           <div class="graph-page-wrapper">
              <canvas id="graphCanvas" aria-label="三维知识图谱"></canvas>
           </div>
        </section>
      </div>
    </main>
  </div>

  <div class="modal-backdrop" id="settingsModal" aria-hidden="true">
    <div class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settingsTitle">
      <div class="settings-header">
        <h2 id="settingsTitle">模型配置</h2>
        <button class="icon-btn" id="closeSettingsBtn" title="关闭" aria-label="关闭">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      <div class="field">
        <label for="chatModelSelect">问答模型</label>
        <select id="chatModelSelect"></select>
      </div>
      <div class="field">
        <label for="analyzeModelSelect">AI 解析模型</label>
        <select id="analyzeModelSelect"></select>
      </div>
      <div class="settings-actions">
        <button class="btn btn-small" id="cancelSettingsBtn">取消</button>
        <button class="btn btn-primary btn-small" id="saveModelBtn">保存设置</button>
      </div>
    </div>
  </div>

  <script>
    const state = { docs: [], activeDocId: null, activePanel: 'readerPanel', sessions: {}, isStreaming: false, streamStartedAt: 0, currentAbortController: null, currentReader: null, currentMessageId: null, activeRequestId: 0, streamStopped: false, graph: null, chatSessions: [], activeChatSessionId: null, selectedDocIds: [], venues: [] };
    const $ = (s) => document.querySelector(s);
    const $$ = (s) => document.querySelectorAll(s);
    const SEND_ICON = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
    const STOP_ICON = '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>';

    marked.setOptions({ breaks: true, gfm: true });

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function renderMarkdown(value) {
      return marked.parse(normalizeMathText(String(value || '')));
    }

    function normalizeMathText(value) {
      const commandPattern = /\\(?:hat|cdot|langle|rangle|frac|sum|int|sqrt|mathrm|mathbf|bar|overline|times|alpha|beta|gamma|delta|sigma|mu|rho|lambda|theta)\b|[_^]\{/;
      const wrapFormula = (expr) => {
        const trimmed = expr.trim();
        if (!trimmed || trimmed.startsWith('\\(') || trimmed.startsWith('\\[') || trimmed.includes('$')) return expr;
        if (!commandPattern.test(trimmed)) return expr;
        return `\\(${trimmed}\\)`;
      };
      return value.split('\n').map(line => {
        let next = line.replace(/(^|[\s，。：:；;])\(([^()\n]{1,180})\)/g, (match, prefix, expr) => {
          return commandPattern.test(expr) ? `${prefix}${wrapFormula(expr)}` : match;
        });
        if (!commandPattern.test(next) || /(\$\$?[^$]+\$\$?|\\\(|\\\[)/.test(next)) return next;
        const colon = next.search(/[：:]/);
        if (colon >= 0 && colon < next.length - 1) {
          const prefix = next.slice(0, colon + 1);
          const rest = next.slice(colon + 1);
          return prefix + ' ' + wrapFormula(rest);
        }
        if (/^\s*(?:[-*+]|\d+\.)?\s*[A-Za-z0-9_{}\\\s=+\-*/<>≠,.]+$/.test(next)) {
          return wrapFormula(next);
        }
        return next;
      }).join('\n');
    }

    function typesetMath(root) {
      if (!root || !window.renderMathInElement) return;
      try {
        renderMathInElement(root, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
            { left: '$', right: '$', display: false },
          ],
          throwOnError: false,
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        });
      } catch (err) {
        console.warn('Math render failed', err);
      }
    }

    async function init() {
      await loadDocs();
      await loadVenues();
      await loadConfig();
      setupEvents();
      renderSelectedDocs();
      switchPanel('readerPanel');
    }

    async function loadVenues() {
      try {
        const data = await fetch('/api/venues').then(r => r.json());
        state.venues = data.venues || [];
      } catch (err) {
        console.warn('Failed to load venues', err);
        state.venues = [];
      }
    }

    async function loadConfig() {
      const data = await fetch('/api/config').then(r => r.json());
      const llm = data.llm;
      const chatSelect = $('#chatModelSelect');
      const analyzeSelect = $('#analyzeModelSelect');
      
      if (llm.supported_models && llm.supported_models.length > 0) {
        chatSelect.innerHTML = llm.supported_models.map(m => 
          `<option value="${m}" ${m === llm.chat_model ? 'selected' : ''}>${m}</option>`
        ).join('');
        analyzeSelect.innerHTML = llm.supported_models.map(m => 
          `<option value="${m}" ${m === llm.analyze_model ? 'selected' : ''}>${m}</option>`
        ).join('');
      }
    }

    async function saveModels() {
      const chatModel = $('#chatModelSelect').value;
      const analyzeModel = $('#analyzeModelSelect').value;
      
      await fetch('/api/config/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_model: chatModel, analyze_model: analyzeModel })
      }).then(r => r.json());
      
      closeSettings();
    }

    function openSettings() {
      const modal = $('#settingsModal');
      if (!modal) return;
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    }

    function closeSettings() {
      const modal = $('#settingsModal');
      if (!modal) return;
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
    }

    async function loadDocs() {
      const data = await fetch('/api/documents').then(r => r.json());
      state.docs = data.documents;
      $('#statDocs').textContent = data.stats.documents;
      $('#statChunks').textContent = data.stats.chunks;
      $('#statPages').textContent = data.stats.pages;
      renderDocList();
    }

    function renderDocList(hits = []) {
      const list = $('#docList');
      if (state.docs.length === 0) { list.innerHTML = '<div style="text-align:center; padding:20px; color:var(--muted)">空空如也</div>'; return; }
      list.innerHTML = state.docs.map(doc => `
        <div class="doc-item ${state.activeDocId === doc.id ? 'active' : ''} ${hits.includes(doc.id) ? 'hit' : ''}" data-id="${doc.id}">
          <button class="delete-btn" onclick="event.stopPropagation(); deleteDoc('${doc.id}')">✕</button>
          <h3>${doc.title_cn || doc.title}</h3>
          <div class="meta">
            <span style="max-width:160px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${doc.authors && doc.authors.length ? doc.authors.join(', ') : '未知作者'}</span>
            <span style="font-weight:800; color:var(--primary)">评 ${doc.score || '8.5'}</span>
          </div>
        </div>
      `).join('');
      $$('.doc-item').forEach(el => el.addEventListener('click', () => selectDoc(el.dataset.id)));
    }

    async function deleteDoc(id) {
       if(!confirm("确定要删除这篇论文吗？")) return;
       await fetch(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
       init();
    }

    async function selectDoc(id) {
      state.activeDocId = id;
      renderDocList();
      const data = await fetch(`/api/documents/${encodeURIComponent(id)}`).then(r => r.json());
      renderReader(data);
      renderViewer(data);
      if (state.activePanel !== 'viewerPanel' && state.activePanel !== 'graphPanel' && state.activePanel !== 'askPanel') {
          switchPanel('readerPanel');
      }
    }

    function switchPanel(panelId) {
      state.activePanel = panelId;
      $$('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.panel === panelId));
      $$('.panel').forEach(p => {
          if(p.id === panelId) p.style.display = (panelId === 'askPanel' ? 'flex' : 'block');
          else p.style.display = 'none';
      });
      
      // 切换侧边栏视图
      if (panelId === 'askPanel') {
          $('#shelfView').style.display = 'none';
          $('#chatHistoryView').style.display = 'block';
          loadChatSessions();
      } else {
          $('#shelfView').style.display = 'block';
          $('#chatHistoryView').style.display = 'none';
      }
      
      if (panelId === 'graphPanel') loadGraph();
    }
    
    // 加载会话历史
    async function loadChatSessions() {
      // 暂时用简单的会话列表展示，后面可以增强
      state.chatSessions = [];
      state.docs.forEach(doc => {
        const title = doc.title_cn || doc.title;
        state.chatSessions.push({
          id: doc.id,
          title: title.length > 30 ? title.substring(0, 30) + '...' : title,
          lastMessage: '查看该论文的对话'
        });
      });
      renderSessionList();
    }
    
    // 渲染会话列表
    function renderSessionList() {
      const list = $('#sessionList');
      if (state.chatSessions.length === 0) {
        list.innerHTML = '<div style="text-align:center; padding:20px; color:var(--muted)">暂无会话</div>';
        return;
      }
      list.innerHTML = state.chatSessions.map(session => `
        <div class="doc-item ${state.activeChatSessionId === session.id ? 'active' : ''}" data-id="${session.id}">
          <h3>${session.title}</h3>
          <div class="meta">
            <span style="font-size:13px; color:var(--muted);">${session.lastMessage}</span>
          </div>
        </div>
      `).join('');
      $$('#sessionList .doc-item').forEach(el => el.addEventListener('click', () => selectChatSession(el.dataset.id)));
    }
    
    // 选择会话
    async function selectChatSession(id) {
      state.activeChatSessionId = id;
      renderSessionList();
      await loadChatSession(id);
    }
    
    // 加载会话内容
    async function loadChatSession(id) {
       const container = $('#askHistory');
       const doc = state.docs.find(d => d.id === id);
       if (doc) {
         const h = await fetch(`/api/documents/${encodeURIComponent(id)}/chats`).then(r => r.json());
         if (h.sessions?.length > 0) {
           state.sessions[id] = h.sessions[0].id;
           container.innerHTML = h.sessions[0].messages.map(m => renderMessage(m)).join('');
           typesetMath(container);
           setTimeout(() => { container.scrollTop = container.scrollHeight; }, 100);
         } else {
           container.innerHTML = `<div style="text-align:center; padding: 60px; color:var(--muted)"><h2>新会话</h2><p>开始针对论文「${doc.title_cn || doc.title}」提问。</p></div>`;
         }
         // 添加到选中文档
         state.selectedDocIds = [id];
         renderSelectedDocs();
       }
    }
    
    // 新建会话
    function newChatSession() {
      state.activeChatSessionId = null;
      state.selectedDocIds = [];
      renderSessionList();
      renderSelectedDocs();
      $('#askHistory').innerHTML = '<div style="text-align:center; padding: 60px; color:var(--muted)"><h2>向 Agent 提问</h2><p>选择文档或直接提问，Agent 会根据论文证据为您解答。</p></div>';
    }
    
    // 渲染选中的文档
    function renderSelectedDocs() {
      const container = $('#selectedDocs');
      if (state.selectedDocIds.length === 0) {
        container.innerHTML = '<span style="color:var(--muted); font-size:14px;">范围：全库论文</span>';
        return;
      }
      container.innerHTML = state.selectedDocIds.map(docId => {
        const doc = state.docs.find(d => d.id === docId);
        const title = doc ? (doc.title_cn || doc.title) : '未知';
        const shortTitle = title.length > 20 ? title.substring(0, 20) + '...' : title;
        return `<span class="btn-small" style="background:rgba(255, 138, 61, 0.15); border-color:var(--primary); display:flex; align-items:center; gap:6px;">
          @ ${shortTitle}
          <span style="cursor:pointer; font-size:14px; line-height:1;" onclick="removeSelectedDoc('${docId}')">&times;</span>
        </span>`;
      }).join('');
    }
    
    // 移除选中文档
    function removeSelectedDoc(docId) {
      state.selectedDocIds = state.selectedDocIds.filter(id => id !== docId);
      renderSelectedDocs();
    }
    
    // 切换文档选择器
    function toggleDocSelector() {
      const selector = $('#docSelector');
      const isOpen = getComputedStyle(selector).display !== 'none';
      selector.style.display = isOpen ? 'none' : 'block';
      if (selector.style.display === 'block') {
        renderDocSelectorList();
      }
    }
    
    // 渲染文档选择器列表
    function renderDocSelectorList() {
      const list = $('#docSelectorList');
      list.innerHTML = state.docs.map(doc => {
        const isSelected = state.selectedDocIds.includes(doc.id);
        const title = doc.title_cn || doc.title;
        return `<div style="padding:10px; border-radius:10px; cursor:pointer; margin-bottom:5px; ${isSelected ? 'background:rgba(255, 138, 61, 0.1);' : 'background:#f8f9fa;'} hover:background:rgba(255, 138, 61, 0.15);" onclick="toggleDocSelection('${doc.id}')">
          <div style="font-weight:700;">${title.length > 40 ? title.substring(0,40)+'...' : title}</div>
          <div style="font-size:13px; color:var(--muted);">${doc.authors?.slice(0,2).join(', ') || '未知作者'}</div>
        </div>`;
      }).join('');
    }
    
    // 切换文档选择
    function toggleDocSelection(docId) {
      const idx = state.selectedDocIds.indexOf(docId);
      if (idx >= 0) {
        state.selectedDocIds.splice(idx, 1);
      } else {
        state.selectedDocIds.push(docId);
      }
      renderSelectedDocs();
      renderDocSelectorList();
    }

    function renderReader(data) {
      const analysis = data.analysis;
      const content = $('#readerContent');
      if (!analysis || analysis.status === 'missing') {
        content.innerHTML = `<div style="text-align:center; padding:100px;"><h2>该论文尚未解析</h2><button class="btn btn-primary" onclick="analyzeDoc('${data.document.id}')">立即解析</button></div>`;
        return;
      }
      const mods = analysis.modules || {};
      const labels = { motivation: '核心动机', method: '研究方法', result: '实验结果', conclusion: '核心结论', limitations: '局限性', novelty: '创新亮点' };
      let modsHtml = '';
      ['motivation', 'method', 'result', 'conclusion', 'limitations', 'novelty'].forEach(k => {
        const m = mods[k] || { summary: '暂无数据' };
        modsHtml += `<div class="module-card"><h3>${labels[k]}</h3><p>${m.summary}</p>${m.bullets ? `<ul>${m.bullets.map(b => `<li>${b}</li>`).join('')}</ul>` : ''}</div>`;
      });
      const isFallback = analysis.status === 'fallback';
      const meta = data.editable_metadata || {};
      const venues = state.venues.map(v => `<option value="${escapeHtml(v.name)}">${escapeHtml(v.type || '')}</option>`).join('');
      const statusText = isFallback ? '本地抽取(推荐开启 LLM)' : 'LLM 生成';
      content.innerHTML = `<div class="card reader-hero">
        <div class="card-header">
          <h2>${escapeHtml(analysis.title_cn || data.document.title)}</h2>
          <button class="btn btn-small" style="margin-left:auto;" onclick="analyzeDoc('${data.document.id}', true)">重建分析</button>
        </div>
        <div class="paper-info-panel">
          <datalist id="venueList">${venues}</datalist>
          <div class="field"><label for="metaAuthors">作者</label><input id="metaAuthors" value="${escapeHtml((meta.authors || []).join(', '))}" title="${escapeHtml((meta.authors || []).join(', '))}" /></div>
          <div class="field"><label for="metaSource">来源/期刊会议</label><input id="metaSource" list="venueList" value="${escapeHtml(meta.source || '')}" placeholder="Science、CVPR、Nature" /></div>
          <div class="field"><label for="metaDate">日期</label><input id="metaDate" value="${escapeHtml(meta.date || '')}" placeholder="YYYY-MM-DD" /></div>
          <div class="field"><label for="metaScore">评分</label><input id="metaScore" type="number" min="0" max="10" step="0.1" value="${escapeHtml(meta.score ?? '')}" /></div>
          <div class="field"><label for="metaVenueType">类型</label>
            <select id="metaVenueType">
              <option value="journal" ${(meta.venue_type || '') === 'journal' ? 'selected' : ''}>期刊</option>
              <option value="conference" ${(meta.venue_type || '') === 'conference' ? 'selected' : ''}>会议</option>
              <option value="unknown" ${(meta.venue_type || '') === 'unknown' ? 'selected' : ''}>未知</option>
            </select>
          </div>
          <div class="paper-info-actions">
            <button class="btn btn-small" onclick="savePaperMetadata('${data.document.id}')">保存</button>
            <span class="paper-info-status" id="metaSaveStatus">${statusText}</span>
          </div>
        </div>
        <p class="reader-tldr">${escapeHtml(analysis.tldr || '')}</p>
      </div><div class="reader-grid">${modsHtml}</div>`;
    }

    async function savePaperMetadata(id) {
      const status = $('#metaSaveStatus');
      if (status) status.textContent = '正在保存...';
      const payload = {
        authors: $('#metaAuthors')?.value || '',
        source: $('#metaSource')?.value || '',
        venue_type: $('#metaVenueType')?.value || 'unknown',
        date: $('#metaDate')?.value || '',
        score: $('#metaScore')?.value || '',
      };
      try {
        const data = await fetch(`/api/documents/${encodeURIComponent(id)}/metadata`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }).then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        });
        state.venues = data.venues || state.venues;
        await loadDocs();
        await selectDoc(id);
        const nextStatus = $('#metaSaveStatus');
        if (nextStatus) nextStatus.textContent = '已保存，可继续编辑';
      } catch (err) {
        if (status) status.textContent = '保存失败：' + err.message;
      }
    }

    function renderViewer(data) {
      const chunks = data.preview_chunks || data.chunks || [];
      $('#chunkList').innerHTML = chunks.map(c => `<div class="chunk-card" data-page="${c.page_start}" onclick="jumpToPage(${c.page_start}, this)">
        <div class="chunk-page">p.${c.page_start}${c.page_end && c.page_end !== c.page_start ? '-' + c.page_end : ''}${c.section ? ` · ${escapeHtml(c.section)}` : ''}</div>
        <div class="chunk-summary">${escapeHtml(c.summary || '本段大意待生成。')}</div>
        <div class="chunk-snippet">${escapeHtml(c.snippet || '')}</div>
      </div>`).join('');
      if (chunks.length > 0) jumpToPage(chunks[0].page_start || 1);
    }

    function jumpToPage(p, el = null) {
      if(el) { $$('.chunk-card').forEach(c => c.classList.remove('active')); el.classList.add('active'); }
      const url = `/api/documents/${encodeURIComponent(state.activeDocId)}/pdf#page=${p}`;
      const iframe = $('#pdfFrame');
      const newIframe = document.createElement('iframe');
      newIframe.id = 'pdfFrame'; newIframe.src = url; newIframe.style = "width:100%; height:100%; border:0;";
      iframe.parentNode.replaceChild(newIframe, iframe);
    }

    async function analyzeDoc(id, force = false) {
      $('#readerContent').innerHTML = '<div style="text-align:center; padding:100px;"><h3>正在使用 LLM 重建分析，请稍候...</h3></div>';
      await fetch(`/api/documents/${encodeURIComponent(id)}/analysis`, { method: 'POST', body: JSON.stringify({ force: force }) });
      selectDoc(id);
    }

    function uniqueSources(sources) {
      const seen = new Set();
      const unique = [];
      (sources || []).forEach((source, index) => {
        const key = source.chunk_id || source.id || `${source.document_id}:${source.page_start}:${source.page_end}:${source.section || ''}:${(source.snippet || source.text || '').slice(0, 80)}`;
        if (!key || seen.has(key)) return;
        seen.add(key);
        unique.push({ ...source, ref_index: source.ref_index || index + 1 });
      });
      return unique;
    }

    function renderSourceBadges(sources) {
      const unique = uniqueSources(sources);
      if (!unique.length) return '';
      return `<div class="source-list">${unique.map(source => {
        const page = source.page_start ? `p.${source.page_start}` : '';
        const endPage = source.page_end && source.page_end !== source.page_start ? `-${source.page_end}` : '';
        const title = source.paper_title || '未知文档';
        const label = `证据[${source.ref_index}] ${title} ${page}${endPage}`;
        return `<span class="btn-small source-chip" title="${label}" onclick="jumpToSource('${source.document_id}', ${source.page_start || 1}, '${source.chunk_id || source.id || ''}')">${label}</span>`;
      }).join('')}</div>`;
    }

    function renderModelBadge(llm) {
      if (!llm || !llm.model) return '';
      const mode = llm.mode === 'local_fallback' ? '本地兜底' : '远程调用';
      return `<div class="runtime-badge" title="这是后端实际发送给 LLM API 的 model 参数，不是模型自我回答。">实际模型：${llm.model} · ${mode}</div>`;
    }

    function renderMessage(m) {
      const role = m.role === 'user' ? 'user' : 'bot';
      const content = renderMarkdown(m.content || '');
      let thought = m.reasoning ? `<div class="thought-bubble">思考过程：${m.reasoning}</div>` : '';
      return `<div class="message ${role}"><div class="message-bubble">${renderModelBadge(m.llm)}${thought}<div class="markdown-body">${content}</div>${renderSourceBadges(m.sources)}</div></div>`;
    }

    function jumpToSource(docId, page, chunkId) {
      state.activeDocId = docId; renderDocList(); switchPanel('viewerPanel');
      setTimeout(() => {
        const el = $(`[onclick*="jumpToPage(${page})"]`);
        if (el) { el.click(); el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
        else { jumpToPage(page); }
      }, 500);
    }

    function setSendButtonStreaming(streaming) {
      const sendBtn = $('#sendBtn');
      if (!sendBtn) return;
      sendBtn.disabled = false;
      sendBtn.classList.toggle('is-stop', streaming);
      sendBtn.innerHTML = streaming ? STOP_ICON : SEND_ICON;
      sendBtn.title = streaming ? '停止生成' : '发送';
      sendBtn.setAttribute('aria-label', sendBtn.title);
      sendBtn.style.opacity = '';
      sendBtn.style.cursor = '';
    }

    function appendStopNotice(msgId, text) {
      const bubble = msgId ? $(`#${msgId} .message-bubble`) : null;
      if (!bubble || bubble.dataset.stopMarked === '1') return;
      bubble.dataset.stopMarked = '1';
      const note = document.createElement('div');
      note.style.cssText = 'margin-top:12px; color:#888; font-size:14px;';
      note.textContent = text;
      bubble.appendChild(note);
    }

    function stopStreaming(reason = '已停止生成') {
      if (!state.isStreaming) return;
      state.streamStopped = true;
      state.isStreaming = false;
      if (state.currentReader) {
        try { state.currentReader.cancel(reason); } catch (err) { console.warn('reader.cancel failed', err); }
      }
      if (state.currentAbortController) {
        try { state.currentAbortController.abort(reason); } catch (err) { console.warn('abort failed', err); }
      }
      appendStopNotice(state.currentMessageId, reason);
      setSendButtonStreaming(false);
    }

    function isNearChatBottom(container) {
      if (!container) return true;
      return container.scrollHeight - container.scrollTop - container.clientHeight < 120;
    }

    function maybeScrollChatToBottom(container, shouldFollow) {
      if (shouldFollow && container) container.scrollTop = container.scrollHeight;
    }

    async function handleSend() {
      if (state.isStreaming) {
        stopStreaming('已手动停止上一轮回答。');
        return;
      }
      const input = $('#chatInput');
      const sendBtn = $('#sendBtn');
      const q = input.value.trim();
      if (!q) return;
      input.value = ''; input.style.height = '48px';
      const container = $('#askHistory');
      if (container.querySelector('div[style*="text-align:center"]')) container.innerHTML = '';
      container.innerHTML += renderMessage({ role: 'user', content: q });
      typesetMath(container);
      const msgId = 'bot-' + Date.now();
      container.innerHTML += `<div class="message bot" id="${msgId}"><div class="message-bubble"><div class="thinking-dots"><span></span><span></span><span></span></div></div></div>`;
      container.scrollTop = container.scrollHeight;

      const requestId = state.activeRequestId + 1;
      const controller = new AbortController();
      state.activeRequestId = requestId;
      state.isStreaming = true;
      state.streamStopped = false;
      state.streamStartedAt = Date.now();
      state.currentAbortController = controller;
      state.currentReader = null;
      state.currentMessageId = msgId;
      setSendButtonStreaming(true);
      
      // 根据选中文档数量决定 API
      let url, targetKey;
      if (state.selectedDocIds.length === 0) {
        // 全库检索
        url = '/api/ask/stream';
        targetKey = 'all';
      } else if (state.selectedDocIds.length === 1) {
        // 单文档检索
        const targetDoc = state.selectedDocIds[0];
        url = `/api/documents/${encodeURIComponent(targetDoc)}/chat/stream`;
        targetKey = targetDoc;
      } else {
        // 多文档限定检索
        url = '/api/ask/stream';
        targetKey = `multi:${state.selectedDocIds.join('|')}`;
      }
      
      console.log('Sending question:', q);
      console.log('Target key:', targetKey);
      console.log('Session ID:', state.sessions[targetKey]);
      
      let requestTimeoutId = null;
      let timedOut = false;
      let full = '', thought = '', sources = null, llm = null;
      try {
        requestTimeoutId = setTimeout(() => {
          timedOut = true;
          controller.abort('请求超时');
        }, 180000);
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, session_id: state.sessions[targetKey], document_ids: state.selectedDocIds }),
          signal: controller.signal
        });
        
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`HTTP ${res.status}: ${errorText}`);
        }
        
        const reader = res.body.getReader(); const decoder = new TextDecoder();
        state.currentReader = reader;
        let buffer = '';
        let streamDone = false;
        let followStream = true;
        while(state.activeRequestId === requestId && state.isStreaming && !streamDone) {
          const { done, value } = await reader.read(); if(done) break;
          followStream = isNearChatBottom(container);
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for(const line of lines) {
            if (state.activeRequestId !== requestId || !state.isStreaming) break;
            if(line.startsWith('data: ')) {
              let data;
              try {
                data = JSON.parse(line.slice(6));
              } catch(e) {
                console.warn('JSON parse error:', e.message);
                continue;
              }
              if (data.error) {
                throw new Error(data.error);
              }
              if (data.done) {
                streamDone = true;
                state.isStreaming = false;
                break;
              }
              if (data.llm) {
                llm = data.llm;
              }
              if (data.sources) {
                sources = uniqueSources(data.sources);
                renderDocList([...new Set(sources.map(s => s.document_id))]);
              }
              if (data.reasoning) thought += data.reasoning;
              if (data.content || data.answer) full += (data.content || data.answer);
              if (data.session?.id) {
                state.sessions[targetKey] = data.session.id;
                console.log('Session updated:', targetKey, data.session.id);
              }
              const bubble = $(`#${msgId} .message-bubble`);
              let html = renderModelBadge(llm);
              html += thought ? `<div class="thought-bubble">${thought}</div>` : '';
              html += `<div class="markdown-body">${renderMarkdown(full || '...')}</div>`;
              html += renderSourceBadges(sources);
              bubble.innerHTML = html;
              typesetMath(bubble);
            }
          }
          maybeScrollChatToBottom(container, followStream);
        }
        if (state.activeRequestId === requestId && buffer.trim().startsWith('data: ')) {
          try {
            const data = JSON.parse(buffer.trim().slice(6));
            if (data.session?.id) state.sessions[targetKey] = data.session.id;
          } catch(e) {
            console.warn('Trailing SSE parse error:', e.message);
          }
        }
      } catch(e) { 
        console.error('Chat error:', e);
        const wasStopped = state.streamStopped || e.name === 'AbortError';
        const bubble = $(`#${msgId} .message-bubble`);
        if (wasStopped && !timedOut) {
          if (full || thought) appendStopNotice(msgId, '已手动停止上一轮回答。');
          else if (bubble) bubble.innerHTML = '<div style="color:#888;">已手动停止上一轮回答。</div>';
        } else {
          const message = timedOut ? '请求超时，请稍后重试。' : e.message;
          if (bubble) bubble.innerHTML = '<div style="color:#d32f2f;">请求失败: ' + message + '</div>';
        }
      }
      finally {
        if (requestTimeoutId) clearTimeout(requestTimeoutId);
        if (state.activeRequestId === requestId) {
          state.isStreaming = false;
          state.streamStartedAt = 0;
          state.currentAbortController = null;
          state.currentReader = null;
          state.currentMessageId = null;
          state.streamStopped = false;
          setSendButtonStreaming(false);
        }
        if (state.selectedDocIds.length === 1 && state.sessions[targetKey]) {
          try {
            const hist = await api(`/api/documents/${encodeURIComponent(state.selectedDocIds[0])}/chats`);
            console.log('Chat history refreshed:', hist.sessions?.[0]?.id, hist.sessions?.[0]?.messages?.length);
          } catch (err) {
            console.warn('Failed to refresh chat history', err);
          }
        }
      }
    }

    let graphRenderer = null;

    class Graph3DRenderer {
      constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.edges = [];
        this.nodeMap = new Map();
        this.projected = [];
        this.view = { yaw: 0, pitch: -0.18, spin: 0, distance: 2.55, panX: 0, panY: 0 };
        this.pointer = { button: null, lastX: 0, lastY: 0, startX: 0, startY: 0, moved: false };
        this.hoverNode = null;
        this.lastFrame = performance.now();
        this.pixelRatio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        this.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        this.bindEvents();
        this.bindResize();
        this.resize();
        this.frameId = requestAnimationFrame((time) => this.tick(time));
      }

      bindResize() {
        if (window.ResizeObserver && this.canvas.parentElement) {
          this.resizeObserver = new ResizeObserver(() => this.resize());
          this.resizeObserver.observe(this.canvas.parentElement);
        } else {
          window.addEventListener('resize', () => this.resize());
        }
      }

      bindEvents() {
        this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
        this.canvas.addEventListener('auxclick', (event) => event.preventDefault());
        this.canvas.addEventListener('wheel', (event) => {
          event.preventDefault();
          const zoomFactor = Math.exp(event.deltaY * 0.001);
          this.view.distance = this.clamp(this.view.distance * zoomFactor, 1.35, 6.2);
        }, { passive: false });

        this.canvas.addEventListener('pointerdown', (event) => {
          this.pointer.button = event.button;
          this.pointer.lastX = event.clientX;
          this.pointer.lastY = event.clientY;
          this.pointer.startX = event.clientX;
          this.pointer.startY = event.clientY;
          this.pointer.moved = false;
          if (event.button === 2) this.canvas.style.cursor = 'grabbing';
          if (event.button === 1) this.canvas.style.cursor = 'move';
          if (event.button === 1 || event.button === 2) event.preventDefault();
          if (this.canvas.setPointerCapture) this.canvas.setPointerCapture(event.pointerId);
        });

        this.canvas.addEventListener('pointermove', (event) => {
          if (this.pointer.button !== null) {
            const dx = event.clientX - this.pointer.lastX;
            const dy = event.clientY - this.pointer.lastY;
            if (Math.abs(event.clientX - this.pointer.startX) + Math.abs(event.clientY - this.pointer.startY) > 5) {
              this.pointer.moved = true;
            }
            if (this.pointer.button === 2) {
              this.view.yaw += dx * 0.006;
              this.view.pitch = this.clamp(this.view.pitch + dy * 0.006, -1.35, 1.35);
            } else if (this.pointer.button === 1) {
              this.view.panX += dx;
              this.view.panY += dy;
            }
            this.pointer.lastX = event.clientX;
            this.pointer.lastY = event.clientY;
            if (this.pointer.button === 1 || this.pointer.button === 2) event.preventDefault();
          } else {
            this.updateHover(event);
          }
        });

        this.canvas.addEventListener('pointerup', (event) => {
          const wasClick = this.pointer.button === 0 && !this.pointer.moved;
          this.updateHover(event);
          if (wasClick && this.hoverNode && this.hoverNode.type === 'document') {
            selectDoc(this.hoverNode.id);
          }
          this.pointer.button = null;
          this.updateCursor();
          if (this.canvas.releasePointerCapture) {
            try { this.canvas.releasePointerCapture(event.pointerId); } catch (err) {}
          }
        });

        this.canvas.addEventListener('pointercancel', () => {
          this.pointer.button = null;
          this.updateCursor();
        });

        this.canvas.addEventListener('pointerleave', () => {
          if (this.pointer.button === null) {
            this.hoverNode = null;
            this.canvas.title = '';
            this.updateCursor();
          }
        });
      }

      setGraph(graph) {
        const sourceNodes = graph.nodes || [];
        this.nodes = sourceNodes.map((node) => ({ ...node }));
        this.nodeMap = new Map(this.nodes.map((node) => [node.id, node]));
        this.edges = (graph.edges || []).filter((edge) => this.nodeMap.has(edge.source) && this.nodeMap.has(edge.target));
        this.layoutNodes();
      }

      layoutNodes() {
        const docs = this.nodes.filter((node) => node.type === 'document');
        const concepts = this.nodes.filter((node) => node.type !== 'document');
        const maxWeight = Math.max(1, ...concepts.map((node) => Number(node.weight || 1)));

        docs.forEach((node, index) => {
          node._dir = docs.length === 1 ? { x: 0, y: 0, z: 0 } : this.fibonacciPoint(index, docs.length, Math.PI / 8);
          node._shell = docs.length === 1 ? 0 : 0.54;
        });

        concepts.forEach((node, index) => {
          const weight = Number(node.weight || 1);
          node._dir = this.fibonacciPoint(index, concepts.length || 1, Math.PI / 2.8);
          node._shell = 0.86 + Math.sqrt(weight / maxWeight) * 0.16;
        });
      }

      fibonacciPoint(index, count, phase) {
        if (count <= 1) return { x: 0, y: 0, z: 1 };
        const y = 1 - (2 * (index + 0.5)) / count;
        const radius = Math.sqrt(Math.max(0, 1 - y * y));
        const theta = index * Math.PI * (3 - Math.sqrt(5)) + phase;
        return { x: Math.cos(theta) * radius, y, z: Math.sin(theta) * radius };
      }

      resize() {
        const rect = this.canvas.parentElement ? this.canvas.parentElement.getBoundingClientRect() : this.canvas.getBoundingClientRect();
        const width = Math.max(320, Math.floor(rect.width || 1000));
        const height = Math.max(320, Math.floor(rect.height || 720));
        this.pixelRatio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        this.canvas.width = Math.floor(width * this.pixelRatio);
        this.canvas.height = Math.floor(height * this.pixelRatio);
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
        this.ctx.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
      }

      tick(time) {
        const delta = Math.min(48, time - this.lastFrame);
        this.lastFrame = time;
        if (!this.reducedMotion && this.pointer.button === null && state.activePanel === 'graphPanel') {
          this.view.spin += delta * 0.000055;
        }
        this.draw();
        this.frameId = requestAnimationFrame((nextTime) => this.tick(nextTime));
      }

      draw() {
        const width = this.canvas.clientWidth || 1000;
        const height = this.canvas.clientHeight || 720;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, width, height);

        if (!this.nodes.length) {
          ctx.fillStyle = '#7f8c8d';
          ctx.font = '700 18px "Microsoft YaHei", sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('图谱加载中...', width / 2, height / 2);
          return;
        }

        const scene = this.sceneMetrics(width, height);
        this.drawSphereGuides(scene);
        this.projected = this.nodes.map((node) => this.projectNode(node, scene)).filter(Boolean);
        const byId = new Map(this.projected.map((item) => [item.node.id, item]));

        this.edges
          .map((edge) => ({ edge, a: byId.get(edge.source), b: byId.get(edge.target) }))
          .filter((item) => item.a && item.b)
          .sort((left, right) => ((left.a.z + left.b.z) / 2) - ((right.a.z + right.b.z) / 2))
          .forEach((item) => this.drawEdge(item.edge, item.a, item.b));

        this.projected
          .slice()
          .sort((left, right) => left.z - right.z)
          .forEach((item) => this.drawNode(item));

        this.projected
          .slice()
          .sort((left, right) => left.z - right.z)
          .forEach((item) => this.drawLabel(item));
      }

      sceneMetrics(width, height) {
        const radius = Math.min(width, height) * 0.38;
        const distance = radius * this.view.distance + 120;
        return {
          width,
          height,
          radius,
          distance,
          focal: Math.min(width, height) * 1.05,
          centerX: width / 2 + this.view.panX,
          centerY: height / 2 + this.view.panY,
        };
      }

      rotatePoint(point) {
        const yaw = this.view.yaw + this.view.spin;
        const pitch = this.view.pitch;
        const cosY = Math.cos(yaw);
        const sinY = Math.sin(yaw);
        const cosX = Math.cos(pitch);
        const sinX = Math.sin(pitch);
        const x1 = point.x * cosY - point.z * sinY;
        const z1 = point.x * sinY + point.z * cosY;
        return {
          x: x1,
          y: point.y * cosX - z1 * sinX,
          z: point.y * sinX + z1 * cosX,
        };
      }

      projectWorld(point, scene) {
        const rotated = this.rotatePoint(point);
        const depth = scene.distance - rotated.z;
        if (depth <= 24) return null;
        const scale = scene.focal / depth;
        return {
          x: scene.centerX + rotated.x * scale,
          y: scene.centerY + rotated.y * scale,
          z: rotated.z,
          scale,
        };
      }

      projectNode(node, scene) {
        const shell = node._shell ?? 1;
        const dir = node._dir || { x: 0, y: 0, z: 1 };
        const point = { x: dir.x * scene.radius * shell, y: dir.y * scene.radius * shell, z: dir.z * scene.radius * shell };
        const projected = this.projectWorld(point, scene);
        if (!projected) return null;
        const depthRatio = this.clamp((projected.z / scene.radius + 1) / 2, 0, 1);
        const weight = Math.max(1, Number(node.weight || 1));
        const baseSize = node.type === 'document' ? 13 : 6.5 + Math.sqrt(weight) * 0.9;
        return {
          node,
          x: projected.x,
          y: projected.y,
          z: projected.z,
          scale: projected.scale,
          depthRatio,
          radius: this.clamp(baseSize * projected.scale, node.type === 'document' ? 9 : 5, node.type === 'document' ? 26 : 18),
          alpha: this.clamp(0.18 + depthRatio * 0.82, 0.18, 1),
        };
      }

      drawSphereGuides(scene) {
        const latitudes = [-60, -30, 0, 30, 60];
        const longitudes = [0, 30, 60, 90, 120, 150];
        latitudes.forEach((lat) => {
          const rad = lat * Math.PI / 180;
          const y = Math.sin(rad) * scene.radius;
          const ringRadius = Math.cos(rad) * scene.radius;
          const points = [];
          for (let i = 0; i <= 96; i++) {
            const theta = i / 96 * Math.PI * 2;
            points.push({ x: Math.cos(theta) * ringRadius, y, z: Math.sin(theta) * ringRadius });
          }
          this.drawProjectedLine(points, scene, 'rgba(72, 198, 240, 0.10)', 1);
        });
        longitudes.forEach((lon) => {
          const rad = lon * Math.PI / 180;
          const points = [];
          for (let i = 0; i <= 96; i++) {
            const theta = i / 96 * Math.PI * 2;
            points.push({
              x: Math.cos(rad) * Math.cos(theta) * scene.radius,
              y: Math.sin(theta) * scene.radius,
              z: Math.sin(rad) * Math.cos(theta) * scene.radius,
            });
          }
          this.drawProjectedLine(points, scene, 'rgba(255, 138, 61, 0.08)', 1);
        });
      }

      drawProjectedLine(points, scene, color, width) {
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        let started = false;
        points.forEach((point) => {
          const projected = this.projectWorld(point, scene);
          if (!projected) {
            started = false;
            return;
          }
          if (!started) {
            ctx.moveTo(projected.x, projected.y);
            started = true;
          } else {
            ctx.lineTo(projected.x, projected.y);
          }
        });
        ctx.stroke();
        ctx.restore();
      }

      drawEdge(edge, a, b) {
        const ctx = this.ctx;
        const alpha = this.clamp((a.alpha + b.alpha) * 0.16, 0.05, edge.type === 'related' ? 0.26 : 0.34);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = edge.type === 'related' ? '#FF8A3D' : '#48C6F0';
        ctx.lineWidth = this.clamp(Math.sqrt(Number(edge.weight || 1)) * 0.75, 0.8, 4.2);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
      }

      drawNode(item) {
        const { node } = item;
        const ctx = this.ctx;
        const isDocument = node.type === 'document';
        const isActive = isDocument && node.id === state.activeDocId;
        const isHover = this.hoverNode && this.hoverNode.id === node.id;
        const fill = isDocument ? '#FF8A3D' : '#48C6F0';
        ctx.save();
        ctx.globalAlpha = item.alpha;
        ctx.shadowColor = isDocument ? 'rgba(255, 138, 61, 0.35)' : 'rgba(72, 198, 240, 0.30)';
        ctx.shadowBlur = isHover || isActive ? 18 : 9;
        ctx.beginPath();
        ctx.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = isActive ? 4 : 2.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        if (isActive || isHover) {
          ctx.globalAlpha = 0.75;
          ctx.lineWidth = 2;
          ctx.strokeStyle = isDocument ? '#FF8A3D' : '#48C6F0';
          ctx.beginPath();
          ctx.arc(item.x, item.y, item.radius + 8, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }

      drawLabel(item) {
        const { node } = item;
        const isDocument = node.type === 'document';
        const isHover = this.hoverNode && this.hoverNode.id === node.id;
        if (!isDocument && !isHover && item.depthRatio < 0.48) return;
        const text = this.shortLabel(node.label || node.title || node.id, isDocument ? 22 : 16);
        const ctx = this.ctx;
        const fontSize = isDocument ? 13 : 12;
        const y = item.y + item.radius + 8;
        ctx.save();
        ctx.globalAlpha = this.clamp(item.alpha + (isHover ? 0.25 : 0), 0.22, 1);
        ctx.font = `900 ${fontSize}px "Microsoft YaHei", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.strokeText(text, item.x, y);
        ctx.fillStyle = isDocument ? '#2c3e50' : '#33616f';
        ctx.fillText(text, item.x, y);
        ctx.restore();
      }

      updateHover(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        let hit = null;
        this.projected
          .slice()
          .sort((left, right) => right.z - left.z)
          .some((item) => {
            const dx = item.x - x;
            const dy = item.y - y;
            const hitRadius = Math.max(12, item.radius + 6);
            if (dx * dx + dy * dy <= hitRadius * hitRadius) {
              hit = item.node;
              return true;
            }
            return false;
          });
        this.hoverNode = hit;
        this.canvas.title = hit ? (hit.title || hit.label || '') : '';
        this.updateCursor();
      }

      updateCursor() {
        if (this.pointer.button === 2) this.canvas.style.cursor = 'grabbing';
        else if (this.pointer.button === 1) this.canvas.style.cursor = 'move';
        else if (this.hoverNode && this.hoverNode.type === 'document') this.canvas.style.cursor = 'pointer';
        else this.canvas.style.cursor = 'grab';
      }

      shortLabel(value, limit) {
        const label = String(value || '').trim();
        if (label.length <= limit) return label;
        return label.slice(0, Math.max(0, limit - 3)).trimEnd() + '...';
      }

      clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }
    }

    function renderGraph(graph) {
      const canvas = $('#graphCanvas');
      if (!canvas) return;
      if (!graphRenderer) graphRenderer = new Graph3DRenderer(canvas);
      graphRenderer.setGraph(graph);
      graphRenderer.resize();
    }

    async function loadGraph() {
      const g = await fetch('/api/graph').then(r => r.json());
      state.graph = g;
      renderGraph(g);
    }

    function setupEvents() {
      $$('.nav-tab').forEach(t => t.addEventListener('click', () => switchPanel(t.dataset.panel)));
      $('#sendBtn').addEventListener('click', handleSend);
      $('#chatInput').addEventListener('keydown', (e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });
      $('#refreshBtn').addEventListener('click', init);
      $('#uploadZone').addEventListener('click', () => $('#fileInput').click());
      $('#fileInput').addEventListener('change', (e) => uploadFiles(e.target.files));
      const tx = $('#chatInput');
      tx.addEventListener('input', function() { this.style.height = '48px'; this.style.height = (this.scrollHeight) + 'px'; });
      
      // 新会话按钮
      const newSessionBtn = $('#newSessionBtn');
      if (newSessionBtn) newSessionBtn.addEventListener('click', newChatSession);
      
      // 文档选择按钮
      const addDocBtn = $('#addDocBtn');
      if (addDocBtn) addDocBtn.addEventListener('click', toggleDocSelector);
      
      // 模型保存按钮
      const saveModelBtn = $('#saveModelBtn');
      if (saveModelBtn) saveModelBtn.addEventListener('click', saveModels);
      const settingsBtn = $('#settingsBtn');
      if (settingsBtn) settingsBtn.addEventListener('click', openSettings);
      const closeSettingsBtn = $('#closeSettingsBtn');
      if (closeSettingsBtn) closeSettingsBtn.addEventListener('click', closeSettings);
      const cancelSettingsBtn = $('#cancelSettingsBtn');
      if (cancelSettingsBtn) cancelSettingsBtn.addEventListener('click', closeSettings);
      const settingsModal = $('#settingsModal');
      if (settingsModal) {
        settingsModal.addEventListener('click', (e) => {
          if (e.target === settingsModal) closeSettings();
        });
      }
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeSettings();
      });
      
      // 点击其他地方关闭文档选择器
      document.addEventListener('click', (e) => {
        const selector = $('#docSelector');
        const addBtn = $('#addDocBtn');
        if (selector && selector.style.display === 'block' && 
            addBtn && !addBtn.contains(e.target) && 
            !selector.contains(e.target)) {
          selector.style.display = 'none';
        }
      });
    }

    async function uploadFiles(files) {
      const form = new FormData(); for(let f of files) form.append('files', f);
      await fetch('/api/upload', { method: 'POST', body: form });
      init();
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PaperAgent Web UI.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8601)
    args = parser.parse_args()
    run(args.host, args.port)
