from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paperagent.agent import PaperAgent
from paperagent.models import Chunk, PaperDocument
from paperagent.retriever import HybridRetriever
from paperagent.store import DocumentStore


def main() -> None:
    data_dir = PROJECT_ROOT / ".tmp_smoke"
    data_dir.mkdir(exist_ok=True)
    library = data_dir / "library.json"
    if library.exists():
        library.unlink()

    store = DocumentStore(data_dir)
    document = PaperDocument(
        id="paper-demo",
        title="Demo RAG Paper",
        filename="demo.pdf",
        stored_path="",
        sha256="demo",
        pages=2,
        chunks=2,
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )
    chunks = [
        Chunk(
            id="paper-demo-c00001",
            document_id=document.id,
            paper_title=document.title,
            page_start=1,
            page_end=1,
            text="Retrieval augmented generation combines document retrieval with answer generation. It improves factual grounding by citing source passages.",
            keywords=["retrieval", "generation", "grounding"],
        ),
        Chunk(
            id="paper-demo-c00002",
            document_id=document.id,
            paper_title=document.title,
            page_start=2,
            page_end=2,
            text="Agent systems can plan tool calls, inspect retrieved evidence, and summarize findings for users.",
            keywords=["agent", "tools", "summary"],
        ),
    ]
    store.add_document(document, chunks)
    retriever = HybridRetriever(store.list_chunks())
    results = retriever.search("RAG source citation", top_k=2)
    assert results, "retriever returned no results"
    assert results[0].chunk.document_id == document.id

    agent = PaperAgent(store, retriever)
    response = agent.ask("How does RAG improve factual grounding?", top_k=2)
    assert response["sources"], "agent returned no sources"
    print("smoke test ok")


if __name__ == "__main__":
    main()
