from __future__ import annotations

from .models import Chunk, SearchResult
from .llm import answer_with_llm, chat_completion, llm_config
from .models import PaperDocument
from .retriever import HybridRetriever
from .store import DocumentStore
from .summarizer import summarize_chunks, summarize_results


class PaperAgent:
    """Tool-using paper assistant built around a local RAG pipeline."""

    def __init__(self, store: DocumentStore, retriever: HybridRetriever):
        self.store = store
        self.retriever = retriever

    def ask(self, question: str, top_k: int = 6) -> dict:
        plan = [
            {
                "tool": "advanced_hybrid_retrieve",
                "why": "使用 BM25、关键词命中、字符语义相似度和 MMR 去重召回论文证据块。",
            },
            {
                "tool": "llm_or_extract_answer",
                "why": "若配置了 LLM，则基于证据块生成回答；否则使用本地抽取式摘要兜底。",
            },
            {
                "tool": "source_citation",
                "why": "保留文档标题、页码和片段，便于用户回到原文核验。",
            },
        ]
        results = self.retriever.search(question, top_k=top_k, mode="hybrid")
        if not results:
            return {
                "answer": "知识库中没有找到足够相关的内容。可以换一个关键词，或先上传更多 PDF。",
                "plan": plan,
                "sources": [],
            }

        llm_answer = answer_with_llm(question, results)
        answer = llm_answer or summarize_results(results, question, max_sentences=min(8, max(4, top_k)))
        return {
            "answer": answer,
            "plan": plan,
            "sources": [result.to_source_dict(query=question) for result in results],
            "llm": llm_config(),
        }

    def ask_stream(self, question: str, top_k: int = 6, document_ids: list[str] | None = None):
        """Streaming version of ask."""
        results = self.retriever.search(question, top_k=top_k, mode="hybrid", document_ids=document_ids)
        if not results:
            yield {"answer": "知识库中没有找到足够相关的内容。", "sources": []}
            return

        from .llm import build_messages, stream_chat_completion, llm_config
        
        sources = [result.to_source_dict(query=question) for result in results]
        config = llm_config()
        llm_meta = {
            "provider": config.get("provider"),
            "model": config.get("chat_model"),
            "role": "chat",
            "enabled": config.get("enabled"),
            "paper_content_allowed": config.get("paper_content_allowed"),
        }
        
        if not config.get("paper_content_allowed"):
            # Fallback to extractive summary (non-streaming for now)
            answer = summarize_results(results, question, max_sentences=min(8, max(4, top_k)))
            yield {"answer": answer, "sources": sources, "llm": llm_meta | {"mode": "local_fallback"}}
            return

        messages = build_messages(question, results)
        yield {"sources": sources, "llm": llm_meta | {"mode": "remote_stream"}} # Send sources first
        
        for chunk in stream_chat_completion(messages, max_tokens=2048, timeout=120):
            yield chunk

    def summarize_document(
        self,
        document_id: str,
        query: str | None = None,
        max_sentences: int = 8,
    ) -> str:
        document = self.store.get_document(document_id)
        if not document:
            return "没有找到该文档。"
        chunks = self.store.chunks_for_document(document_id)
        return summarize_chunks(chunks, query=query, max_sentences=max_sentences)

    def summarize_library(
        self,
        query: str | None = None,
        max_sentences: int = 10,
    ) -> str:
        chunks = self.store.list_chunks()
        if not chunks:
            return "知识库为空，请先上传 PDF。"
        if query:
            results = self.retriever.search(query, top_k=max(12, max_sentences * 2), mode="hybrid")
            selected_chunks: list[Chunk] = [result.chunk for result in results]
        else:
            selected_chunks = chunks
        return summarize_chunks(selected_chunks, query=query, max_sentences=max_sentences)

    def ask_document(
        self,
        document: PaperDocument,
        question: str,
        analysis: dict | None,
        history: dict,
        top_k: int = 6,
    ) -> dict:
        results = self.retriever.search(
            question,
            top_k=top_k,
            mode="hybrid",
            document_ids=[document.id],
        )
        if not results:
            fallback_chunks = self._representative_chunks(document.id, top_k)
            results = [
                SearchResult(
                    chunk=chunk,
                    score=0.1,
                    keyword_score=0.0,
                    semantic_score=0.0,
                    bm25_score=0.0,
                )
                for chunk in fallback_chunks
            ]
            if not results:
                return {
                    "answer": "这篇论文中没有找到可用文本。请检查 PDF 是否为扫描版，或是否成功抽取文本。",
                    "sources": [],
                    "llm": llm_config(),
                }

        answer = self._document_llm_answer(document, question, analysis, history, results)
        if not answer:
            answer = summarize_results(results, question, max_sentences=min(8, max(4, top_k)))
        return {
            "answer": answer,
            "sources": [result.to_source_dict(query=question) for result in results],
            "llm": llm_config(),
        }

    def ask_document_stream(
        self,
        document: PaperDocument,
        question: str,
        analysis: dict | None,
        history: dict,
        top_k: int = 6,
    ):
        """Streaming version of document-specific Q&A."""
        results = self.retriever.search(
            question,
            top_k=top_k,
            mode="hybrid",
            document_ids=[document.id],
        )
        if not results:
            fallback_chunks = self._representative_chunks(document.id, top_k)
            results = [
                SearchResult(
                    chunk=chunk,
                    score=0.1,
                    keyword_score=0.0,
                    semantic_score=0.0,
                    bm25_score=0.0,
                )
                for chunk in fallback_chunks
            ]
            if not results:
                yield {"answer": "这篇论文中没有找到可用文本。", "sources": []}
                return

        sources = [result.to_source_dict(query=question) for result in results]
        
        from .llm import stream_chat_completion, llm_config
        config = llm_config()
        llm_meta = {
            "provider": config.get("provider"),
            "model": config.get("chat_model"),
            "role": "chat",
            "enabled": config.get("enabled"),
            "paper_content_allowed": config.get("paper_content_allowed"),
        }
        if not config.get("enabled") or not config.get("paper_content_allowed"):
            answer = summarize_results(results, question, max_sentences=min(8, max(4, top_k)))
            yield {"answer": answer, "sources": sources, "llm": llm_meta | {"mode": "local_fallback"}}
            return

        yield {"sources": sources, "llm": llm_meta | {"mode": "remote_stream"}} # Send sources first
        messages = self._build_document_messages(document, question, analysis, history, results)
        
        for chunk in stream_chat_completion(messages):
            yield chunk

    def _build_document_messages(
        self,
        document: PaperDocument,
        question: str,
        analysis: dict | None,
        history: dict,
        results,
    ) -> list[dict[str, str]]:
        recent = []
        for session in (history or {}).get("sessions", [])[:1]:
            recent.extend(session.get("messages", [])[-8:])
        evidence = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            evidence.append(
                f"证据片段[{index}] {chunk.section or 'Unknown'} p.{chunk.page_start}-{chunk.page_end} "
                f"chunk={chunk.id}\n{chunk.text[:2400]}"
            )
        return [
            {
                "role": "system",
                "content": (
                    "你是 PaperAgent 的论文内多轮问答 Agent。你正在围绕同一篇论文持续追问。"
                    "回答必须基于论文精读卡片、最近会话和检索证据。"
                    "如果用户追问里的“它/这个方法/上述结果”指代不清，要结合最近会话解析。"
                    "回答中文为主，必要时给出英文术语；关键结论后标注证据片段编号，例如 [1]。"
                    "编号指检索证据片段，不是文档编号；不要写“文档[1]”。"
                    "不要简单照搬原文，要以模块化方式解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"论文：{document.title}\n\n"
                    f"精读卡片：{analysis or {}}\n\n"
                    f"最近会话：{recent}\n\n"
                    f"本轮问题：{question}\n\n"
                    f"检索证据：\n" + "\n\n".join(evidence)
                ),
            },
        ]

    def _document_llm_answer(
        self,
        document: PaperDocument,
        question: str,
        analysis: dict | None,
        history: dict,
        results,
    ) -> str | None:
        config = llm_config()
        if not config.get("enabled") or not config.get("paper_content_allowed"):
            return None
        messages = self._build_document_messages(document, question, analysis, history, results)
        return chat_completion(messages, temperature=0.2, max_tokens=4096)

    def _representative_chunks(self, document_id: str, limit: int) -> list[Chunk]:
        chunks = self.store.chunks_for_document(document_id)
        if not chunks:
            return []
        priority_words = ["abstract", "introduction", "method", "result", "discussion", "conclusion"]
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                next(
                    (idx for idx, word in enumerate(priority_words) if word in (chunk.section or "").lower()),
                    99,
                ),
                chunk.page_start,
                chunk.chunk_index,
            ),
        )
        return ranked[:limit]
