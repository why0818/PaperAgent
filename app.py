from __future__ import annotations

from pathlib import Path

import streamlit as st

from paperagent.knowledge_base import KnowledgeBase
from paperagent.paths import DATA_DIR


st.set_page_config(
    page_title="PaperAgent",
    page_icon="PA",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(DATA_DIR)


def reset_kb_cache() -> KnowledgeBase:
    get_kb.clear()
    return get_kb()


def render_source(source: dict, index: int) -> None:
    title = source.get("paper_title", "Untitled")
    page = source.get("page_start", "?")
    score = source.get("score", 0.0)
    with st.expander(f"[{index}] {title} - p.{page} - score {score:.3f}"):
        st.caption(f"chunk: {source.get('chunk_id')}")
        st.write(source.get("snippet", ""))


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        background: #ffffff;
    }
    .pa-muted { color: #6b7280; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

kb = get_kb()
documents = kb.list_documents()
chunks = kb.store.list_chunks()

with st.sidebar:
    st.title("PaperAgent")
    st.caption("RAG + Agent paper knowledge base")
    col_a, col_b = st.columns(2)
    col_a.metric("文档", len(documents))
    col_b.metric("切片", len(chunks))
    st.caption(f"数据目录: {Path(DATA_DIR).resolve()}")
    if st.button("刷新知识库", use_container_width=True):
        kb = reset_kb_cache()
        st.rerun()

st.title("PaperAgent")
st.markdown(
    '<div class="pa-muted">上传 PDF，管理论文知识库，并通过关键词 + 语义混合检索完成汇总和问答。</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_library, tab_search, tab_agent, tab_summary = st.tabs(
    ["上传入库", "知识库", "检索", "Agent 问答", "汇总"]
)

with tab_upload:
    st.subheader("上传 PDF 文档")
    uploaded_files = st.file_uploader(
        "选择一个或多个 PDF 文件",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    col_import, col_hint = st.columns([1, 3])
    with col_import:
        import_clicked = st.button("导入知识库", type="primary", use_container_width=True)
    with col_hint:
        st.caption("导入后会抽取页面文本、切片、建立混合检索索引，并自动去重。")

    if import_clicked:
        if not uploaded_files:
            st.warning("请先选择 PDF 文件。")
        else:
            progress = st.progress(0)
            messages: list[str] = []
            for idx, file in enumerate(uploaded_files, start=1):
                with st.spinner(f"正在导入 {file.name}"):
                    result = kb.ingest_pdf_bytes(file.name, file.getvalue())
                progress.progress(idx / len(uploaded_files))
                doc = result.get("document")
                if result["status"] == "duplicate":
                    messages.append(f"已存在: {doc.title}")
                else:
                    messages.append(
                        f"已导入: {doc.title} / {doc.pages} 页 / {doc.chunks} 个切片"
                    )
            st.success("\n".join(messages))
            kb = reset_kb_cache()
            st.rerun()

with tab_library:
    st.subheader("知识库管理")
    docs = kb.list_documents()
    if not docs:
        st.info("知识库还没有文档，请先上传 PDF。")
    else:
        for doc in docs:
            with st.container(border=True):
                st.write(f"**{doc.title}**")
                c1, c2, c3, c4 = st.columns([1, 1, 2, 3])
                c1.metric("页数", doc.pages)
                c2.metric("切片", doc.chunks)
                c3.caption(doc.filename)
                c4.caption(doc.created_at)

        selected_title = st.selectbox(
            "选择文档",
            options=[doc.id for doc in docs],
            format_func=lambda doc_id: kb.store.get_document(doc_id).title,
        )
        selected_doc = kb.store.get_document(selected_title)
        if selected_doc:
            st.write(f"**{selected_doc.title}**")
            st.caption(f"SHA256: {selected_doc.sha256}")
            c1, c2, c3 = st.columns([1, 1, 4])
            with c1:
                if st.button("生成摘要", use_container_width=True):
                    summary = kb.agent.summarize_document(selected_doc.id, max_sentences=8)
                    st.session_state["library_summary"] = summary
            with c2:
                if st.button("删除文档", type="secondary", use_container_width=True):
                    kb.delete_document(selected_doc.id)
                    kb = reset_kb_cache()
                    st.success("已删除文档及其切片。")
                    st.rerun()
            if "library_summary" in st.session_state:
                st.markdown("#### 文档摘要")
                st.write(st.session_state["library_summary"])

with tab_search:
    st.subheader("关键词 + 语义检索")
    query = st.text_input("检索问题或关键词", placeholder="例如：transformer 在长文档检索中的作用")
    col1, col2, col3 = st.columns([1, 1, 2])
    mode = col1.selectbox("检索模式", ["hybrid", "keyword", "semantic"], index=0)
    top_k = col2.slider("返回数量", min_value=3, max_value=20, value=8)
    doc_filter = col3.multiselect(
        "限定文档",
        options=[doc.id for doc in kb.list_documents()],
        format_func=lambda doc_id: kb.store.get_document(doc_id).title,
    )

    if st.button("开始检索", type="primary") and query.strip():
        results = kb.search(query, top_k=top_k, mode=mode, document_ids=doc_filter or None)
        if not results:
            st.warning("没有检索到匹配内容。")
        for idx, result in enumerate(results, start=1):
            render_source(result.to_source_dict(), idx)

with tab_agent:
    st.subheader("Agent 问答")
    question = st.text_area(
        "向知识库提问",
        placeholder="例如：这些论文如何评价 RAG 在科学文献问答中的主要挑战？",
        height=120,
    )
    col1, col2 = st.columns([1, 4])
    agent_top_k = col1.slider("证据数量", min_value=3, max_value=15, value=6, key="agent_top_k")
    col2.caption("Agent 会规划检索、证据筛选、摘要生成和引用返回。")
    if st.button("让 Agent 回答", type="primary") and question.strip():
        with st.spinner("Agent 正在检索和整理证据"):
            response = kb.agent.ask(question, top_k=agent_top_k)
        st.markdown("#### 回答")
        st.write(response["answer"])
        st.markdown("#### Agent 计划")
        for step in response["plan"]:
            st.caption(f"{step['tool']}: {step['why']}")
        st.markdown("#### 来源")
        for idx, source in enumerate(response["sources"], start=1):
            render_source(source, idx)

with tab_summary:
    st.subheader("检索汇总")
    docs = kb.list_documents()
    if not docs:
        st.info("知识库还没有文档，请先上传 PDF。")
    else:
        target = st.selectbox(
            "汇总范围",
            options=["__all__"] + [doc.id for doc in docs],
            format_func=lambda value: "全部文档" if value == "__all__" else kb.store.get_document(value).title,
        )
        focus = st.text_input("可选：汇总关注点", placeholder="例如：方法、数据集、结论、局限性")
        summary_len = st.slider("摘要句数", min_value=4, max_value=16, value=8)
        if st.button("生成汇总", type="primary"):
            if target == "__all__":
                result = kb.agent.summarize_library(focus or None, max_sentences=summary_len)
            else:
                result = kb.agent.summarize_document(
                    target,
                    query=focus or None,
                    max_sentences=summary_len,
                )
            st.write(result)
