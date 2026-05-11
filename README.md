# PaperAgent

PaperAgent 是一个本地运行的论文知识库工具，使用 Python 构建，包含 PDF 上传入库、知识库管理、关键词/语义混合检索、RAG 证据汇总和 Agent 问答流程。

## 最新增强功能

- **流式 Agent 响应**：LLM 回答现在支持实时流式输出，响应更迅速。
- **思考过程展示**：支持展示模型的 `reasoning_content`（思考链），让 AI 的逻辑透明可见。
- **Markdown 渲染**：问答结果通过 `marked.js` 进行专业排版，支持表格、公式、代码块等。
- **全新视觉主题**：采用“橘子海”渐变配色，界面现代、大气、美观。
- **对话式交互**：底部对话框设计，支持针对单篇论文的“精读追问”和针对全库的“全局检索问答”。
- **单篇与全局区分**：在 AI 精读面板可进行深度理解，在 Agent 问答面板可进行跨文献 RAG。
- **三维知识图谱**：知识图谱已从 SVG 平面布局升级为 Canvas 三维球面布局，支持自动自转、透视缩放、视角调整和平移。

## 功能

- 上传 PDF：抽取每页文本，按页切片，并记录标题、页码、文件哈希和元数据。
- 知识库管理：查看文档、去重、删除文档、重建索引，并预览原始 PDF。
- 高级切片：章节/段落感知长上下文块，保留页码范围和章节名。
- 混合检索：BM25 + 关键词命中 + 字符 n-gram 语义相似度 + MMR 去重，支持限定文档。
- 知识图谱：抽取文档关键词概念，以三维球面网络展示文档-概念-文档的关联关系。
- RAG 汇总：基于检索证据抽取摘要句，并保留来源页码和原文跳转。
- Agent 问答：按“高级检索 -> 流式 LLM 回答 -> 来源引用 -> PDF 回看”的流程回答问题。

## 三维知识图谱交互

知识图谱页面使用原生 Canvas 渲染，不依赖额外前端构建工具。当前实现包含：

- 三维球面布局：文档节点位于内层，概念节点位于外层，边按文档-概念引用和文档-文档共享概念连接。
- 地球自转式动画：图谱围绕纵轴缓慢自转，形成稳定的空间感。
- 正向文字标签：节点文字始终以屏幕坐标绘制，不随球体翻转，因此不会出现倒置文字。
- 滚轮缩放：鼠标滚轮调整透视距离，放大或缩小整体图谱。
- 右键视角：按住鼠标右键拖动可改变观察方向。
- 中键平移：按住鼠标中键拖动可平移图谱视口。
- 深度表现：节点和连线按深度排序绘制，远处元素透明度更低，近处元素更清晰。

## 快速开始

推荐使用新的原生 Web 界面，不依赖 Streamlit 和 pyarrow：

```powershell
python web_app.py --host localhost --port 8601
```

浏览器打开：

```text
http://localhost:8601
```

## 可选 LLM 接入

默认不需要外部服务，Agent 使用本地抽取式回答。若要启用生成式回答，设置 OpenAI-compatible Chat Completions 环境变量：

```powershell
$env:PAPERAGENT_LLM_API_KEY="你的 API Key"
$env:PAPERAGENT_LLM_MODEL="你的模型名"
$env:PAPERAGENT_LLM_BASE_URL="https://api.openai.com/v1/chat/completions"
$env:PAPERAGENT_ALLOW_EXTERNAL_LLM="true"
python web_app.py --host localhost --port 8601
```

`PAPERAGENT_LLM_BASE_URL` 可换成其他兼容服务地址。
`PAPERAGENT_ALLOW_EXTERNAL_LLM=true` 表示允许把检索到的论文证据块发送给外部 LLM，用于生成精读卡片和 Agent 回答。

如果仍想使用 Streamlit 原型界面：

```powershell
python -m pip install streamlit
streamlit run app.py
```

本项目也支持把依赖安装到本地 `.packages` 目录：

```powershell
python -m pip install --target .\.packages -r requirements.txt
streamlit run app.py
```

## 项目结构

```text
paperagent/
  agent.py           Agent 编排：检索、汇总、引用
  chunking.py        PDF 页面文本切片
  knowledge_base.py  入库、删除、检索统一入口
  knowledge_graph.py 文档-概念知识图谱
  llm.py             可选 LLM 生成式回答
  pdf_loader.py      PDF 文本抽取
  retriever.py       BM25 + 语义 + MMR 混合检索
  store.py           JSON 持久化知识库
  summarizer.py      抽取式汇总
app.py               Streamlit 用户界面
web_app.py           原生 Web 用户界面和 API
scripts/smoke_test.py
```

## 验证

```powershell
python scripts/smoke_test.py
python -m compileall app.py paperagent scripts
```

## 后续增强方向

- 接入 `sentence-transformers`、FAISS 或 Chroma，替换默认的字符 n-gram 语义通道。
- 接入 OpenAI 或本地大模型，让 Agent 生成更自然的综合回答。
- 为扫描版 PDF 增加 OCR 流程。
- 增加标签、作者、年份和主题聚类等知识库字段。
