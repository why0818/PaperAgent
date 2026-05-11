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

## 克隆后快速开始

从 GitHub 克隆后，建议使用 Python 3.10+：

```powershell
git clone <你的仓库地址>
cd PaperAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
git clone <你的仓库地址>
cd PaperAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

推荐使用新的原生 Web 界面，不依赖 Streamlit 和 pyarrow：

```powershell
python web_app.py --host localhost --port 8601
```

浏览器打开：

```text
http://localhost:8601
```

首次启动会自动创建本地 `data/` 目录。上传的 PDF、索引、分析结果、聊天历史、期刊会议库和本地配置都保存在 `data/` 下；这些属于个人数据，默认不应该提交到 GitHub。

## 可选 LLM 接入

默认不需要外部服务，系统可以使用本地抽取式回答和本地预览能力。但如果希望启用更完整的 AI 精读、生成式 Agent 问答、段落归纳和流式回答，需要配置 OpenAI-compatible Chat Completions API。

### 最小配置

PowerShell：

```powershell
$env:PAPERAGENT_LLM_API_KEY="你的 API Key"
$env:PAPERAGENT_ALLOW_EXTERNAL_LLM="true"
python web_app.py --host localhost --port 8601
```

Bash：

```bash
export PAPERAGENT_LLM_API_KEY="你的 API Key"
export PAPERAGENT_ALLOW_EXTERNAL_LLM="true"
python web_app.py --host localhost --port 8601
```

默认接口和模型在 [paperagent/llm.py](paperagent/llm.py) 中定义：

- 默认 API 地址：`https://api.siliconflow.cn/v1/chat/completions`
- 默认问答模型：`Qwen/Qwen3-235B-A22B-Instruct-2507`
- 默认精读模型：`deepseek-ai/DeepSeek-V3.1-Terminus`

### 完整环境变量

```powershell
$env:PAPERAGENT_LLM_API_KEY="你的 API Key"
$env:PAPERAGENT_LLM_BASE_URL="https://api.siliconflow.cn/v1/chat/completions"
$env:PAPERAGENT_CHAT_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
$env:PAPERAGENT_ANALYZE_MODEL="deepseek-ai/DeepSeek-V3.1-Terminus"
$env:PAPERAGENT_ALLOW_EXTERNAL_LLM="true"
python web_app.py --host localhost --port 8601
```

变量说明：

- `PAPERAGENT_LLM_API_KEY`：必填，外部 LLM 服务的 API Key。
- `PAPERAGENT_LLM_BASE_URL`：可选，OpenAI-compatible 的 `/chat/completions` 完整地址。
- `PAPERAGENT_CHAT_MODEL`：可选，Agent 问答使用的模型。
- `PAPERAGENT_ANALYZE_MODEL`：可选，AI 精读和论文分析使用的模型。
- `PAPERAGENT_ALLOW_EXTERNAL_LLM=true`：允许把检索到的论文证据片段发送给外部 LLM。不开启时，即使配置了 API Key，也不会把论文内容发出去，系统会尽量使用本地抽取式能力。

当前前端模型下拉框支持以下模型名：

```text
Qwen/Qwen3-235B-A22B-Instruct-2507
deepseek-ai/DeepSeek-V3.1-Terminus
moonshotai/Kimi-K2-Instruct-0905
```

如果要接入其他 OpenAI-compatible 服务，可以改 `PAPERAGENT_LLM_BASE_URL`，并在 [paperagent/llm.py](paperagent/llm.py) 的 `SUPPORTED_MODELS` 中加入对应模型名。

### 本地配置文件方式

也可以创建 `data/config.local.json` 保存配置。该文件包含密钥，已经被 `.gitignore` 忽略，请不要提交。

```json
{
  "api_key": "你的 API Key",
  "base_url": "https://api.siliconflow.cn/v1/chat/completions",
  "chat_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "analyze_model": "deepseek-ai/DeepSeek-V3.1-Terminus",
  "allow_external_paper_content": true
}
```

环境变量优先级高于 `data/config.local.json`。

### 前端资源说明

Web 页面通过 CDN 加载 `marked.js` 和 `KaTeX`，用于 Markdown 和公式渲染。首次使用时浏览器需要能访问这些 CDN；如果要完全离线部署，需要把这些前端资源下载到本地并修改 [web_app.py](web_app.py) 中的引用。

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
  metadata_store.py  论文可编辑元数据和期刊会议库
  pdf_loader.py      PDF 文本抽取
  preview.py         原文预览清洗、去参考文献和段落大意
  retriever.py       BM25 + 语义 + MMR 混合检索
  store.py           JSON 持久化知识库
  summarizer.py      抽取式汇总
app.py               Streamlit 用户界面
web_app.py           原生 Web 用户界面和 API
scripts/smoke_test.py
```

## 本地数据与 Git 提交

以下内容是用户本地数据，通常不应提交：

```text
data/uploads/          上传的 PDF
data/library.json      本地知识库索引
data/analyses/         AI 精读结果和可编辑元数据
data/chats/            对话历史
data/config.local.json API Key 和本地模型配置
data/venues.json       期刊/会议来源库
```

如果你希望多人共享同一批论文数据，可以单独备份或同步 `data/` 目录；如果只是发布代码，请保持这些文件不入库。

## 数据包导出与同步

推荐使用内置数据包功能把本地知识库同步给别人：

1. 打开 `http://localhost:8601`。
2. 点击右上角齿轮按钮进入设置。
3. 在“本地数据同步”里点击“导出数据包”。
4. 把下载得到的 `paperagent-data-*.zip` 发给对方。
5. 对方 clone 项目并启动后，在同一位置点击“导入数据包”，选择该 zip。

数据包会包含：

```text
library.json          文档索引和切片索引
uploads/              原始 PDF
analyses/             AI 精读结果和可编辑论文元数据
chats/                对话历史
venues.json           期刊/会议来源库
```

数据包不会包含：

```text
config.local.json     API Key、本地模型配置和隐私开关
*.tmp                 临时文件
```

导入时会把数据包里的论文合并到当前本地知识库，并自动把 PDF 的 `stored_path` 改写为导入者本机的 `data/uploads/` 路径。已有相同 `document_id` 的论文、分析和聊天记录会被数据包中的版本覆盖。

如果只是临时迁移到另一台电脑，也可以手动复制整个 `data/` 目录；但对外分享时更推荐数据包导出，因为它会自动排除 API Key。

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
