# 📜 造纸智能助手 · 小纸

> 基于 RAG + Agent 的造纸工艺垂直领域智能问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--V3-green.svg)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**「小纸」** 是一位 AI 造纸工艺专家，能够回答制浆、抄纸、施胶、涂布、质量控制等专业问题，并自主诊断常见故障，提供分步骤的解决方案。

**核心承诺**：回答 100% 可溯源，绝不编造信息。

---

## ✨ 特性

- 🔍 **双模式架构**：知识问答（快速查参数）+ 故障排查 Agent（多轮推理诊断）
- 🧠 **ReAct Agent**：自主选择工具 → 多轮检索 → 结构化输出（准备 → 步骤 → 避坑）
- 📚 **三阶段检索**：BM25 关键词 + BGE 语义向量 + CrossEncoder 精排
- 🎯 **100% 可溯源**：所有回答均标注引用来源，知识库外问题主动拒答
- 🐳 **Docker 一键部署**：无需配置 Python 环境，一条命令启动
- 📊 **完整评测体系**：RAG 检索评测 + Agent 回答质量评测双轨验证

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────┐
│              Streamlit 前端 (app.py)              │
│    ┌──────────────┐  ┌──────────────────────┐    │
│    │  知识问答模式  │  │   故障排查 Agent 模式  │    │
│    └──────┬───────┘  └──────────┬───────────┘    │
├───────────┼──────────────────────┼────────────────┤
│           ▼                      ▼                │
│  ┌────────────────┐  ┌──────────────────────┐    │
│  │  RAG Pipeline  │  │   ReAct Agent Loop   │    │
│  │                │  │  Thought→Action→Obs   │    │
│  │ HybridRetriever│  │                      │    │
│  │   ├─ BM25      │  │  Tool Calling:       │    │
│  │   └─ Semantic  │  │   ├─ search_knowledge │    │
│  │ Reranker       │  │   ├─ query_tool_list  │    │
│  │ LLM Generator  │  │   └─ check_mistakes   │    │
│  └───────┬────────┘  └──────────┬───────────┘    │
│          │                      │                 │
│          ▼                      ▼                 │
│  ┌───────────────────────────────────────────┐    │
│  │           Chroma 向量数据库                 │    │
│  │    BGE Embedding + BM25 倒排索引           │    │
│  └───────────────────────────────────────────┘    │
│          ▲                                        │
│  ┌───────┴────────┐  ┌──────────────────┐        │
│  │ 文档解析+分块   │  │  DeepSeek-V3 API │        │
│  └────────────────┘  └──────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 方式一：本地运行

```bash
# 1. 克隆项目
git clone https://github.com/ssmtkx/paper_agent.git && cd paper_agent

# 2. 安装依赖
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY=sk-xxxxx

# 4. 初始化知识库（首次运行自动创建模拟数据）
python scripts/seed_data.py

# 5. 启动 Web 应用
streamlit run app.py
# 打开 http://localhost:8501
```

### 方式二：Docker 部署

```bash
# 1. 配置 .env
echo "DEEPSEEK_API_KEY=sk-xxxxx" > .env

# 2. 一键启动
docker compose up -d

# 3. 打开浏览器
open http://localhost:8501
```

### 方式三：CLI 模式

```bash
# 简单 RAG 对话
python run.py

# ReAct Agent 对话（含思考链可视化）
python run_agent.py
```

---

## 📖 使用指南

### Web 界面

| 模式 | 适合场景 | 示例问题 |
|------|----------|----------|
| 🔍 知识问答 | 查参数、问原理 | 「AKD施胶的用量和pH是多少？」 |
| 🤖 Agent 诊断 | 故障排查、全流程 | 「纸张出现气泡怎么办？」 |

**Agent 模式下**，你可以看到完整的思考链（💭 思考 → 🔧 工具调用 → 📖 观察结果），了解 AI 的推理过程。

### CLI 命令

```
/help   查看帮助
/clear  清空对话记忆
/index  重建知识索引（放入新 PDF 后使用）
/quit   退出
```

### 接入真实数据

将造纸相关的 PDF 文件放入 `data/raw/` 目录，然后在 CLI 或 Web 侧边栏执行 `/index` 即可重建知识库索引。

---

## 📁 项目结构

```
paper_agent/
├── app.py                  # Streamlit Web 应用
├── run.py                  # CLI — 简单 RAG 模式
├── run_agent.py            # CLI — ReAct Agent 模式
├── eval_rag.py             # RAG 检索评测
├── eval_agent.py           # Agent 回答质量评测
│
├── src/
│   ├── ingestion/          # 文档摄入
│   │   ├── parser.py       #   PDF 解析（PyMuPDF + pdfplumber）
│   │   └── splitter.py     #   语义分块（按章节/段落）
│   ├── indexing/           # 索引构建
│   │   ├── vector_store.py #   Chroma + BGE Embedding
│   │   ├── bm25_index.py   #   BM25 倒排索引（jieba 分词）
│   │   └── indexer.py      #   一键建库入口
│   ├── retrieval/          # 检索增强
│   │   ├── hybrid_retriever.py  # BM25+语义融合（α=0.3）
│   │   └── reranker.py     #   CrossEncoder 精排 + 权威加权
│   ├── generation/         # 对话生成
│   │   ├── prompts.py      #   身份 Prompt + RAG 模板
│   │   └── rag_pipeline.py #   PaperAgent（意图分拣+记忆）
│   ├── agent/              # Agent 智能体
│   │   ├── tools.py        #   3 个检索工具（OpenAI function-calling）
│   │   └── react_agent.py  #   ReAct 循环 + 结构化输出
│   └── eval/               # 评测体系
│       └── evaluator.py    #   Hit/MRR/Recall/NDCG 指标
│
├── scripts/
│   └── seed_data.py        # 模拟知识库（14 条造纸知识）
├── data/
│   ├── raw/                # 放入 PDF 文件（gitignore）
│   ├── eval_qa.json        # RAG 评测集（50 QA 对）
│   └── agent_eval.json     # Agent 评测集（20 故障场景）
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 📊 评测结果

### RAG 检索评测（50 QA 对，`eval_rag.py`）

| 指标 | 全量 (50条) | 知识库内 (42条) |
|------|-------------|-----------------|
| **Hit Rate@5** | 84.0% | **100%** ✅ |
| **MRR@5** | 84.0% | **100%** |
| **Recall@5** | 99.2% | 99.0% |
| 未命中 | 8 条（全部为知识库外问题） | 0 条 |

> 里程碑：知识库内 Hit@5 ≥ 89% — **超额达成** ✅  
> 8 条未命中均为"知识库外"问题（钛白粉、COD、碳交易等），系统正确返回空结果。

### Agent 评测（20 故障场景，`eval_agent.py`）

| 维度 | 得分 | 说明 |
|------|------|------|
| Tool Selection | 93% | 工具选择准确率 |
| Source Citation | 100% | 来源标注覆盖率 |
| Structure | 80% | 三段式结构完整性 |
| Keyword Coverage | 70% | 关键信息覆盖率 |
| **平均总分** | **85.8 / 100** ✅ | 优秀 60%，良好 40% |

---

## 🛠 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek-V3 | 高性价比，OpenAI 兼容 API |
| **Embedding** | BAAI/bge-small-zh-v1.5 | 中文语义向量，本地 CPU 推理 |
| **Reranker** | BAAI/bge-reranker-v2-m3 | CrossEncoder 精排 |
| **向量库** | ChromaDB | 轻量级，嵌入式部署 |
| **关键词检索** | BM25 + jieba | 自实现 BM25（k1=1.5, b=0.75） |
| **文档解析** | PyMuPDF + pdfplumber | 双引擎，互为兜底 |
| **前端** | Streamlit | 宣纸·水墨国风主题（纸墨斋）|
| **容器化** | Docker + Compose | 一键部署 |
| **分词** | jieba | 中文分词 |
| **框架** | LlamaIndex | RAG 编排 |

---

## ⚙️ 环境变量

在 `.env` 文件中配置：

```bash
# 必填
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx

# 可选
DEEPSEEK_BASE_URL=https://api.deepseek.com
HF_ENDPOINT=https://hf-mirror.com    # 国内用户建议设置镜像
```

---

## 📝 后续规划

- [ ] 真实造纸 PDF 数据接入（当前为 14 条模拟数据）
- [ ] BM25 索引持久化到磁盘
- [ ] 支持更多 LLM（通义千问、GLM 等）
- [ ] 多模态支持（工艺流程图识别）
- [ ] 用户反馈闭环（👍👎 数据驱动优化）

---

## 📄 License

MIT © 2026
