"""Streamlit web app — 造纸智能助手「小纸」（宣纸·水墨主题）

Dual-mode interface:
    - 🔍 知识问答 — simple RAG, one-shot retrieval + generation
    - 🤖 故障排查 Agent — ReAct loop with tool calling + visible thought chain

Usage::

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# ensure this project root is importable BEFORE any src.* imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Must be set BEFORE any HF imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.generation.rag_pipeline import PaperAgent
from src.agent.react_agent import PaperReActAgent
from src.ui.theme import THEME_CSS
from src.ui import components as ui

# ── 本地 SVG 资源（favicon 与气泡头像，离线可用）──
_ASSETS = Path(__file__).resolve().parent / "assets"
FAVICON = str(_ASSETS / "favicon.svg")
AVATAR_USER = str(_ASSETS / "avatar-user.svg")
AVATAR_ASSISTANT = str(_ASSETS / "avatar-assistant.svg")

# ═══════════════════════════════════════════════════════════════
#  Page configuration & theme
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="造纸智能助手 · 小纸",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入宣纸·水墨主题（设计令牌 + 全局样式）
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  Cached resource: retriever (shared across sessions)
# ═══════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def load_retriever():
    """Load the three-stage retrieval chain.  Cached globally."""
    from src.retrieval.factory import build_retriever
    return build_retriever()


# ═══════════════════════════════════════════════════════════════
#  Session state initialisation
# ═══════════════════════════════════════════════════════════════

# 进行各个状态的初始化（对话从空白开始；「小纸」介绍以静态卡片展示在主页面顶部）
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback: Dict[str, str] = {}  # msg_index → "up" | "down"

if "retriever" not in st.session_state:
    with st.spinner("正在加载知识库与检索模型 ..."):
        st.session_state.retriever, st.session_state.chunk_count = load_retriever()

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = PaperAgent(st.session_state.retriever, top_k=5)

if "react_agent" not in st.session_state:
    st.session_state.react_agent = PaperReActAgent(
        st.session_state.retriever,
        max_iterations=10,
        max_history=10,
        top_k=5,
        on_thought=None,  # will be set dynamically per-turn
    )

# ═══════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    ui.sidebar_header()
    ui.daily_line()

    st.divider()

    # ── mode selector ──
    st.subheader("工作模式")
    mode = st.radio(
        "选择模式",
        ["知识问答", "Agent 故障排查"],
        label_visibility="collapsed",
    )
    is_agent_mode = "Agent" in mode

    st.divider()

    # ── knowledge base stats（印章）──
    ui.side_label("知识库统计")
    chunk_count = st.session_state.get("chunk_count", 0)
    stage_count = 3 if is_agent_mode else 2
    answered = sum(
        1 for m in st.session_state.messages
        if m.get("role") == "assistant" and m.get("mode") != "system"
    )
    col1, col2 = st.columns(2)
    with col1:
        ui.render_stamp(chunk_count, "知识片段")
    with col2:
        ui.render_stamp(stage_count, "检索工序")
    ui.render_stamp(answered, "本卷已答")

    st.divider()

    # ── actions ──
    ui.side_label("常用操作")

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.session_state.rag_agent.clear_history()
        st.session_state.react_agent.clear_history()
        st.rerun()

    if st.button("重建知识索引", use_container_width=True, help="解析 data/raw/ 下所有 PDF 并重建知识库"):
        with st.spinner("正在重建知识索引 ..."):
            try:
                from src.retrieval.factory import rebuild_index
                from src.retrieval.reranker import RerankerProcessor

                new_hybrid = rebuild_index("data/raw")

                if new_hybrid:
                    reranker = RerankerProcessor(new_hybrid, candidate_pool=20)
                    st.session_state.retriever = reranker
                    st.session_state.rag_agent = PaperAgent(st.session_state.retriever, top_k=5)
                    st.session_state.react_agent = PaperReActAgent(
                        st.session_state.retriever, max_iterations=10, max_history=10, top_k=5
                    )
                    st.session_state.chunk_count = new_hybrid.collection.count()
                    st.success("索引重建完成，知识库已更新")
                else:
                    st.warning("未找到 PDF 文件，请将 PDF 放入 data/raw/ 目录")
                st.rerun()
            except Exception as exc:
                st.error(f"重建失败：{exc}")

    st.divider()

    # ── about ──
    with st.expander("关于"):
        st.markdown(
            """
            **造纸智能助手 · 小纸** 是一个基于 RAG + Agent 的
            垂直领域问答系统。

            **技术栈**
            - DeepSeek-V4 · BGE Embedding
            - Chroma · BM25 · CrossEncoder
            - LlamaIndex · Streamlit

            **能力范围**
            - 制浆、抄纸、施胶、涂布
            - 质量控制与故障排查

            知识库内容为模拟数据，仅供演示。
            """
        )

    ui.render_side_tagline()

# ═══════════════════════════════════════════════════════════════
#  Main chat area — brand banner & message history
# ═══════════════════════════════════════════════════════════════

ui.brand_banner()
ui.render_welcome()            # 「小纸」介绍 —— 主界面上方
ui.render_mode_intro(mode)     # 具体 Agent 模式介绍 —— 其下

# render message history
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    avatar = AVATAR_USER if role == "user" else AVATAR_ASSISTANT
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # source citations
        if msg.get("sources"):
            ui.render_sources(msg["sources"])

        # thought chain (agent mode)
        if msg.get("thoughts"):
            ui.render_thoughts(msg["thoughts"])

        # feedback for assistant messages
        if role == "assistant" and msg.get("mode") != "system":
            ui.render_feedback(i)

# ═══════════════════════════════════════════════════════════════
#  Chat input & processing
# ═══════════════════════════════════════════════════════════════

if prompt := st.chat_input(
    "输入你的造纸工艺问题，例如：纸张出现气泡怎么办？"
):
    # ── add user message ──
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": None, "thoughts": None, "mode": "user"}
    )
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    # ── generate response ──
    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        if is_agent_mode:
            # ── Agent mode with thought chain ──
            thoughts: List[Dict[str, str]] = []

            def collect_thought(step_type: str, content: str):
                thoughts.append({"type": step_type, "content": content})

            # inject the thought collector into the agent for this turn
            st.session_state.react_agent.on_thought = collect_thought

            with st.spinner("小纸正在思考并检索资料 ..."):
                try:
                    result = st.session_state.react_agent.chat(prompt)
                    answer = result.get("answer", "抱歉，处理过程中出现了问题。")
                    tool_calls = result.get("tool_calls", [])
                    iterations = result.get("iterations", 0)
                except Exception as exc:
                    answer = f"处理出错：{exc}\n\n请检查 API Key 配置或稍后重试。"
                    tool_calls = []
                    iterations = 0
                    thoughts.append({"type": "answer", "content": f"Error: {exc}"})

            thoughts.append({"type": "answer", "content": f"回答完成（{iterations} 轮推理，{len(tool_calls)} 次工具调用）"})

            st.markdown(answer)

            # sources from tool calls
            sources = [
                tc.get("result_preview", "")[:200]
                for tc in tool_calls
            ]

            ui.render_sources(sources)
            ui.render_thoughts(thoughts)

            # save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "thoughts": thoughts,
                "mode": "agent",
            })

        else:
            # ── Simple RAG mode ──
            with st.spinner("正在检索知识库 ..."):
                try:
                    result = st.session_state.rag_agent.chat(prompt)
                    answer = result.get("answer", "抱歉，处理过程中出现了问题。")
                    raw_sources = result.get("sources", [])
                    mode_label = result.get("mode", "rag")
                except Exception as exc:
                    answer = f"处理出错：{exc}\n\n请检查 API Key 配置或稍后重试。"
                    raw_sources = []
                    mode_label = "rag"

            st.markdown(answer)

            # format sources
            sources = [
                f"[来源{i+1}] {src[:200]}{'...' if len(src) > 200 else ''}"
                for i, src in enumerate(raw_sources)
            ] if raw_sources else []

            if mode_label == "chat":
                st.caption("闲聊模式（未检索知识库）")

            ui.render_sources(sources)

            # save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "thoughts": None,
                "mode": mode_label,
            })

        # feedback for the new message
        ui.render_feedback(len(st.session_state.messages) - 1)

# ═══════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════

ui.footer(chunk_count=st.session_state.get("chunk_count", 0))
