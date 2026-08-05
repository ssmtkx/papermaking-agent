"""「小纸」UI 组件 —— 宣纸水墨主题的 HTML 渲染块。

所有组件基于 Streamlit ≥1.28（用 ``st.markdown(..., unsafe_allow_html=True)``
渲染，不依赖更高版本的 ``st.html``）。仅负责渲染，不携带业务状态。
"""

from __future__ import annotations

import datetime
import html
from typing import Dict, List

import streamlit as st

from src.ui.icons import ICON_PULSE, ICON_SEARCH

# ═══════════════════════════════════════════════════════════
# 品牌
# ═══════════════════════════════════════════════════════════


def brand_banner() -> None:
    """主区域顶部的品牌横幅：朱砂方印 + 楷体题签 + 副题。"""
    st.markdown(
        f"""
        <div class="xz-banner">
          <div class="xz-seal">小纸</div>
          <div>
            <div class="xz-banner-title">造纸智能助手 · 小纸</div>
            <div class="xz-banner-sub">{ICON_SEARCH} 知识问答　｜　{ICON_PULSE} Agent 故障排查</div>
            <div class="xz-banner-tag">造纸工艺智能问答 · 回答 100% 可溯源</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_header() -> None:
    """侧栏头部：小方印 + 品牌名 + 一句说明。"""
    st.markdown(
        """
        <div class="xz-side-seal">纸</div>
        <div class="xz-side-title">小纸</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("造纸工艺智能助手")


def side_label(text: str) -> None:
    """侧栏分区小标签，如「知识库统计」「常用操作」。"""
    st.markdown(f'<div class="xz-side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 生活气息小件
# ═══════════════════════════════════════════════════════════

# 周一 ~ 周日的今日提示
_DAILY_LINES = {
    0: "研究一下制浆和抄纸的工艺参数吧",
    1: "检查检查纸张的湿度与平整度",
    2: "排查设备故障，思路理清了就不难",
    3: "翻翻旧知识，补充点新工艺",
    4: "核对一下施胶与涂布的配方",
    5: "整理整理知识库和笔记",
    6: "休息一下，喝杯茶再继续",
}


def daily_line() -> None:
    """按星期展示一句今日提示，添一点生活气。"""
    line = _DAILY_LINES[datetime.date.today().weekday()]
    st.markdown(f'<div class="xz-daily">今日 · {html.escape(line)}</div>', unsafe_allow_html=True)


def render_divider(text: str | None = None) -> None:
    """鎏金回纹分隔线；传 text 则居中显示（如「欢迎提问」）。"""
    inner = f"<span>{html.escape(text)}</span>" if text else ""
    st.markdown(f'<div class="xz-divider">{inner}</div>', unsafe_allow_html=True)


def render_welcome() -> None:
    """主页面顶部的「小纸」介绍卡（原先的第一条欢迎消息）。"""
    st.markdown(
        """
        <div class="xz-welcome">
          <div class="xz-welcome-title">认识一下，我是小纸</div>
          <div class="xz-welcome-desc">
            造纸工艺智能助手，帮你解答制浆、抄纸、施胶、涂布等专业问题，
            也能一步步排查常见工艺故障。回答均来自造纸知识库，100% 可溯源，不编造。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 两种模式的介绍文案与示例问题
_MODE_INTRO = {
    "qa": {
        "title": "知识问答",
        "icon": ICON_SEARCH,
        "desc": "快速查询工艺参数、原理与流程，适合明确的、单点的知识性问题。",
        "examples": ["AKD施胶的用量和 pH 是多少？", "打浆游离度对纸张强度有什么影响？"],
        "tone": "var(--xz-cinnabar)",
    },
    "agent": {
        "title": "Agent 故障排查",
        "icon": ICON_PULSE,
        "desc": "多步推理诊断复杂故障：自主检索知识、列出设备清单、给出分步骤的排查与避坑建议。",
        "examples": ["纸张出现气泡怎么排查？", "纸机干燥部卷曲起拱怎么处理？"],
        "tone": "var(--xz-indigo)",
    },
}


def render_mode_intro(mode: str) -> None:
    """根据当前模式，在主页横幅下渲染一段模式介绍与示例问题。"""
    info = _MODE_INTRO["agent"] if "Agent" in mode else _MODE_INTRO["qa"]
    chips = "".join(
        f'<span class="xz-chip">{html.escape(e)}</span>' for e in info["examples"]
    )
    st.markdown(
        f"""
        <div class="xz-mode-intro" style="border-left-color:{info['tone']};">
          <span style="color:{info['tone']};">{info['icon']}</span>
          <div>
            <div class="xz-mode-intro-title">当前模式 · {html.escape(info['title'])}</div>
            <div class="xz-mode-intro-desc">{html.escape(info['desc'])}</div>
            <div>{chips}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_side_tagline() -> None:
    """侧栏底部一行小标语。"""
    st.markdown(
        """
        <div class="xz-side-tagline">
          制浆 · 抄纸 · 施胶 · 涂布<br/>
          造纸工艺问题，有问必答
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# 印章统计
# ═══════════════════════════════════════════════════════════


def render_stamp(value, label: str) -> None:
    """印章统计盒：主区朱砂色，侧栏自动切换鎏金色。"""
    st.markdown(
        f'<div class="xz-stamp"><div class="value">{value}</div>'
        f'<div class="label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# 消息内容
# ═══════════════════════════════════════════════════════════


def render_sources(sources: List[str] | None) -> None:
    """引用来源：宣纸卡片 + 朱砂「引」标目。"""
    if not sources:
        return
    with st.expander(f"引用来源（{len(sources)} 条）", expanded=False):
        for src in sources:
            st.markdown(f'<div class="xz-source">{html.escape(src)}</div>', unsafe_allow_html=True)


# 思考链各步骤的标签
_THOUGHT_META = {
    "thought": ("思考", "xz-thought-thought"),
    "action": ("行动", "xz-thought-action"),
    "observation": ("观察", "xz-thought-observation"),
    "answer": ("完成", "xz-thought-answer"),
}
_NUMS = "①②③④⑤⑥⑦⑧⑨⑩"


def render_thoughts(thoughts: List[Dict[str, str]] | None) -> None:
    """思考过程：思考(黛蓝) / 行动(朱砂) / 观察(竹青) / 完成(鎏金)。"""
    if not thoughts:
        return
    with st.expander(f"思考过程（{len(thoughts)} 步）", expanded=True):
        for i, step in enumerate(thoughts, 1):
            step_type = step.get("type", "thought")
            content = step.get("content", "")
            name, cls = _THOUGHT_META.get(step_type, _THOUGHT_META["thought"])
            num = _NUMS[i - 1] if i <= len(_NUMS) else str(i)
            st.markdown(
                f'<div class="xz-thought {cls}">'
                f'<span class="xz-tnum">{num}{name}</span>{html.escape(content)}'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_feedback(msg_index: int) -> None:
    """有帮助 / 需改进 反馈按钮 + 已选提示（逻辑与原实现一致）。"""
    key_up = f"fb_up_{msg_index}"
    key_down = f"fb_down_{msg_index}"
    current = st.session_state.feedback.get(str(msg_index))

    c1, c2, c3 = st.columns([1, 1, 14])
    with c1:
        if st.button("有帮助", key=key_up, help="回答是否有帮助", disabled=(current == "up"),
                     use_container_width=True):
            st.session_state.feedback[str(msg_index)] = "up"
            st.rerun()
    with c2:
        if st.button("需改进", key=key_down, help="回答需要改进", disabled=(current == "down"),
                     use_container_width=True):
            st.session_state.feedback[str(msg_index)] = "down"
            st.rerun()
    if current:
        with c3:
            label = "已反馈：有帮助" if current == "up" else "已反馈：需改进"
            st.caption(label)


# ═══════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════


def footer(chunk_count: int = 0) -> None:
    """页脚：溯源/免责说明 + 知识库片段数。"""
    st.markdown(
        f"""
        <div class="xz-footer">
          本助手由造纸专业知识库驱动 · 回答 100% 可溯源<br/>
          生产决策请以实际工艺手册与设备厂商指导为准<br/>
          <span style="color:var(--xz-ink-muted);">知识库片段 {chunk_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
