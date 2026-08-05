"""「小纸」宣纸·水墨主题 —— 设计令牌与全局样式。

只负责前端观感，不含任何业务逻辑。所有样式以 Streamlit ≥1.28 的
稳定 DOM 选择器为锚点，并附 `.xz-*` 自定义类供 components 使用。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# 调色板（宣纸·水墨）
# ═══════════════════════════════════════════════════════════════
PALETTE = {
    "paper": "#F6F0E4",         # 宣纸米黄 —— 应用底色
    "paper_deep": "#EDE3CD",    # 深米色 —— 次级背景
    "paper_card": "#FBF7EC",    # 亮米 —— 卡片 / 气泡
    "cinnabar": "#A63A2B",      # 朱砂印红 —— 主色
    "cinnabar_dark": "#8F3124",  # 深朱砂 —— hover / 按下
    "ink": "#2F2A25",           # 浓墨 —— 标题
    "ink_text": "#3E362E",      # 正文墨
    "ink_muted": "#8A7B66",     # 淡墨 —— 弱化文字 / 占位符
    "indigo": "#3B5568",        # 黛蓝 —— 工序「思」
    "bamboo": "#5B7F5E",        # 竹青 —— 工序「观」
    "gold": "#C9A227",          # 鎏金 —— 分隔线 / 点缀
    "sidebar": "#352A21",       # 墨褐 —— 侧栏面板底
    "sidebar_text": "#EFE4CE",  # 侧栏纸色文字
    "sidebar_muted": "#C9B48A",
    "line": "#D9C9A8",          # 信笺淡格线
    "red_line": "#B03A2E",      # 信笺红标
}

# 字体栈（离线安全，无外部字体依赖 —— 国内环境不引 Google Fonts）
FONT_SERIF = '"Source Han Serif SC","Noto Serif SC","Songti SC","STSong","SimSun","宋体",serif'
FONT_KAI = '"Kaiti SC","STKaiti","KaiTi","楷体","BiauKai",serif'

# ── 宣纸纸纹：极淡的 SVG 噪点（data-URI，纯本地）────────────
_PAPER_TEXTURE_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'>"
    "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' "
    "numOctaves='2' stitchTiles='stitch'/>"
    "<feColorMatrix type='matrix' values='0 0 0 0 0.62  0 0 0 0 0.55  "
    "0 0 0 0 0.47  0 0 0 0.045 0'/>"
    "</filter><rect width='100%' height='100%' filter='url(%23n)'/>"
    "</svg>"
)

_ROOT = f"""
<style>
:root {{
  --xz-paper: {PALETTE['paper']};
  --xz-paper-deep: {PALETTE['paper_deep']};
  --xz-paper-card: {PALETTE['paper_card']};
  --xz-cinnabar: {PALETTE['cinnabar']};
  --xz-cinnabar-dark: {PALETTE['cinnabar_dark']};
  --xz-ink: {PALETTE['ink']};
  --xz-ink-text: {PALETTE['ink_text']};
  --xz-ink-muted: {PALETTE['ink_muted']};
  --xz-indigo: {PALETTE['indigo']};
  --xz-bamboo: {PALETTE['bamboo']};
  --xz-gold: {PALETTE['gold']};
  --xz-sidebar: {PALETTE['sidebar']};
  --xz-sidebar-text: {PALETTE['sidebar_text']};
  --xz-sidebar-muted: {PALETTE['sidebar_muted']};
  --xz-line: {PALETTE['line']};
  --xz-red-line: {PALETTE['red_line']};
  --xz-font-serif: {FONT_SERIF};
  --xz-font-kai: {FONT_KAI};
  --xz-paper-texture: url("{_PAPER_TEXTURE_SVG}");
}}
</style>
"""

_BODY = """
<style>
/* ═══════════════════════════════════════════════════════════
   小纸 · 宣纸水墨 —— 全局样式
   ═══════════════════════════════════════════════════════════ */

/* ── 基础字体 ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--xz-font-serif);
    color: var(--xz-ink-text);
}

/* ── 宣纸底：噪点纸纹 + 暖色墨晕 ── */
[data-testid="stAppViewContainer"] {
    background-color: var(--xz-paper);
    background-image:
        var(--xz-paper-texture),
        radial-gradient(1200px 800px at 12% -8%, rgba(201,162,39,0.07), transparent 60%),
        radial-gradient(1000px 700px at 100% 0%, rgba(166,58,43,0.05), transparent 55%);
    background-attachment: fixed;
}

/* ── 主内容列：居中收窄，纸面更舒服 ── */
.block-container {
    max-width: 1000px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}

/* ── 隐藏 Streamlit 原生菜单，顶栏透明 ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
    box-shadow: none;
}

/* ── 标题排版 ── */
h1, h2, h3, h4 {
    font-family: var(--xz-font-serif);
    color: var(--xz-ink);
    letter-spacing: 0.04em;
}
[data-testid="stMarkdown"] p { line-height: 1.75; }
[data-testid="stCaptionContainer"] { color: var(--xz-ink-muted); }
[data-testid="stMarkdown"] a { color: var(--xz-cinnabar); }
::selection { background: rgba(166, 58, 43, 0.20); }

/* ── 表格（欢迎页模式表）── */
[data-testid="stMarkdown"] table { border-collapse: collapse; width: 100%; }
[data-testid="stMarkdown"] th {
    background: rgba(166, 58, 43, 0.08);
    color: var(--xz-ink);
    font-weight: 700;
}
[data-testid="stMarkdown"] th,
[data-testid="stMarkdown"] td {
    border: 1px solid var(--xz-line);
    padding: 6px 12px;
}

/* ═══════════════════════════════════════════════════════════
   侧栏 —— 墨褐面板
   ═══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: var(--xz-sidebar);
    background-image:
        var(--xz-paper-texture),
        linear-gradient(180deg, rgba(201,162,39,0.05), transparent 35%);
}
[data-testid="stSidebarContent"] { background: transparent; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #F2E6CE;
    font-family: var(--xz-font-serif);
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] li { color: var(--xz-sidebar-text); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--xz-sidebar-muted); }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: var(--xz-sidebar-text); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p { color: #FFF3D6; }

/* ═══════════════════════════════════════════════════════════
   按钮 —— 朱砂 / 纸感幽灵
   ═══════════════════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button,
[data-testid="stBaseButton-secondary"] {
    border-radius: 9px;
    border: 1px solid var(--xz-cinnabar);
    background: var(--xz-paper-card);
    color: var(--xz-cinnabar);
    font-family: var(--xz-font-serif);
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all .15s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--xz-cinnabar-dark);
    background: rgba(166, 58, 43, 0.08);
    color: var(--xz-cinnabar-dark);
}
.stButton > button:active,
.stDownloadButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--xz-cinnabar);
    color: #FBF7EC;
    border-color: var(--xz-cinnabar);
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--xz-cinnabar-dark);
    color: #FFF6E8;
}

/* ═══════════════════════════════════════════════════════════
   聊天气泡 —— 助手=信笺纸卡，用户=黛墨深底
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background:
        repeating-linear-gradient(0deg, rgba(217,201,168,0.18) 0 1px, transparent 1px 27px),
        linear-gradient(90deg, rgba(176,58,46,0.05), transparent 30%),
        var(--xz-paper-card);
    border: 1px solid var(--xz-line);
    border-left: 4px solid var(--xz-red-line);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(47, 42, 37, 0.07);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(120deg, var(--xz-indigo), #2A414F);
    border-left: 4px solid var(--xz-gold);
    color: #F4EAD6;
}
[data-testid="stChatMessage"] .stChatMessageContent { font-size: 0.95rem; }

/* 头像 —— 印章感方印（本地 SVG 图片） */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: transparent;
    border-radius: 7px;
    overflow: hidden;
    width: 28px;
    height: 28px;
}
[data-testid="chatAvatarIcon-user"] img,
[data-testid="chatAvatarIcon-assistant"] img {
    width: 100%;
    height: 100%;
    display: block;
}

/* ═══════════════════════════════════════════════════════════
   输入框
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: var(--xz-paper-card);
    border: 1px solid var(--xz-line);
    border-radius: 12px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--xz-cinnabar);
    box-shadow: 0 0 0 2px rgba(166, 58, 43, 0.14);
}
[data-testid="stChatInput"] textarea { color: var(--xz-ink-text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--xz-ink-muted); }

/* ═══════════════════════════════════════════════════════════
   折叠（引用 / 工序）
   ═══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--xz-line);
    border-radius: 10px;
    background: rgba(251, 247, 236, 0.55);
}
.streamlit-expanderHeader {
    font-family: var(--xz-font-serif);
    color: var(--xz-ink);
    font-weight: 700;
    letter-spacing: 0.04em;
}

/* ═══════════════════════════════════════════════════════════
   .xz-* 自定义组件
   ═══════════════════════════════════════════════════════════ */

/* ── 品牌横幅 ── */
.xz-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 20px;
    margin-bottom: 8px;
    background:
        radial-gradient(340px 200px at 88% 120%, rgba(166,58,43,0.06), transparent 70%),
        linear-gradient(120deg, rgba(166,58,43,0.07), rgba(201,162,39,0.05) 55%, transparent);
    border: 1px solid var(--xz-line);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
}
.xz-seal {
    flex: 0 0 auto;
    width: 54px; height: 54px;
    display: flex; align-items: center; justify-content: center;
    background: var(--xz-cinnabar);
    color: #FBF7EC;
    font-family: var(--xz-font-kai);
    font-size: 27px; font-weight: 700;
    border-radius: 9px;
    box-shadow: 0 0 0 2px var(--xz-paper-card), 0 0 0 3.5px var(--xz-cinnabar);
}
.xz-banner-title {
    font-family: var(--xz-font-kai);
    font-size: 1.65rem;
    color: var(--xz-ink);
    letter-spacing: 0.08em;
    line-height: 1.25;
}
.xz-banner-sub {
    font-size: 0.92rem;
    color: var(--xz-cinnabar);
    letter-spacing: 0.06em;
    margin-top: 2px;
}
.xz-banner-tag {
    font-size: 0.78rem;
    color: var(--xz-ink-muted);
    letter-spacing: 0.12em;
    margin-top: 2px;
}

/* ── 内联 SVG 图标 ── */
.xz-ic {
    display: inline-block;
    width: 1em;
    height: 1em;
    vertical-align: -0.18em;
    margin-right: 0.18em;
}

/* ── 小纸介绍卡 ── */
.xz-welcome {
    padding: 12px 16px;
    margin-bottom: 8px;
    background:
        linear-gradient(120deg, rgba(47,42,37,0.04), rgba(201,162,39,0.05) 60%, transparent);
    border: 1px solid var(--xz-line);
    border-left: 3px solid var(--xz-ink);
    border-radius: 10px;
}
.xz-welcome-title {
    font-family: var(--xz-font-kai);
    font-size: 1.05rem;
    color: var(--xz-ink);
    letter-spacing: 0.06em;
}
.xz-welcome-desc {
    font-size: 0.85rem;
    color: var(--xz-ink-text);
    margin-top: 2px;
    line-height: 1.7;
}

/* ── 当前模式介绍卡 ── */
.xz-mode-intro {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 12px 16px;
    margin-bottom: 10px;
    background:
        linear-gradient(120deg, rgba(166,58,43,0.06), rgba(59,85,104,0.05) 60%, transparent);
    border: 1px solid var(--xz-line);
    border-left: 3px solid var(--xz-cinnabar);
    border-radius: 10px;
}
.xz-mode-intro > .xz-ic {
    flex: 0 0 auto;
    margin-top: 4px;
    width: 1.1em;
    height: 1.1em;
}
.xz-mode-intro-title {
    font-family: var(--xz-font-kai);
    font-size: 1.05rem;
    color: var(--xz-ink);
    letter-spacing: 0.06em;
}
.xz-mode-intro-desc {
    font-size: 0.85rem;
    color: var(--xz-ink-text);
    margin-top: 2px;
    line-height: 1.7;
}
.xz-chip {
    display: inline-block;
    font-size: 0.78rem;
    color: var(--xz-cinnabar);
    background: rgba(166, 58, 43, 0.06);
    border: 1px solid rgba(166, 58, 43, 0.25);
    border-radius: 20px;
    padding: 2px 11px;
    margin: 7px 8px 0 0;
    white-space: nowrap;
}

/* ── 印章统计盒 ── */
.xz-stamp {
    text-align: center;
    padding: 9px 6px 7px;
    background: transparent;
    border: 2px solid var(--xz-cinnabar);
    border-radius: 10px;
    box-shadow: 0 0 0 2px var(--xz-paper-deep), inset 0 0 0 1px var(--xz-cinnabar);
    color: var(--xz-cinnabar);
    margin: 2px 0;
}
.xz-stamp .value {
    font-family: var(--xz-font-kai);
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.15;
}
.xz-stamp .label {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    margin-top: 2px;
}
/* 侧栏内印章换鎏金色 */
[data-testid="stSidebar"] .xz-stamp {
    border-color: var(--xz-gold);
    color: var(--xz-gold);
    box-shadow: 0 0 0 2px var(--xz-sidebar), inset 0 0 0 1px var(--xz-gold);
}
[data-testid="stSidebar"] .xz-stamp .label { color: var(--xz-sidebar-muted); }

/* ── 侧栏标题 / 分区标签 / 小方印 ── */
.xz-side-title {
    font-family: var(--xz-font-kai);
    font-size: 1.45rem;
    color: #F2E6CE;
    text-align: center;
    letter-spacing: 0.2em;
    margin: 2px 0 0;
}
.xz-side-seal {
    width: 42px; height: 42px;
    margin: 0 auto 8px;
    display: flex; align-items: center; justify-content: center;
    background: var(--xz-cinnabar);
    color: #FBF7EC;
    font-family: var(--xz-font-kai);
    font-size: 21px; font-weight: 700;
    border-radius: 8px;
    box-shadow: 0 0 0 2px var(--xz-sidebar), 0 0 0 3.5px var(--xz-cinnabar);
}
.xz-side-label {
    font-family: var(--xz-font-serif);
    font-size: 0.78rem;
    color: var(--xz-sidebar-muted);
    letter-spacing: 0.22em;
    margin: 12px 0 5px;
    border-left: 3px solid var(--xz-gold);
    padding-left: 8px;
}

/* ── 每日一句 ── */
.xz-daily {
    font-family: var(--xz-font-kai);
    font-size: 0.85rem;
    color: #E5C97B;
    text-align: center;
    padding: 9px 6px;
    margin: 10px 0 2px;
    border-top: 1px dashed rgba(229,201,123,0.35);
    border-bottom: 1px dashed rgba(229,201,123,0.35);
    letter-spacing: 0.06em;
}

/* ── 鎏金分隔线 ── */
.xz-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 8px;
    color: var(--xz-ink-muted);
    font-size: 0.85rem;
    letter-spacing: 0.3em;
}
.xz-divider::before,
.xz-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--xz-gold), transparent);
}

/* ── 引用来源（来源卡片）── */
.xz-source {
    position: relative;
    background:
        linear-gradient(90deg, rgba(176,58,46,0.05), transparent 45%),
        var(--xz-paper-card);
    border: 1px solid var(--xz-line);
    border-left: 3px solid var(--xz-cinnabar);
    border-radius: 8px;
    padding: 8px 12px 8px 42px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: var(--xz-ink-text);
    line-height: 1.6;
}
.xz-source::before {
    content: "引";
    position: absolute;
    left: 9px; top: 50%;
    transform: translateY(-50%);
    font-family: var(--xz-font-kai);
    color: var(--xz-cinnabar);
    border: 1px solid var(--xz-cinnabar);
    border-radius: 4px;
    padding: 1px 4px;
    font-size: 0.7rem;
}

/* ── 思考过程（思考链）── */
.xz-thought {
    padding: 7px 12px;
    margin: 6px 0;
    border-radius: 8px;
    font-size: 0.85rem;
    color: var(--xz-ink-text);
    line-height: 1.6;
}
.xz-thought .xz-tnum {
    font-family: var(--xz-font-kai);
    font-weight: 700;
    margin-right: 8px;
    white-space: nowrap;
}
.xz-thought-thought     { background: rgba(59,85,104,0.09);  border-left: 3px solid var(--xz-indigo);  color: #2E4352; }
.xz-thought-action      { background: rgba(166,58,43,0.07);  border-left: 3px solid var(--xz-cinnabar); color: #7C2A1F; }
.xz-thought-observation { background: rgba(91,127,94,0.10);  border-left: 3px solid var(--xz-bamboo);   color: #3E5B41; }
.xz-thought-answer      { background: rgba(201,162,39,0.11); border-left: 3px solid var(--xz-gold);     color: #6B5410; }

/* ── 页脚 ── */
.xz-footer {
    text-align: center;
    font-size: 0.78rem;
    color: var(--xz-ink-muted);
    margin-top: 24px;
    padding-top: 14px;
    border-top: 1px solid var(--xz-line);
    line-height: 1.9;
}
.xz-footer .couplet {
    font-family: var(--xz-font-kai);
    color: var(--xz-cinnabar);
    font-size: 0.9rem;
    letter-spacing: 0.16em;
}

/* ── 侧栏小标语 ── */
.xz-side-tagline {
    font-family: var(--xz-font-serif);
    font-size: 0.8rem;
    line-height: 1.9;
    color: var(--xz-sidebar-muted);
    text-align: center;
    padding: 10px 4px 2px;
    margin-top: 6px;
    border-top: 1px dashed rgba(201,180,138,0.3);
    letter-spacing: 0.04em;
}

/* ── 细滚动条 ── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(166, 58, 43, 0.30);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(166, 58, 43, 0.5); }
"""

# ═══════════════════════════════════════════════════════════
# 完整样式串（供 app.py 一次性注入）
# ═══════════════════════════════════════════════════════════
THEME_CSS = _ROOT + _BODY
