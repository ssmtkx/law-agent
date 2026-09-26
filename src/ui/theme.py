"""简约黑白主题 —— 设计令牌与全局样式。

只负责前端观感，不含任何业务逻辑。所有样式以 Streamlit ≥1.28 的稳定 DOM
选择器为锚点，并附 `.ui-*` 自定义类供 components 使用。

设计原则
--------
1. **无彩色**：全站只有黑、白、灰。交互态用黑色，层级用灰阶。
2. **无装饰**：没有纸纹、渐变、投影、圆角阴影。分隔靠 1px 细线。
3. **层级靠明度而非色相**：原先思考链的四类步骤用四种颜色区分，现在用
   四级灰阶 + 左边框深浅区分 —— 黑白条件下同样可辨，且不依赖色觉。
4. **留白优先**：行高与段间距整体放大，靠空间而不是线条划分区域。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# 调色板（纯灰阶）
# ═══════════════════════════════════════════════════════════════
PALETTE = {
    "bg": "#FFFFFF",            # 页面底色
    "surface": "#FAFAFA",       # 次级背景（侧栏 / 卡片）
    "surface_2": "#F4F4F4",     # 再深一级（用户气泡 / 代码块）
    "border": "#E5E5E5",        # 常规描边
    "border_mid": "#D4D4D4",    # 稍强描边
    "line_strong": "#111111",   # 强调线（左侧强调条 / 焦点）
    "text": "#111111",          # 正文
    "text_soft": "#555555",     # 次级文字
    "text_muted": "#8A8A8A",    # 弱化文字 / 占位符
    "inverse": "#FFFFFF",       # 反白文字

    # 思考链四级灰阶（浅 → 深）
    "step_1": "#F7F7F7",
    "step_2": "#F0F0F0",
    "step_3": "#E8E8E8",
    "step_4": "#1A1A1A",        # 最终回答：黑底反白，最重
}

# 无衬线字体栈（离线安全：只依赖系统字体，不引 Google Fonts）
FONT_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", '
    '"Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", '
    '"Source Han Sans SC", sans-serif'
)
# 数字与条号用等宽，让统计数字对齐、条号更易扫读
FONT_MONO = (
    '"SF Mono", "JetBrains Mono", "Cascadia Mono", Consolas, '
    '"Liberation Mono", "Courier New", monospace'
)

_ROOT = f"""
<style>
:root {{
  --ui-bg: {PALETTE['bg']};
  --ui-surface: {PALETTE['surface']};
  --ui-surface-2: {PALETTE['surface_2']};
  --ui-border: {PALETTE['border']};
  --ui-border-mid: {PALETTE['border_mid']};
  --ui-line-strong: {PALETTE['line_strong']};
  --ui-text: {PALETTE['text']};
  --ui-text-soft: {PALETTE['text_soft']};
  --ui-text-muted: {PALETTE['text_muted']};
  --ui-inverse: {PALETTE['inverse']};
  --ui-step-1: {PALETTE['step_1']};
  --ui-step-2: {PALETTE['step_2']};
  --ui-step-3: {PALETTE['step_3']};
  --ui-step-4: {PALETTE['step_4']};
  --ui-font-sans: {FONT_SANS};
  --ui-font-mono: {FONT_MONO};
}}
</style>
"""

_BODY = """
<style>
/* ═══════════════════════════════════════════════════════════
   法规检索助手 · 简约黑白 —— 全局样式
   ═══════════════════════════════════════════════════════════ */

/* ── 基础 ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--ui-font-sans);
    color: var(--ui-text);
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] { background-color: var(--ui-bg); }

/* ── 主内容列：收窄留白 ── */
.block-container {
    max-width: 880px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}

/* ── 顶栏 ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
    box-shadow: none;
}

/* ── 排版 ── */
h1, h2, h3, h4 {
    font-family: var(--ui-font-sans);
    color: var(--ui-text);
    letter-spacing: -0.01em;
    font-weight: 600;
}
[data-testid="stMarkdown"] p {
    line-height: 1.85;
    color: var(--ui-text-soft);
}
[data-testid="stMarkdown"] strong { color: var(--ui-text); font-weight: 600; }
[data-testid="stCaptionContainer"] { color: var(--ui-text-muted); }
[data-testid="stMarkdown"] a {
    color: var(--ui-text);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-thickness: 1px;
}
::selection { background: var(--ui-text); color: var(--ui-inverse); }
hr, [data-testid="stMarkdown"] hr {
    border: none;
    border-top: 1px solid var(--ui-border);
    margin: 1.6rem 0;
}

/* ── 表格 ── */
[data-testid="stMarkdown"] table { border-collapse: collapse; width: 100%; }
[data-testid="stMarkdown"] th {
    background: var(--ui-surface);
    color: var(--ui-text);
    font-weight: 600;
}
[data-testid="stMarkdown"] th,
[data-testid="stMarkdown"] td {
    border: 1px solid var(--ui-border);
    padding: 8px 12px;
}

/* ═══════════════════════════════════════════════════════════
   侧栏 —— 浅灰面板，右侧一条细线
   ═══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: var(--ui-surface);
    border-right: 1px solid var(--ui-border);
}
[data-testid="stSidebarContent"] { background: transparent; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: var(--ui-text);
    font-family: var(--ui-font-sans);
    font-weight: 600;
    letter-spacing: 0;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] li { color: var(--ui-text-soft); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--ui-text-muted); }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: var(--ui-text-soft); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p { color: var(--ui-text); }
[data-testid="stSidebar"] hr { border-top-color: var(--ui-border); }

/* ═══════════════════════════════════════════════════════════
   按钮 —— 描边为主，主操作用实心黑
   ═══════════════════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button,
[data-testid="stBaseButton-secondary"] {
    border-radius: 6px;
    border: 1px solid var(--ui-border-mid);
    background: var(--ui-bg);
    color: var(--ui-text);
    font-family: var(--ui-font-sans);
    font-weight: 500;
    transition: background .12s ease, border-color .12s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--ui-text);
    background: var(--ui-surface);
    color: var(--ui-text);
}
.stButton > button:active,
.stDownloadButton > button:active { transform: translateY(1px); }
.stButton > button:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible {
    outline: 2px solid var(--ui-text);
    outline-offset: 1px;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--ui-text);
    color: var(--ui-inverse);
    border-color: var(--ui-text);
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #333333;
    border-color: #333333;
    color: var(--ui-inverse);
}

/* ═══════════════════════════════════════════════════════════
   聊天气泡 —— 平的，只用描边和一条左侧竖线
   ═══════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: var(--ui-bg);
    border: 1px solid var(--ui-border);
    border-left: 2px solid var(--ui-text);
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: none;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--ui-surface-2);
    border-color: var(--ui-border);
    border-left: 2px solid var(--ui-border-mid);
    color: var(--ui-text);
}
[data-testid="stChatMessage"] .stChatMessageContent { font-size: 0.94rem; }

/* ── 头像 ── */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: transparent;
    border-radius: 4px;
    overflow: hidden;
    width: 26px;
    height: 26px;
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
    background: var(--ui-bg);
    border: 1px solid var(--ui-border-mid);
    border-radius: 6px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--ui-text);
    box-shadow: none;
}
[data-testid="stChatInput"] textarea { color: var(--ui-text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--ui-text-muted); }

/* ═══════════════════════════════════════════════════════════
   折叠面板
   ═══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--ui-border);
    border-radius: 6px;
    background: var(--ui-bg);
}
[data-testid="stExpander"] summary:hover { color: var(--ui-text); }
.streamlit-expanderHeader {
    font-family: var(--ui-font-sans);
    color: var(--ui-text-soft);
    font-weight: 500;
}

/* ═══════════════════════════════════════════════════════════
   .ui-* 自定义组件
   ═══════════════════════════════════════════════════════════ */

/* ── 品牌横幅：不用卡片，只用一条底线 ── */
.ui-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 16px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--ui-text);
}
.ui-seal {
    flex: 0 0 auto;
    width: 44px; height: 44px;
    display: flex; align-items: center; justify-content: center;
    background: var(--ui-text);
    color: var(--ui-inverse);
    font-family: var(--ui-font-sans);
    font-size: 20px; font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0;
}
.ui-banner-title {
    font-family: var(--ui-font-sans);
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ui-text);
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.ui-banner-sub {
    font-size: 0.85rem;
    color: var(--ui-text-soft);
    margin-top: 2px;
}
.ui-banner-tag {
    font-size: 0.74rem;
    color: var(--ui-text-muted);
    letter-spacing: 0.06em;
    margin-top: 2px;
}
.ui-banner .ui-ic { color: var(--ui-text-muted); }

/* ── 内联 SVG 图标 ── */
.ui-ic {
    display: inline-block;
    width: 1em;
    height: 1em;
    vertical-align: -0.16em;
    margin-right: 0.3em;
}

/* ── 助手介绍卡 ── */
.ui-welcome {
    padding: 14px 16px;
    margin-bottom: 12px;
    background: var(--ui-surface);
    border-left: 2px solid var(--ui-text);
}
.ui-welcome-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--ui-text);
}
.ui-welcome-desc {
    font-size: 0.84rem;
    color: var(--ui-text-soft);
    margin-top: 4px;
    line-height: 1.8;
}

/* ── 当前模式介绍卡 ── */
.ui-mode-intro {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 14px 16px;
    margin-bottom: 14px;
    background: var(--ui-surface);
    border-left: 2px solid var(--ui-border-mid);
}
.ui-mode-intro > .ui-ic {
    flex: 0 0 auto;
    margin-top: 3px;
    width: 1.05em;
    height: 1.05em;
    color: var(--ui-text-muted);
}
.ui-mode-intro-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--ui-text);
}
.ui-mode-intro-desc {
    font-size: 0.84rem;
    color: var(--ui-text-soft);
    margin-top: 3px;
    line-height: 1.8;
}
.ui-chip {
    display: inline-block;
    font-size: 0.76rem;
    color: var(--ui-text-soft);
    background: var(--ui-bg);
    border: 1px solid var(--ui-border);
    border-radius: 3px;
    padding: 3px 10px;
    margin: 8px 6px 0 0;
    white-space: nowrap;
}

/* ── 统计方框 ── */
.ui-stamp {
    text-align: center;
    padding: 10px 6px 8px;
    background: var(--ui-bg);
    border: 1px solid var(--ui-border-mid);
    border-radius: 4px;
    color: var(--ui-text);
    margin: 2px 0;
}
.ui-stamp .value {
    font-family: var(--ui-font-mono);
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.ui-stamp .label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    margin-top: 3px;
    color: var(--ui-text-muted);
}

/* ── 侧栏标题 / 分区标签 / 方印 ── */
.ui-side-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ui-text);
    text-align: center;
    letter-spacing: 0.06em;
    margin: 4px 0 0;
}
.ui-side-seal {
    width: 36px; height: 36px;
    margin: 0 auto 6px;
    display: flex; align-items: center; justify-content: center;
    background: var(--ui-text);
    color: var(--ui-inverse);
    font-size: 17px; font-weight: 600;
    border-radius: 4px;
}
.ui-side-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--ui-text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 16px 0 7px;
    padding-left: 8px;
    border-left: 2px solid var(--ui-text);
}

/* ── 今日提示 ── */
.ui-daily {
    font-size: 0.8rem;
    color: var(--ui-text-muted);
    text-align: center;
    padding: 8px 4px;
    margin: 10px 0 2px;
    border-top: 1px solid var(--ui-border);
    border-bottom: 1px solid var(--ui-border);
    line-height: 1.7;
}

/* ── 分隔线 ── */
.ui-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 20px 0 10px;
    color: var(--ui-text-muted);
    font-size: 0.78rem;
    letter-spacing: 0.24em;
}
.ui-divider::before,
.ui-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--ui-border);
}

/* ── 引用来源 ── */
.ui-source {
    position: relative;
    background: var(--ui-bg);
    border: 1px solid var(--ui-border);
    border-left: 2px solid var(--ui-text);
    border-radius: 0 4px 4px 0;
    padding: 9px 12px 9px 34px;
    margin: 6px 0;
    font-size: 0.84rem;
    color: var(--ui-text-soft);
    line-height: 1.7;
}
.ui-source::before {
    content: "§";
    position: absolute;
    left: 11px; top: 50%;
    transform: translateY(-50%);
    font-family: var(--ui-font-mono);
    color: var(--ui-text-muted);
    font-size: 0.9rem;
}

/* ═══════════════════════════════════════════════════════════
   思考链 —— 用四级灰阶替代四种色相
   （黑白条件下同样可辨，且不依赖色觉）
   ═══════════════════════════════════════════════════════════ */
.ui-thought {
    padding: 9px 13px;
    margin: 6px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.84rem;
    color: var(--ui-text-soft);
    line-height: 1.7;
}
.ui-thought .ui-tnum {
    font-family: var(--ui-font-mono);
    font-weight: 600;
    margin-right: 9px;
    white-space: nowrap;
    color: var(--ui-text);
}
.ui-thought-thought     { background: var(--ui-step-1); border-left: 2px solid #CCCCCC; }
.ui-thought-action      { background: var(--ui-step-2); border-left: 2px solid #999999; }
.ui-thought-observation { background: var(--ui-step-3); border-left: 2px solid #555555; }
.ui-thought-answer      {
    background: var(--ui-step-4);
    border-left: 2px solid var(--ui-step-4);
    color: #EDEDED;
    font-weight: 500;
}
.ui-thought-answer .ui-tnum { color: var(--ui-inverse); }

/* ── 页脚 ── */
.ui-footer {
    text-align: center;
    font-size: 0.76rem;
    color: var(--ui-text-muted);
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--ui-border);
    line-height: 2;
}

/* ── 侧栏小标语 ── */
.ui-side-tagline {
    font-size: 0.76rem;
    line-height: 2;
    color: var(--ui-text-muted);
    text-align: center;
    padding: 12px 4px 2px;
    margin-top: 8px;
    border-top: 1px solid var(--ui-border);
    letter-spacing: 0.02em;
}

/* ── 滚动条 ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--ui-border-mid);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--ui-text-muted); }
"""

# ═══════════════════════════════════════════════════════════
# 完整样式串（供 app.py 一次性注入）
# ═══════════════════════════════════════════════════════════
THEME_CSS = _ROOT + _BODY
