"""UI components — 简约黑白主题的 HTML 渲染块。

所有组件基于 Streamlit ≥1.28（用 ``st.markdown(..., unsafe_allow_html=True)``
渲染，不依赖更高版本的 ``st.html``）。仅负责渲染，不携带业务状态。

品牌名与知识库范围取自 :mod:`src.domain`，不在本文件里重写。
"""

from __future__ import annotations

import datetime
import html
from typing import Dict, List

import streamlit as st

from src.domain import (ASSISTANT_NAME, ASSISTANT_ROLE, DISCLAIMER,
                        KB_FIELD_TEXT)
from src.ui.icons import ICON_PULSE, ICON_SEARCH

# ═══════════════════════════════════════════════════════════
# 品牌
# ═══════════════════════════════════════════════════════════


def brand_banner() -> None:
    """主区域顶部的品牌横幅：黑色方印 + 题签 + 副题。"""
    st.markdown(
        f"""
        <div class="ui-banner">
          <div class="ui-seal">{ASSISTANT_NAME}</div>
          <div>
            <div class="ui-banner-title">{ASSISTANT_ROLE} · {ASSISTANT_NAME}</div>
            <div class="ui-banner-sub">{ICON_SEARCH} 条文问答　｜　{ICON_PULSE} Agent 法律分析</div>
            <div class="ui-banner-tag">中国法律法规检索 · 条文可溯源</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_header() -> None:
    """侧栏头部：小方印 + 品牌名 + 一句说明。"""
    st.markdown(
        f"""
        <div class="ui-side-seal">法</div>
        <div class="ui-side-title">{ASSISTANT_NAME}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(ASSISTANT_ROLE)


def side_label(text: str) -> None:
    """侧栏分区小标签，如「知识库统计」「常用操作」。"""
    st.markdown(f'<div class="ui-side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 生活气息小件
# ═══════════════════════════════════════════════════════════

# 周一 ~ 周日的今日提示
_DAILY_LINES = {
    0: "查查这周要用到的条文吧",
    1: "合同条款逐条看，签字之前不急",
    2: "程序问题先理清，实体问题才好谈",
    3: "翻翻旧法条，留意有没有修订",
    4: "时效期间要记牢，别让它悄悄溜走",
    5: "整理整理知识库和笔记",
    6: "休息一下，喝杯茶再继续",
}


def daily_line() -> None:
    """按星期展示一句今日提示，添一点生活气。"""
    line = _DAILY_LINES[datetime.date.today().weekday()]
    st.markdown(f'<div class="ui-daily">今日 · {html.escape(line)}</div>', unsafe_allow_html=True)


def render_divider(text: str | None = None) -> None:
    """细分隔线；传 text 则居中显示（如「欢迎提问」）。"""
    inner = f"<span>{html.escape(text)}</span>" if text else ""
    st.markdown(f'<div class="ui-divider">{inner}</div>', unsafe_allow_html=True)


def render_welcome() -> None:
    """主页面顶部的介绍卡（原先的第一条欢迎消息）。"""
    st.markdown(
        f"""
        <div class="ui-welcome">
          <div class="ui-welcome-title">认识一下，我是{ASSISTANT_NAME}</div>
          <div class="ui-welcome-desc">
            法律智能助手，帮你检索现行有效的法规条文，涉及{KB_FIELD_TEXT}等领域。
            回答只引用检索到的条文并标注条号，供你到官方文本核对；检索不到会直说。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 两种模式的介绍文案与示例问题。
# tone 用明度而非色相区分两种模式 —— 黑白主题下没有颜色可用。
_MODE_INTRO = {
    "qa": {
        "title": "条文问答",
        "icon": ICON_SEARCH,
        "desc": "快速检索法条、查适用条件，适合明确的、单点的法律问题。",
        "examples": ["违约金过高可以请求法院调减吗？", "房屋租赁合同最长可以签多少年？"],
        "tone": "var(--ui-text)",
    },
    "agent": {
        "title": "Agent 法律分析",
        "icon": ICON_PULSE,
        "desc": "多步推理梳理法律关系：自主检索相关法条，分析适用要件，提示风险与举证要点。",
        "examples": ["公司拖欠工资三个月，我可以主张什么？", "买的房子有抵押，合同还有效吗？"],
        "tone": "var(--ui-text-muted)",
    },
}


def render_mode_intro(mode_key: str) -> None:
    """根据当前模式，在主页横幅下渲染一段模式介绍与示例问题。

    ``mode_key`` 是 ``"qa"`` / ``"agent"``，不是展示用的标签文本。
    此前的实现用 ``"Agent" in mode`` 去嗅探中文标签，标签一改就静默
    显示错误的模式介绍卡。
    """
    info = _MODE_INTRO.get(mode_key, _MODE_INTRO["qa"])
    chips = "".join(
        f'<span class="ui-chip">{html.escape(e)}</span>' for e in info["examples"]
    )
    st.markdown(
        f"""
        <div class="ui-mode-intro" style="border-left-color:{info['tone']};">
          <span style="color:{info['tone']};">{info['icon']}</span>
          <div>
            <div class="ui-mode-intro-title">当前模式 · {html.escape(info['title'])}</div>
            <div class="ui-mode-intro-desc">{html.escape(info['desc'])}</div>
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
        <div class="ui-side-tagline">
          法律 · 行政法规 · 司法解释<br/>
          检索条文，标注出处
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# 印章统计
# ═══════════════════════════════════════════════════════════


def render_stamp(value, label: str) -> None:
    """统计方框：展示数值与标签。"""
    st.markdown(
        f'<div class="ui-stamp"><div class="value">{value}</div>'
        f'<div class="label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# 消息内容
# ═══════════════════════════════════════════════════════════


def render_sources(sources: List[str] | None) -> None:
    """引用来源：左侧竖线的条目标签。"""
    if not sources:
        return
    with st.expander(f"引用来源（{len(sources)} 条）", expanded=False):
        for src in sources:
            st.markdown(f'<div class="ui-source">{html.escape(src)}</div>', unsafe_allow_html=True)


# 思考链各步骤的标签
_THOUGHT_META = {
    "thought": ("思考", "ui-thought-thought"),
    "action": ("行动", "ui-thought-action"),
    "observation": ("观察", "ui-thought-observation"),
    "answer": ("完成", "ui-thought-answer"),
}
_NUMS = "①②③④⑤⑥⑦⑧⑨⑩"


def render_thoughts(thoughts: List[Dict[str, str]] | None) -> None:
    """思考过程：四类步骤按明度递增区分（思考 → 行动 → 观察 → 完成）。"""
    if not thoughts:
        return
    with st.expander(f"思考过程（{len(thoughts)} 步）", expanded=True):
        for i, step in enumerate(thoughts, 1):
            step_type = step.get("type", "thought")
            content = step.get("content", "")
            name, cls = _THOUGHT_META.get(step_type, _THOUGHT_META["thought"])
            num = _NUMS[i - 1] if i <= len(_NUMS) else str(i)
            st.markdown(
                f'<div class="ui-thought {cls}">'
                f'<span class="ui-tnum">{num}{name}</span>{html.escape(content)}'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_feedback(msg_index: int, msg: Dict | None = None) -> None:
    """有帮助 / 需改进 按钮。

    点击后**落盘**（此前只写 session_state，刷新即丢且无法复盘）。记录会带上
    问题、回答与引用，否则事后看到一个 ``down`` 完全不知道差在哪。
    """
    from src.utils import feedback

    key_up = f"fb_up_{msg_index}"
    key_down = f"fb_down_{msg_index}"
    current = st.session_state.feedback.get(str(msg_index))

    def _submit(rating: str) -> None:
        st.session_state.feedback[str(msg_index)] = rating
        # 注意别写成 ``msg = msg or {}`` —— 那会让 msg 变成 _submit 的局部变量，
        # 右侧读到未赋值的局部名，抛 UnboundLocalError（且因为在写入之后，UI 仍
        # 显示"已反馈"，看不出落盘从未发生）。
        _msg = msg or {}
        feedback.record(
            rating,
            question=_msg.get("question", ""),
            answer=_msg.get("content", ""),
            mode=_msg.get("mode", ""),
            sources=[
                s.get("source", "") if isinstance(s, dict) else str(s)
                for s in (_msg.get("sources") or [])
            ],
            iterations=_msg.get("iterations"),
        )

    c1, c2, c3 = st.columns([1, 1, 14])
    with c1:
        if st.button("有帮助", key=key_up, help="回答是否有帮助",
                     disabled=(current == "up"), use_container_width=True):
            _submit("up")
            st.rerun()
    with c2:
        if st.button("需改进", key=key_down, help="回答需要改进",
                     disabled=(current == "down"), use_container_width=True):
            _submit("down")
            st.rerun()
    if current:
        with c3:
            label = "已反馈：有帮助" if current == "up" else "已反馈：需改进"
            st.caption(label)


# ═══════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════


def footer(chunk_count: int = 0) -> None:
    """页脚：溯源/免责说明 + 知识库条文数。"""
    st.markdown(
        f"""
        <div class="ui-footer">
          本助手基于公开法规条文检索生成 · 回答标注法规与条号<br/>
          {DISCLAIMER}<br/>
          <span style="color:var(--ui-text-muted);">知识库条文 {chunk_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
