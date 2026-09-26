"""Streamlit web app — 法律智能助手（简约黑白主题）

Dual-mode interface:
    - 📖 条文问答 — simple RAG, one-shot retrieval + generation
    - 🤖 Agent 法律分析 — ReAct loop with tool calling + visible thought chain

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

from src.agent.react_agent import LawReActAgent
from src.domain import ASSISTANT_NAME, ASSISTANT_ROLE, KB_CATEGORY_TEXT
from src.generation.rag_pipeline import LawAgent
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
    page_title=f"{ASSISTANT_ROLE} · {ASSISTANT_NAME}",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入主题（设计令牌 + 全局样式）
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

# 进行各个状态的初始化（对话从空白开始；助手介绍以静态卡片展示在主页面顶部）
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback: Dict[str, str] = {}  # msg_index → "up" | "down"

if "retriever" not in st.session_state:
    with st.spinner("正在加载知识库与检索模型 ..."):
        st.session_state.retriever, st.session_state.chunk_count = load_retriever()

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = LawAgent(st.session_state.retriever, top_k=12)

if "react_agent" not in st.session_state:
    st.session_state.react_agent = LawReActAgent(
        st.session_state.retriever,
        max_iterations=10,
        max_history=10,
        top_k=12,
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
    # 用「显示标签 → 模式键」的显式映射，而不是让下游去嗅探标签文本。
    # 之前 render_mode_intro 用 `"Agent" in mode` 判断，标签一改就会
    # 静默显示错误的模式介绍卡。
    _MODE_LABELS = {"qa": "条文问答", "agent": "Agent 法律分析"}
    st.subheader("工作模式")
    mode_label = st.radio(
        "选择模式",
        list(_MODE_LABELS.values()),
        label_visibility="collapsed",
    )
    mode = next(k for k, v in _MODE_LABELS.items() if v == mode_label)
    is_agent_mode = mode == "agent"

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
        ui.render_stamp(chunk_count, "收录条文")
    with col2:
        ui.render_stamp(stage_count, "检索工序")
    ui.render_stamp(answered, "已答问题")

    st.divider()

    # ── actions ──
    ui.side_label("常用操作")

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.session_state.rag_agent.clear_history()
        st.session_state.react_agent.clear_history()
        st.rerun()

    if st.button("重建知识索引", use_container_width=True,
                 help="从 data/raw/laws/ 重新解析法规条文并重建索引"):
        with st.spinner("正在重建知识索引（嵌入需要几分钟）..."):
            try:
                from src.retrieval.factory import rebuild_index

                # 不要在这里包 RerankerProcessor —— 精排默认关闭（实测净负作用，
                # 会把查询改写的收益抹平），包上会让重建后的链路与默认配置分叉。
                new_hybrid, chunk_count = rebuild_index()

                if new_hybrid:
                    st.session_state.retriever = new_hybrid
                    st.session_state.rag_agent = LawAgent(st.session_state.retriever, top_k=12)
                    st.session_state.react_agent = LawReActAgent(
                        st.session_state.retriever, max_iterations=10, max_history=10, top_k=12
                    )
                    st.session_state.chunk_count = chunk_count
                    st.success(f"索引重建完成，共 {chunk_count:,} 条")
                else:
                    st.warning("未找到法规语料，请先运行 scripts/crawl_laws.py")
                st.rerun()
            except Exception as exc:
                st.error(f"重建失败：{exc}")

    st.divider()

    # ── about ──
    with st.expander("关于"):
        st.markdown(
            f"""
            **{ASSISTANT_ROLE} · {ASSISTANT_NAME}** 是一个基于 RAG + Agent 的
            法律垂直领域问答系统。

            **技术栈**
            - DeepSeek-V4-Pro（查询改写 + 答案生成）
            - BGE Embedding · Chroma · BM25 · RRF 融合
            - Streamlit

            **语料范围**
            - {KB_CATEGORY_TEXT}
            - 按条文切分，一条一个知识块

            语料来自国家法律法规数据库公开的正式文本。
            回答不构成法律意见。
            """
        )

    ui.render_side_tagline()

# ═══════════════════════════════════════════════════════════════
#  Main chat area — brand banner & message history
# ═══════════════════════════════════════════════════════════════

ui.brand_banner()
ui.render_welcome()            # 助手介绍 —— 主界面上方
ui.render_mode_intro(mode)     # 当前模式介绍 —— 其下（传模式键，非标签文本）

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
            ui.render_feedback(i, msg)

# ═══════════════════════════════════════════════════════════════
#  Chat input & processing
# ═══════════════════════════════════════════════════════════════

if prompt := st.chat_input(
    "输入你的法律问题，例如：违约金过高可以请求法院调减吗？"
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

            with st.spinner(f"{ASSISTANT_NAME}正在检索法条 ..."):
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

            # sources from tool calls (带真实来源文件名)
            sources = []
            for tc in tool_calls:
                call_sources = tc.get("sources") or []
                if call_sources:
                    for s in call_sources:
                        snippet = (s.get("snippet") or "")[:200]
                        if snippet:
                            src_name = s.get("source") or "知识库"
                            sources.append(f"[{tc['tool']}] 来源：{src_name} · {snippet}")
                else:
                    preview = tc.get("result_preview", "")[:200]
                    if preview:
                        sources.append(f"[{tc['tool']}] {preview}")

            ui.render_sources(sources)
            ui.render_thoughts(thoughts)

            # save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": prompt,        # 反馈要连同问题一起记录
                "sources": sources,
                "thoughts": thoughts,
                "iterations": iterations,
                "mode": "agent",
            })

        else:
            # ── Simple RAG mode ──
            with st.spinner("正在检索知识库 ..."):
                try:
                    result = st.session_state.rag_agent.chat(prompt)
                    answer = result.get("answer", "抱歉，处理过程中出现了问题。")
                    raw_sources = result.get("sources", [])
                    source_metas = result.get("source_metas", [])
                    mode_label = result.get("mode", "rag")
                except Exception as exc:
                    answer = f"处理出错：{exc}\n\n请检查 API Key 配置或稍后重试。"
                    raw_sources = []
                    source_metas = []
                    mode_label = "rag"

            st.markdown(answer)

            # format sources（带来源文件名）
            sources = [
                f"[来源{i+1}] {meta.get('source', '知识库')} · {meta.get('snippet', src)[:200]}"
                for i, (src, meta) in enumerate(zip(raw_sources, source_metas))
            ] if raw_sources else []

            if mode_label == "chat":
                st.caption("闲聊模式（未检索知识库）")
            elif mode_label == "out_of_kb":
                st.caption("知识库外问题（未检索到相关资料，未调用模型生成）")

            ui.render_sources(sources)

            # save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": prompt,        # 反馈要连同问题一起记录
                "sources": sources,
                "thoughts": None,
                "mode": mode_label,
            })

        # feedback for the new message
        _last = len(st.session_state.messages) - 1
        ui.render_feedback(_last, st.session_state.messages[_last])

# ═══════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════

ui.footer(chunk_count=st.session_state.get("chunk_count", 0))
