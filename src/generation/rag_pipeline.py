"""Legal RAG agent: identity + conversation memory + retrieval pipeline.

Architecture:
    User input → intent detection →
      ├─ chat (greeting/identity) → direct LLM response
      └─ knowledge query → retrieve → RAG → LLM response
"""

import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.domain import DISCLAIMER
from src.generation.prompts import (
    SYSTEM_PROMPT,
    CHAT_RULES,
    RAG_RULES,
    RAG_QA_PROMPT,
    OUT_OF_KB_REPLY,
)
from src.utils.tracker import get_tracker

load_dotenv()

# ── Simple intent keywords ──
_CHAT_KEYWORDS = {
    "你好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好",
    "你是谁", "你叫什么", "介绍一下你自己", "你能做什么", "你有什么功能",
    "谢谢", "感谢", "再见", "拜拜", "bye",
    "你是什么",
}


class LawAgent:
    """A legal assistant agent with identity, conversation memory and RAG."""

    def __init__(self, retriever, top_k: int = 12, model: Optional[str] = None):
        self.retriever = retriever  # HybridRetriever or Chroma collection
        self.top_k = top_k
        # top_k 默认 12 而非 5，是实测定的：这类问题的决定性条文常常不在前 5。
        # 例：问「公司没交社保，离职能否要赔偿」，法条链是
        # 劳动合同法 38 条（可解除）→ 46 条（应付经济补偿）→ 47 条（计算标准），
        # 而实测 38/46 分别排在第 11、12 名，前 10 名全是匹配了问题表面字眼
        # （"社保""赔偿"）但法律上不决定结论的条文。取 5 条时模型只能诚实地说
        # "资料未覆盖"，取 12 条即可完整作答。
        # 代价是上下文从约 2k 涨到约 4.8k 字符，延迟影响可忽略。
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
        )
        # conversation memory: list of {"role": "user"|"assistant", "content": str}
        self.history: list[dict] = []
        self.max_history = 10  # keep last N turns

    # ── public API ─────────────────────────────────────────

    def chat(self, question: str) -> dict:
        """Handle one turn of conversation.

        Returns {"answer": str, "sources": list, "question": str, "mode": "chat"|"rag"}
        """
        question = question.strip()

        # ── intent detection ──
        if self._is_chat(question):
            result = self._chat_reply(question)
        else:
            result = self._rag_reply(question)

        # ── update memory ──
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return result

    def clear_history(self):
        """Reset conversation memory."""
        self.history = []

    # ── internal ───────────────────────────────────────────

    def _is_chat(self, question: str) -> bool:
        """Rule-based intent detection.

        先去掉所有标点与空白，再判断是否为纯问候/身份类闲聊：
        整句完全等于聊天关键词，或剥离开头问候词后剩余部分很短（≤4字）。
        这样「你好，劳动合同违约金怎么算？」这类含问候语的专业问题
        不会被误路由到闲聊。
        """
        compact = re.sub(
            r"[\s，。！？!?,.、；;：:（()）【】\[\]\"'“”‘’]+",
            "",
            question.strip().lower(),
        )
        if not compact:
            return False
        if compact in _CHAT_KEYWORDS:
            return True
        # 剥离开头问候词后剩余内容很短（如「你好，你是谁」「介绍一下你自己」）→ 闲聊
        for kw in _CHAT_KEYWORDS:
            if compact.startswith(kw):
                rest = compact[len(kw):]
                if not rest or len(rest) <= 4:
                    return True
        return False

    def _chat_reply(self, question: str) -> dict:
        """Respond to greetings / identity questions without RAG."""
        system = SYSTEM_PROMPT.format(chat_rules=CHAT_RULES)
        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-6:])  # recent context
        messages.append({"role": "user", "content": question})

        tracker = get_tracker()
        with tracker.track("rag_chat", model=self.model) as ctx:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                # reasoning_effort="low"：实测同一问题
                #   baseline   78.9s  推理 9402 字  正文 1058 字
                #   low         9.4s  推理  192 字  正文 1223 字
                # 关掉大部分推理反而答案更完整 —— 不设限时模型会过度推理，
                # 把预算耗在分析上，最后正文反而更短。同时它也是空答案的
                # 根因（推理吃光 max_tokens）。
                reasoning_effort="low",
                max_tokens=8192,
            )
            ctx.record(resp)
        return {
            "answer": resp.choices[0].message.content,
            "sources": [],
            "question": question,
            "mode": "chat",
        }

    def _rag_reply(self, question: str) -> dict:
        """Retrieve → RAG prompt → generate."""
        # ── retrieve ──
        results = self.retriever.query(
            query_texts=[question],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]
        metas: List[Optional[dict]] = results.get("metadatas", [[]])[0] or []
        if len(metas) < len(docs):
            metas = metas + [None] * (len(docs) - len(metas))

        if not docs:
            # 未命中知识库 → 如实拒答，不调用 LLM，避免幻觉。
            # 这条分支只有在召回层设了相关性下限时才可能走到（见 reranker
            # 的 min_score）；此前它一直是死代码，任何问题都会拿到 k 条上下文。
            return {
                "answer": OUT_OF_KB_REPLY,
                "sources": [],
                "source_metas": [],
                "question": question,
                "mode": "out_of_kb",
            }

        # ── build RAG prompt ──
        context = "\n\n".join(
            f"[来源{i+1}]（{_source_name(metas[i])}）{doc}"
            for i, doc in enumerate(docs)
        )
        rag_body = RAG_QA_PROMPT.format(context=context, question=question)

        system = SYSTEM_PROMPT.format(chat_rules=RAG_RULES)
        messages = [{"role": "system", "content": system}]
        # include recent history so agent can reference previous turns
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": rag_body})

        tracker = get_tracker()
        with tracker.track("rag_knowledge", model=self.model) as ctx:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                # reasoning_effort="low"：实测同一问题
                #   baseline   78.9s  推理 9402 字  正文 1058 字
                #   low         9.4s  推理  192 字  正文 1223 字
                # 关掉大部分推理反而答案更完整 —— 不设限时模型会过度推理，
                # 把预算耗在分析上，最后正文反而更短。同时它也是空答案的
                # 根因（推理吃光 max_tokens）。
                reasoning_effort="low",
                max_tokens=8192,
            )
            ctx.record(resp)
        return {
            "answer": resp.choices[0].message.content,
            "sources": docs,
            "source_metas": [
                {"source": _source_name(meta), "snippet": doc[:200]}
                for doc, meta in zip(docs, metas)
            ],
            "question": question,
            "mode": "rag",
        }


def _source_name(meta: Optional[dict]) -> str:
    """Human-readable citation label for a retrieved chunk.

    Renders as 《法规全称》第X条, which is what a lawyer would cite and what
    a reader can verify against the official text. Falls back to the
    category, and only then to a generic label.

    注意：元数据缺失时退化成「知识库」会让「100% 可溯源」名不副实 ——
    这正是 BM25 缓存陈旧导致的现象（块 id 与向量库对不上、元数据全丢）。
    Agent 侧因此把元数据缺失率作为一项可观测指标。
    """
    if not meta:
        return "知识库"
    title = meta.get("law_title")
    if title:
        article = meta.get("article_no") or ""
        # 尚未生效的条文必须带标识 —— 引用标签是用户最先看到的东西
        pending = "（尚未生效）" if meta.get("status") == 4 else ""
        return f"《{title}》{article}{pending}"
    return str(meta.get("category") or "知识库")
