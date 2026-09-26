"""ReAct Agent for the legal assistant.

A ReAct (Reasoning + Acting) loop with tool calling, a legal-analyst
persona, structured output and conversation memory.

Architecture::

    User input
      │
      ▼
    ReAct Loop (max 10 iterations)
      ├─ Thought: LLM decides which tool(s) to call
      ├─ Action:  execute tool(s) against the retrieval pipeline
      ├─ Observation: feed results back to LLM
      └─ ... repeat until Final Answer
      │
      ▼
    Structured output (法律依据 → 适用分析 → 风险提示)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from src.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from src.domain import (ASSISTANT_NAME, DISCLAIMER, KB_CATEGORY_TEXT,
                        KB_FIELD_TEXT)
from src.utils.tracker import get_tracker

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = f"""你叫「{ASSISTANT_NAME}」，是一位法律智能助手，为中国的法律从业者与普通公众提供法规查询与适用分析。
你的知识库收录{KB_CATEGORY_TEXT}的正式条文，涉及{KB_FIELD_TEXT}等领域。
你可以通过以下三个工具检索法规条文：

## 工具说明

| 工具 | 用途 | 何时使用 |
|------|------|----------|
| `search_knowledge` | 条文检索 | 用户问某行为是否合法、法律后果、条文含义等 |
| `query_tool_list` | 查询适用法条清单 | 用户问"这涉及哪些法律""该看哪部法""有哪些相关规定" |
| `check_common_mistakes` | 查询常见误区与法律风险 | 用户问"有什么风险""要注意什么""常见的坑" |

## 工作流程

1. **理解问题**：先梳理用户描述的事实与诉求，识别涉及的法律关系
2. **选择工具**：根据问题类型选择工具（通常先用 search_knowledge 找条文）
3. **检索条文**：调用工具获取知识库中的法规原文
4. **组织回答**：依据检索到的条文作答，写明法规全称与条号
5. **不足时补检**：一次检索不够时，换关键词或换工具再检索

## 回答格式

分析类问题时，按以下三段式结构组织：

### 📖 法律依据
列出检索到的具体条文，写明《法规全称》第X条，并摘录关键内容

### 🔍 适用分析
结合用户描述的事实，分析条文如何适用；指出成立要件、可能的结果或救济途径

### ⚠️ 风险提示
指出该情形下的常见误区、举证要点、时效限制，或结论的不确定之处

## 核心规则

1. **只引用检索到的条文**，写明法规全称与条号，并在句末标注 [来源X]。**严禁凭记忆编造条号、司法解释或案例**
2. 引用条文内容要与原文一致，可以摘录关键句，但不要改写法条的规范含义
3. 如果检索结果**不足以完整回答**：说明现有条文未能覆盖，可以补充一般法律原理，但必须明确区分二者
4. 如果工具明确提示"知识库未命中"，如实告知该问题暂不在知识库覆盖范围内，建议用户查阅国家法律法规数据库
5. 如果问题完全超出法律领域（如问天气、编程等），友好说明你的专业范围并引导回正题
6. 你**不提供法律意见**，不替用户决定该怎么做。回答保持严谨克制，用分点和短句
7. 涉及具体行动建议时，结尾提醒：{DISCLAIMER}
8. 多轮对话时可以参考之前的对话内容，但每次新问题仍需重新检索确认"""

# ═══════════════════════════════════════════════════════════════════
#  ReAct Agent
# ═══════════════════════════════════════════════════════════════════


class LawReActAgent:
    """Legal analyst agent with ReAct reasoning and tool calling.

    Parameters
    ----------
    retriever :
        The retrieval object (e.g. RerankerProcessor wrapping HybridRetriever).
    api_key : str
        DeepSeek API key.  Falls back to ``DEEPSEEK_API_KEY`` env var.
    base_url : str
        DeepSeek API base URL.  Falls back to ``DEEPSEEK_BASE_URL`` env var
        or ``https://api.deepseek.com``.
    model : str
        Model name to use (default ``deepseek-v4-pro``).
    max_iterations : int
        Maximum ReAct loop iterations per turn (default 10).
    max_history : int
        Number of recent conversation turns to retain (default 10).
    top_k : int
        Default number of results per tool call (default 5).
    on_thought : Callable or None
        Optional callback invoked with each Thought/Action/Observation step
        for UI visualisation (Streamlit / CLI).  Signature::
            on_thought(step_type: str, content: str) -> None
    """

    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-v4-pro",
        max_iterations: int = 10,
        max_history: int = 10,
        top_k: int = 5,
        on_thought: Optional[Callable[[str, str], None]] = None,
    ):
        import os

        self.retriever = retriever
        self.model = model
        self.max_iterations = max_iterations
        self.max_history = max_history
        self.on_thought = on_thought

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
        )
        self.tool_executor = ToolExecutor(retriever, top_k=top_k)

        # conversation memory: user ↔ assistant pairs only (no tool messages)
        self.history: List[Dict[str, str]] = []

    # ── public API ──────────────────────────────────────────────

    def chat(self, question: str) -> Dict[str, Any]:
        """Handle one conversation turn through the ReAct loop.

        Returns
        -------
        dict with keys:
            ``answer``   — final text answer
            ``sources``  — list of source strings cited
            ``question`` — original question
            ``iterations`` — ReAct loop count for this turn
            ``tool_calls`` — list of tool calls made (for UI visualisation)
        """
        question = question.strip()

        # ── build the message list for this turn ──
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        # include past conversation (compressed: last N user/assistant pairs)
        messages.extend(self.history[-self.max_history * 2 :])
        messages.append({"role": "user", "content": question})

        # ── ReAct loop ──
        tool_calls_log: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            try:
                tracker = get_tracker()
                with tracker.track("react_agent", model=self.model) as ctx:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        temperature=0.3,
                        # 见 rag_pipeline 的说明：不设限会过度推理，
                        # 又慢又容易把预算耗光导致正文为空
                        reasoning_effort="low",
                        max_tokens=8192,
                    )
                    ctx.record(response)
            except Exception as exc:
                logger.error("LLM call failed (iteration %d): %s", iterations, exc)
                # retry once after a short description of the error
                if iterations == 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"（系统提示：上一次调用遇到错误，请重试。错误信息：{exc}）",
                        }
                    )
                    continue
                return self._fallback_answer(question, f"API 调用失败: {exc}")

            message = response.choices[0].message

            # ── no tool calls → final answer ──
            if not message.tool_calls:
                answer = message.content or ""
                self._emit("answer", answer)
                # save to conversation memory
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": answer})
                if len(self.history) > self.max_history * 2:
                    self.history = self.history[-self.max_history * 2 :]
                return {
                    "answer": answer,
                    "sources": self._extract_sources(tool_calls_log),
                    "question": question,
                    "iterations": iterations,
                    "tool_calls": tool_calls_log,
                }

            # ── tool calls → execute and continue ──
            # append assistant message (with tool_calls) to the conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                self._emit("action", f"调用工具: {tool_name}({arguments})")

                result, sources = self.tool_executor.execute(tool_name, arguments)
                self._emit("observation", f"返回 {len(result)} 字符")

                tool_calls_log.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result_preview": result[:300],
                        "sources": sources,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # ── max iterations reached ──
        # force a final answer
        messages.append(
            {
                "role": "user",
                "content": "（请基于目前已检索到的资料，给出最终回答。如果资料不足，请诚实说明。）",
            }
        )
        try:
            tracker = get_tracker()
            with tracker.track("react_agent_final", model=self.model) as ctx:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
                ctx.record(response)
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = "抱歉，查询过程中遇到了技术问题。请稍后重试或简化您的问题。"

        self._emit("answer", answer)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        return {
            "answer": answer,
            "sources": self._extract_sources(tool_calls_log),
            "question": question,
            "iterations": iterations,
            "tool_calls": tool_calls_log,
        }

    # ── helpers ─────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.history = []

    def _emit(self, step_type: str, content: str) -> None:
        """Notify the optional thought callback."""
        if self.on_thought:
            try:
                self.on_thought(step_type, content)
            except Exception:
                pass

    @staticmethod
    def _extract_sources(tool_calls_log: List[Dict[str, Any]]) -> List[str]:
        """Extract source citations from the tool-call log."""
        sources: List[str] = []
        for call in tool_calls_log:
            call_sources = call.get("sources") or []
            if call_sources:
                for s in call_sources:
                    snippet = (s.get("snippet") or "")[:200]
                    if snippet:
                        src_name = s.get("source") or "知识库"
                        sources.append(f"[{call['tool']}·{src_name}] {snippet}")
            else:
                preview = call.get("result_preview", "")
                if preview:
                    sources.append(f"[{call['tool']}] {preview}")
        return sources

    @staticmethod
    def _fallback_answer(question: str, reason: str) -> Dict[str, Any]:
        """Return a safe fallback when the agent cannot complete."""
        return {
            "answer": f"抱歉，处理您的问题时遇到了技术问题（{reason}）。请稍后重试。",
            "sources": [],
            "question": question,
            "iterations": 0,
            "tool_calls": [],
        }
