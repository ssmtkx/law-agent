"""Tool definitions and execution for the legal ReAct loop.

Three tools:

- ``search_knowledge``      — statute search (the general-purpose one)
- ``query_tool_list``       — which statutes govern a given matter
- ``check_common_mistakes`` — common misconceptions and legal risks

The names are kept from the original design; only the descriptions and the
internal query-augmentation strings changed. Tool *shape* is domain-neutral
(one general search plus two facet searches), so reusing the structure keeps
the ReAct loop, caching and UI rendering untouched.

Tool definitions follow the OpenAI function-calling schema so they can be
passed directly to the DeepSeek chat-completions endpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional


def _citation_label(meta: Optional[dict]) -> str:
    """Render a chunk's citation as 《法规全称》第X条.

    Falls back to the category, then to a generic label. A generic fallback
    across the board is a symptom, not a normal state — it means the chunk
    metadata was lost (historically: a stale BM25 cache whose ids no longer
    matched the vector store).
    """
    if not meta:
        return "知识库"
    title = meta.get("law_title")
    if title:
        pending = "（尚未生效）" if meta.get("status") == 4 else ""
        return f"《{title}》{meta.get('article_no') or ''}{pending}"
    return str(meta.get("category") or "知识库")

# ═══════════════════════════════════════════════════════════════════
#  OpenAI-compatible tool definitions
# ═══════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索法规条文，获取与问题相关的法律依据。"
                "适用于查询某行为是否合法、法律后果、条文含义、适用条件等。"
                "这是最通用的检索工具，大部分问题都应先使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的法律问题或关键词，使用中文表述。例如：'违约金过高可以调整吗'、'盗窃罪的量刑标准'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_tool_list",
            "description": (
                "查询某个事项涉及的法规清单，即该看哪几部法、有哪些相关规定。"
                "当用户问'这涉及哪些法律'、'该查哪部法'、'有哪些法律规定'时使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "technique": {
                        "type": "string",
                        "description": "法律事项或法律关系名称，如'劳动合同解除'、'房屋租赁'、'交通事故赔偿'、'公司股权转让'等",
                    }
                },
                "required": ["technique"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_common_mistakes",
            "description": (
                "查询某个法律事项中常见的误区、风险点和注意事项。"
                "当用户问'有什么风险'、'要注意什么'、'常见的坑'、'容易吃亏的地方'时使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "description": "法律事项名称，如'民间借贷'、'劳动争议'、'婚前财产'、'合同签订'、'遗嘱继承'等",
                    }
                },
                "required": ["step"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════
#  Tool executor
# ═══════════════════════════════════════════════════════════════════


class ToolExecutor:
    """Execute tool calls against the retrieval pipeline.

    Parameters
    ----------
    retriever :
        The retrieval object (HybridRetriever or RerankerProcessor) with a
        ``.query(query_texts, n_results)`` method.
    top_k : int
        Number of results to return per tool call (default 8).

        比问答路径的 12 条少：工具结果会进 ReAct 循环的上下文，多轮累加
        容易挤占预算。
    cache_size : int
        Max number of retrieval results to keep in the LRU cache (default 64).
    """

    def __init__(self, retriever, top_k: int = 8, cache_size: int = 64):
        self.retriever = retriever
        self.top_k = top_k
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.cache_size = cache_size

    # ── dispatch ──────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[str, List[dict]]:
        """Route a tool call to the right handler.

        Returns ``(result_text, sources)`` where ``sources`` is a list of
        ``{"source": <file>, "snippet": <text>}`` for UI citation rendering.
        Results are cached per (tool, arguments) to avoid repeating expensive
        retrieval + reranking during a multi-step ReAct loop.
        """
        key = (tool_name, tuple(sorted(arguments.items())))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        handlers = {
            "search_knowledge": self._search_knowledge,
            "query_tool_list": self._query_tool_list,
            "check_common_mistakes": self._check_common_mistakes,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"[ERROR] 未知工具: {tool_name}", []

        try:
            result = handler(**arguments)
        except Exception as exc:
            return f"[ERROR] 工具执行失败 ({tool_name}): {exc}", []

        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    # ── tool implementations ──────────────────────────────────

    def _search_knowledge(self, query: str) -> tuple[str, List[dict]]:
        """General-purpose statute search."""
        docs, metas = self._query(query)
        if not docs:
            return self._no_hit(query), []
        return self._format_docs("条文检索结果", docs, metas), self._to_sources(docs, metas)

    def _query_tool_list(self, technique: str) -> tuple[str, List[dict]]:
        """Which statutes govern a given matter.

        查询增强串不经过 LLM，但直接决定召回质量：这些词把一条笼统的
        「劳动合同解除」推到具体条文所在的语义空间。换成设备类词汇会
        把召回拉向完全无关的条文。
        """
        augmented_query = f"{technique} 法律规定 条文 适用 依据 相关法规"
        docs, metas = self._query(augmented_query)
        if not docs:
            return self._no_hit(technique), []
        return (
            self._format_docs(f"「{technique}」相关法律依据", docs, metas),
            self._to_sources(docs, metas),
        )

    def _check_common_mistakes(self, step: str) -> tuple[str, List[dict]]:
        """Common misconceptions and legal risks for a given matter."""
        augmented_query = f"{step} 常见误区 法律风险 责任 举证 时效 注意事项"
        docs, metas = self._query(augmented_query)
        if not docs:
            return self._no_hit(step), []
        return (
            self._format_docs(f"「{step}」常见误区与法律风险", docs, metas),
            self._to_sources(docs, metas),
        )

    # ── helpers ───────────────────────────────────────────────

    def _query(self, query: str) -> tuple[List[str], List[Optional[dict]]]:
        """Run one retrieval and split documents from their metadata."""
        results = self.retriever.query(
            query_texts=[query],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]
        metas: List[Optional[dict]] = results.get("metadatas", [[]])[0] or []
        if len(metas) < len(docs):
            metas = metas + [None] * (len(docs) - len(metas))
        return docs, metas

    @staticmethod
    def _to_sources(docs: List[str], metas: List[Optional[dict]]) -> List[dict]:
        """Build citation entries ``{"source", "snippet"}`` for the UI."""
        sources = []
        for doc, meta in zip(docs, metas):
            sources.append({"source": _citation_label(meta), "snippet": doc[:200]})
        return sources

    @staticmethod
    def _format_docs(
        title: str, docs: List[str], metas: List[Optional[dict]]
    ) -> str:
        """Format retrieved statutes into a readable string block."""
        lines = [f"【{title}】共检索到 {len(docs)} 条条文：\n"]
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            lines.append(f"[来源{i}]（{_citation_label(meta)}）{doc}")
        return "\n\n".join(lines)

    @staticmethod
    def _no_hit(topic: str) -> str:
        """Honest no-hit guidance: never fabricate article numbers."""
        return (
            f"（知识库未检索到与「{topic}」相关的条文。"
            "请如实告知用户该问题暂不在知识库覆盖范围内，并建议其到国家法律法规数据库查询；"
            "可以补充一般法律原理，但严禁编造条号、司法解释或案例。）"
        )
