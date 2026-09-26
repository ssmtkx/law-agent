"""Second-stage reranking with BGE cross-encoder + source weighting."""

from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

# ── Authoritative source keywords for boosting ──
#
# 注意：不要放「专家」这类词。上一版包含「专家」，而语料里的问答块格式是
# 「【专家经验问答 #N】」—— 命中是自指的，结果是所有合成内容被统一加权 1.3，
# 把最未经核实的内容系统性地推到最前。关键词必须指向外部权威标识，
# 不能是语料自己给自己贴的标签。
_AUTHORITY_KEYWORDS = [
    "访谈录", "官方", "典籍", "国家标准", "行业标准", "专利", "院士",
]

# 相关性下限：cross-encoder 概率低于该值即丢弃。
#
# 这是让「知识库外拒答」可达的唯一机制 —— 全部候选低于阈值时返回空列表，
# 调用方的 `if not docs` 才会触发拒答分支。
#
# 0.5 是实测标定的，依据是 bge-reranker 在无关输入上的行为：
#   · 真正相关的条文       0.70 – 0.85
#   · 完全域外的提问       0.00 – 0.51（多数恰好 0.0）
# 模型对"毫无关系"的 pair 饱和到 0，而不是负数，所以下限要贴着 0.5 定。
# 改这个值前请先实测分布 —— 之前的 0.35 是拍脑袋定的，配合一个重复
# sigmoid 的 bug，使得任何东西都能过阈值，拒答从未生效过。
DEFAULT_MIN_SCORE = 0.5


class RerankerProcessor:
    """Wraps a retriever and applies cross-encoder reranking.

    The wrapped retriever must expose a ``.query(query_texts, n_results)``
    method returning ``{"documents": [[...]], ...}``.

    Parameters
    ----------
    retriever :
        The first-stage retriever (e.g. HybridRetriever).
    model_name : str
        BGE reranker model id on HuggingFace Hub.
    candidate_pool : int
        How many candidates to fetch from the first stage before reranking.
    source_boost : float
        Multiplier applied to documents from authoritative sources.
    """

    def __init__(
        self,
        retriever,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        candidate_pool: int = 20,
        source_boost: float = 1.3,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.retriever = retriever
        self.candidate_pool = candidate_pool
        self.source_boost = source_boost
        self.min_score = min_score
        self.model = CrossEncoder(model_name) #交叉编码器

    # ------------------------------------------------------------------
    #  public API  (drop-in replacement for retriever.query)
    # ------------------------------------------------------------------

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> dict:
        """Retrieve → rerank → return top-N (with ids and metadata)."""
        all_docs: List[List[str]] = []
        all_dists: List[List[float]] = []
        all_ids: List[List[str]] = []
        all_metas: List[List[Optional[dict]]] = []

        for query in query_texts:
            docs, scores, ids, metas = self._rerank_one(query, n_results)
            all_docs.append(docs)
            all_dists.append(scores)
            all_ids.append(ids)
            all_metas.append(metas)

        return {
            "documents": all_docs,
            "distances": all_dists,
            "ids": all_ids,
            "metadatas": all_metas,
        }

    # ------------------------------------------------------------------
    #  internal
    # ------------------------------------------------------------------

    def _rerank_one(
        self, query: str, n_results: int
    ) -> Tuple[List[str], List[float], List[str], List[Optional[dict]]]:
        # ── 1. fetch candidates from first stage ──
        raw = self.retriever.query(
            query_texts=[query],
            n_results=self.candidate_pool,
        )
        docs: List[str] = raw.get("documents", [[]])[0] # 初召回，，取出当前 query 对应的文档列表
        # 例如 ["文档A", "文档B", ...]

        if not docs:
            return [], [], [], []

        ids: List[str] = raw.get("ids", [[]])[0] or []
        metas: List[Optional[dict]] = raw.get("metadatas", [[]])[0] or []
        if len(ids) < len(docs):
            ids = ids + [f"chunk_{i}" for i in range(len(ids), len(docs))]
        if len(metas) < len(docs):
            metas = metas + [None] * (len(docs) - len(metas))

        # ── 2. rerank with cross-encoder ──
        pairs = [[query, doc] for doc in docs]
        raw_scores: List[float] = self.model.predict(pairs).tolist()

        # ── 3. apply source authority boost ──
        scored = [
            self._score_one(float(s), doc, meta)
            for s, doc, meta in zip(raw_scores, docs, metas)
        ]

        # ── 4. drop anything below the relevance floor ──
        # 这是「知识库外拒答」可达的唯一机制：全部候选低于阈值时返回空，
        # 调用方的 `if not docs` 才会走拒答分支。上一版没有下限，加上组内
        # min-max 归一化，任何问题都会拿到 k 条上下文 —— 拒答分支是死代码。
        ranked = sorted(
            zip(docs, scored, ids, metas),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked = [r for r in ranked if r[1] >= self.min_score][:n_results]

        return (
            [item[0] for item in ranked],
            [item[1] for item in ranked],
            [item[2] for item in ranked],
            [item[3] for item in ranked],
        )

    def _score_one(self, prob: float, doc: str, meta: Optional[dict]) -> float:
        """Apply the source boost to a cross-encoder score.

        ``prob`` is already a probability: ``CrossEncoder.predict`` applies
        its activation (Sigmoid for bge-reranker) by default. Squeezing it
        through a second sigmoid is monotonic, so **ranking survives** — which
        is exactly why the mistake is easy to miss — but it compresses every
        score into [0.5, 1.0], and a relevance floor below 0.5 then lets
        everything through. The refusal path silently stops working while
        retrieval still looks fine.

        The boost multiplies a probability rather than the raw logit: logits
        go negative for poor pairs, and ``-3.0 * 1.3 = -3.9`` would push a
        document further down — the opposite of a boost.
        """
        if self._is_authoritative(doc, meta):
            prob = min(1.0, prob * self.source_boost)
        return prob

    @staticmethod
    def _is_authoritative(doc: str, meta: Optional[dict]) -> bool:
        """Whether a chunk comes from an authoritative source."""
        haystack = doc
        if meta:
            haystack += " " + " ".join(
                str(meta.get(k, "")) for k in ("source", "law_title", "category")
            )
        return any(kw in haystack for kw in _AUTHORITY_KEYWORDS)

