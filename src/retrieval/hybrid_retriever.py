"""Hybrid retrieval: BM25 (keyword) + Chroma (semantic) fused by rank.

Fusion uses **Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR
2009), not score normalisation.

分数归一化的问题在于两路分数本就不可比：余弦相似度落在 [-1, 1]，BM25
无上界；而组内 min-max 还让结果依赖**当前查询候选池的极值** —— 池子
换一批，同一条法条的分数就变。更糟的是它没有下限：若池子里全是无关
条文，"最不无关"的那条照样归一化到 1.0，看起来像强命中。

RRF 只用排名，因此不需要任何校准，且天然对两路的分数量纲免疫：

    score(d) = α/(k + rank_bm25(d)) + (1-α)/(k + rank_dense(d))

``k`` 的经典取值是 60，作用是压低头部名次的权重差，避免某一路的第 1 名
压倒另一路的第 3 名。
"""

import math
from typing import Dict, List, Optional, Tuple

from src.indexing.bm25_index import BM25Index

# Cormack et al. (SIGIR 2009) 的经典常数
RRF_K = 60

# 相关性下限（余弦相似度）。低于它就返回空，调用方的 `if not docs`
# 才会走拒答分支。
#
# **默认关闭（0.0），因为实测它区分不开。** 开启查询改写（HyDE）后，
# 每个问题——包括"明天天气怎么样"——都会被改写成法条腔，于是总能和
# 84k 条里的某一条有中等相似度：
#
#     库内问题  n=141   0.822 – 0.984（中位 0.903）
#     域外问题  n= 12   0.770 – 0.915（中位 0.822）
#
# 两条分布重叠，域外最大值(0.915)高于库内最小值(0.822)。阈值定在 0.75
# 时误拒 0%，但域外问题 100% 漏放。
#
# 也就是说：**改写越有效，"语料能不能答"这个信号就越弱**。拒答需要另
# 想办法（目前靠提示词层约束），不能靠一个相似度阈值。
#
# 注意 Chroma 集合默认是 l2 距离而非 cosine，换算见 _to_similarity。
DEFAULT_MIN_SIMILARITY = 0.0


def _collection_space(collection) -> str:
    """集合的向量距离度量（建集合时由 ``hnsw:space`` 指定，默认 l2）。"""
    try:
        cfg = (collection.metadata or {})
        return str(cfg.get("hnsw:space", "l2")).lower()
    except Exception:                                # noqa: BLE001
        return "l2"


class HybridRetriever:
    """Fuse BM25 keyword scores with semantic vector scores by rank.

    Parameters
    ----------
    collection : Chroma collection
        The semantic vector store (must have .query() method).
    bm25_index : BM25Index
        Pre-built BM25 keyword index.
    alpha : float
        Weight of the BM25 leg in the fusion.  0 = semantic only, 1 = BM25
        only.  Default 0.3 gives moderate keyword lift while favouring
        semantics — 实测 BM25 单独召回明显弱于语义（@500 为 0.574 对
        0.858），所以不宜给它更高权重。
    fusion : str
        ``"rrf"`` (default) or ``"minmax"``.  The latter keeps the old
        per-query min-max score normalisation for A/B comparison.
    """

    def __init__(
        self,
        collection,
        bm25_index: BM25Index,
        alpha: float = 0.3,
        fusion: str = "rrf",
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self.collection = collection
        self.bm25 = bm25_index
        self.alpha = alpha
        self.fusion = fusion
        self.min_similarity = min_similarity
        self._space = _collection_space(collection)

    # ------------------------------------------------------------------
    #  public API  (drop-in replacement for collection.query)
    # ------------------------------------------------------------------

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        candidate_pool: int = 20,
    ) -> dict:
        """Run hybrid search for each query string.

        Returns the same shape as ``collection.query()``:
            {"documents": [[doc, ...]], "distances": [[score, ...]],
             "ids": [[chunk_id, ...]], "metadatas": [[{...}, ...]]}
        """
        all_docs: List[List[str]] = []
        all_dists: List[List[float]] = []
        all_ids: List[List[str]] = []
        all_metas: List[List[Optional[dict]]] = []

        for q in query_texts:
            docs, dists, ids, metas = self._search_one(q, n_results, candidate_pool)
            all_docs.append(docs)
            all_dists.append(dists)
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

    def _search_one(
        self, query: str, n_results: int, candidate_pool: int
    ) -> Tuple[List[str], List[float], List[str], List[Optional[dict]]]:
        """Run BM25 + semantic, fuse, return top N."""

        # ── 1. BM25 leg（结果已按分数降序，下标即名次）──
        bm25_hits = self.bm25.search(query, top_k=candidate_pool)

        # ── 2. semantic leg ──
        chroma_result = self.collection.query(
            query_texts=[query],
            n_results=candidate_pool,
        )
        chroma_docs: List[str] = chroma_result.get("documents", [[]])[0]
        chroma_dists: List[float] = chroma_result.get("distances", [[]])[0]
        chroma_ids: List[str] = chroma_result.get("ids", [[]])[0] or []
        chroma_metas: List[Optional[dict]] = (
            chroma_result.get("metadatas", [[]])[0] or []
        )
        if len(chroma_ids) < len(chroma_docs):
            chroma_ids = chroma_ids + [
                f"chunk_{i}" for i in range(len(chroma_ids), len(chroma_docs))
            ]
        if len(chroma_metas) < len(chroma_docs):
            chroma_metas = chroma_metas + [None] * (len(chroma_docs) - len(chroma_metas))

        # ── 3. fuse ──
        # 以 chunk id 为键：同一块被两路都召回时分数相加，而不是产生两条。
        fused: Dict[str, dict] = {}

        def add(chunk_id: str, text: str, meta: Optional[dict], gain: float) -> None:
            entry = fused.get(chunk_id)
            if entry is None:
                fused[chunk_id] = {"text": text, "meta": meta, "score": gain}
                return
            # BM25 路先建条目时可能没有元数据；语义路带的更完整，补上，
            # 否则引用来源会退化成「知识库」
            if entry["meta"] is None and meta is not None:
                entry["meta"] = meta
            entry["score"] += gain

        if self.fusion == "minmax":
            self._fuse_minmax(add, bm25_hits, chroma_docs, chroma_dists,
                              chroma_ids, chroma_metas)
        else:
            self._fuse_rrf(add, bm25_hits, chroma_docs, chroma_ids, chroma_metas)

        # ── 3.5 相关性下限 ──
        # 用**稠密路的最佳余弦**判断"这个问题语料能不能答"。取最佳而非
        # 平均：一个强命中就足以说明能答，平均会被凑数的候选稀释。
        if self.min_similarity > 0 and chroma_dists:
            best_sim = max(self._to_similarity(d) for d in chroma_dists)
            if best_sim < self.min_similarity:
                return [], [], [], []

        # ── 4. sort and return top N ──
        ranked = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)
        ranked = ranked[:n_results]

        return (
            [item[1]["text"] for item in ranked],
            [item[1]["score"] for item in ranked],
            [item[0] for item in ranked],
            [item[1]["meta"] for item in ranked],
        )

    def _to_similarity(self, distance: float) -> float:
        """Chroma 距离 → 相似度。按集合实际的距离度量换算。

        这个换算必须和建集合时的 ``hnsw:space`` 一致：
        ``cosine`` 下 distance = 1 - cos；``l2`` 下（向量已归一化）
        distance = 2(1 - cos)。两者差一倍，混用会让阈值完全失准。
        """
        if self._space in ("cosine", "ip"):
            return 1.0 - distance
        # l2（Chroma 默认）：嵌入已归一化，故 distance = 2(1 - cos)
        return 1.0 - distance / 2.0

    def _fuse_rrf(self, add, bm25_hits, chroma_docs, chroma_ids,
                  chroma_metas) -> None:
        """Reciprocal Rank Fusion — 只用名次。"""
        for rank, (idx, _score) in enumerate(bm25_hits, start=1):
            add(self.bm25._doc_ids[idx], self.bm25._doc_texts[idx],
                self.bm25._doc_metas[idx] if self.bm25._doc_metas else None,
                self.alpha / (RRF_K + rank))
        for rank, (doc, chunk_id, meta) in enumerate(
                zip(chroma_docs, chroma_ids, chroma_metas), start=1):
            add(chunk_id, doc, meta, (1.0 - self.alpha) / (RRF_K + rank))

    def _fuse_minmax(self, add, bm25_hits, chroma_docs, chroma_dists,
                     chroma_ids, chroma_metas) -> None:
        """Legacy per-query min-max fusion, kept only for A/B comparison."""
        if bm25_hits:
            scores = [s for _, s in bm25_hits]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            for idx, score in bm25_hits:
                add(self.bm25._doc_ids[idx], self.bm25._doc_texts[idx],
                    self.bm25._doc_metas[idx] if self.bm25._doc_metas else None,
                    self.alpha * (score - lo) / span)

        if chroma_docs:
            # Chroma 返回距离（越小越好），先转成相似度
            sims = [1.0 / (1.0 + d) for d in chroma_dists]
            lo, hi = min(sims), max(sims)
            span = (hi - lo) or 1.0
            for doc, sim, chunk_id, meta in zip(
                    chroma_docs, sims, chroma_ids, chroma_metas):
                add(chunk_id, doc, meta, (1.0 - self.alpha) * (sim - lo) / span)
