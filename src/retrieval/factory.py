"""Shared retriever factory — used by CLI, Streamlit and the eval scripts.

Eliminates duplicated retriever setup across ``run_agent.py``, ``app.py``
and ``eval_rag.py``.
"""

from __future__ import annotations

import os

from src.domain import DEFAULT_COLLECTION
from src.indexing.indexer import build_hybrid_from_collection
from src.indexing.vector_store import VectorStore
from src.retrieval.query_expander import HyDEExpander
from src.retrieval.reranker import RerankerProcessor

# BM25 索引的磁盘缓存。重建索引时先删掉它，强制从全量数据重新构建。
BM25_CACHE = os.path.join("data", "bm25_index.pkl")

# 查询改写开关。实测（148 条评测题）语义召回 @20 从 0.480 提到 0.757，
# 代价是每次提问多一次 LLM 生成（12–20 秒）。设为 0 可关闭。
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "1") not in ("0", "false", "False")


def build_retriever(
    collection_name: str = DEFAULT_COLLECTION,
    alpha: float = 0.3,
    candidate_pool: int = 20,
    hyde: bool | None = None,
    rerank: bool = False,
) -> tuple[object, int]:
    """Build the retrieval chain and return ``(retriever, chunk_count)``.

    默认链路（生产配置）：**查询改写 → HybridRetriever（BM25 + 语义，RRF 融合）**。

    两个与常见做法不同的选择，都是实测逼出来的：

    * **查询改写默认开**。用户口语与法条书面语的语域差是主要瓶颈：同一批
      148 条评测题上，语义召回 @20 从 0.480 提到 0.757。
    * **精排默认关**。交叉编码器用**用户原问题**打分，而那正是因语域差而
      失效的信号，实测它会把改写带来的收益整个抹平（0.567 → 0.233）；
      它同时是最慢的一环（约 10 秒/条）。需要时传 ``rerank=True``。

    An empty collection is left empty rather than auto-seeded. The previous
    version silently injected a small hand-written sample corpus, which made
    an unindexed deployment look like a working one — answers came from the
    fallback data with no indication anything was missing.
    """
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    count = collection.count()
    if count == 0:
        print("[!] 知识库为空。先构建索引：")
        print("      python scripts/crawl_laws.py     # 采集法规")
        print("      python scripts/build_index.py    # 建立索引")

    hybrid = build_hybrid_from_collection(collection, alpha=alpha)
    if hyde is None:
        hyde = HYDE_ENABLED
    if hyde:
        print("[*] 查询改写已启用（HyDE，每次提问多一次 LLM 调用）")
        hybrid = HyDEExpander(hybrid)

    # 精排**默认关闭**。实测（30 条评测题）：
    #     无改写 + 无精排   Hit@5 = 0.133
    #     有改写 + 无精排   Hit@5 = 0.567   ← 最优
    #     有改写 + 精排     Hit@5 = 0.233   ← 精排把改写带来的收益抹平了
    # 原因是交叉编码器用**用户原问题**打分，而那正是因语域差而失效的信号。
    # 它同时也是整条链路最慢的一环（约 10 秒/条）。需要时用 rerank=True 打开。
    if rerank:
        print("[*] 精排已启用（约 10 秒/条）")
        return RerankerProcessor(hybrid, candidate_pool=candidate_pool), count
    return hybrid, count


def rebuild_index(corpus_dir: str | None = None):
    """Rebuild the full index from the crawled corpus.

    Clears the BM25 cache first so a failed rebuild cannot leave a stale
    cache behind for the next run to trust.
    """
    from src.domain import DEFAULT_CORPUS_DIR
    from src.indexing.indexer import build_law_index

    if os.path.exists(BM25_CACHE):
        os.remove(BM25_CACHE)

    return build_law_index(corpus_dir or DEFAULT_CORPUS_DIR)
