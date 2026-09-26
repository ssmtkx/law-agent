"""BM25 keyword index with jieba Chinese tokenization and disk persistence."""

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba


# 缓存 schema 版本。序列化状态的结构一旦变更就必须递增 —— 否则旧缓存会被判定为
# "新鲜"而静默加载：实测中它导致 _doc_ids 退化成 chunk_{i}、与向量库的真实 id
# 对不上，融合阶段的去重因此失效（同一个块在 top-5 里出现两次，其中一次没有元数据）。
_SCHEMA_VERSION = 2


def _doc_signature(documents: List[str]) -> str:
    """Deterministic fingerprint of a document list (for BM25 cache freshness).

    The schema version is part of the fingerprint, so a cache written by an
    older code version can never be mistaken for a fresh one.
    """
    import hashlib

    payload = "\x00".join(documents).encode("utf-8", errors="replace")
    return hashlib.sha1(f"v{_SCHEMA_VERSION}\x00".encode() + payload).hexdigest()


class BM25Index:
    """BM25 inverted index for Chinese text retrieval.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b  : float
        Length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # internal state
        self.corpus: List[List[str]] = []   # tokenized documents
        self._doc_texts: List[str] = []     # original strings
        self._doc_ids: List[str] = []       # chunk ids (parallel to _doc_texts)
        self._doc_metas: List[Optional[dict]] = []  # per-chunk metadata
        self._doc_lens: List[int] = []       # token count per doc
        self._avgdl: float = 0.0
        self._df: defaultdict = defaultdict(int)  # document frequency
        self._idf: dict = {}
        self._postings: Dict[str, List[int]] = {}  # term → doc indices
        self._N: int = 0
        self._signature: Optional[str] = None  # fingerprint of _doc_texts

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def build(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Optional[dict]]] = None,
    ):
        """Tokenise ``documents`` and construct the BM25 index.

        Parameters
        ----------
        documents :
            Original document strings.
        doc_ids :
            Parallel list of stable chunk ids, used to align BM25 hits with
            the vector store.  Generated as ``chunk_{i}`` when omitted.
        metadatas :
            Optional parallel list of per-chunk metadata dicts; propagated to
            retrievers so answers can cite real source documents.
        """
        self._doc_texts = documents # 原始文档字符串
        self._doc_ids = list(doc_ids) if doc_ids else [
            f"chunk_{i}" for i in range(len(documents))
        ]
        self._doc_metas = (
            list(metadatas) if metadatas is not None else [None] * len(documents)
        )
        self._signature = _doc_signature(documents)  # 用于缓存失效校验
        # 用 Counter 而不是词表：search() 只需查词频，不必每次线性扫描整篇文档。
        # 8 万块规模下这同时解决了检索速度和 pickle 体积两个问题。
        self.corpus = [Counter(jieba.cut(doc)) for doc in documents]  # 文档的词频表
        self._doc_lens = [sum(c.values()) for c in self.corpus]  # 每篇文档词条数
        self._N = len(self.corpus) # 文档总数
        self._avgdl = sum(self._doc_lens) / self._N if self._N else 0.0 # 所有文档平均词条数

        # document frequencies + inverted postings lists
        self._postings = defaultdict(list)
        self._df.clear()
        for idx, counter in enumerate(self.corpus):
            for t in counter:
                self._df[t] += 1
                self._postings[t].append(idx)

        # pre-compute IDF
        # 计算每个词条的idf
        self._idf.clear()
        for term, freq in self._df.items():
            self._idf[term] = math.log(
                (self._N - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return the top-k *(doc_index, score)* tuples for *query*.

        Uses the inverted postings lists, so only documents containing at
        least one query term are scored (no full-corpus scan).
        """
        if self._N == 0:
            return []

        tokens = list(jieba.cut(query)) # 将query分词
        scores: Dict[int, float] = {}
        seen_terms = set()

        for term in tokens:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for idx in postings:
                tf = self.corpus[idx].get(term, 0)  # 词频（O(1)）
                dl = self._doc_lens[idx]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * num / den

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True) # 按得分降序排列
        return ranked[:top_k]

    # ------------------------------------------------------------------
    #  persistence
    # ------------------------------------------------------------------
    # 进行一个本地保存（持久化）
    def save(self, path: str) -> None:
        """Persist the index to disk so it can be reloaded without re-tokenising.

        Serialises all internal state with :mod:`pickle`.  The resulting file
        can be several MB for large corpora (the full tokenised corpus is
        stored).
        """
        state = {
            "_schema_version": _SCHEMA_VERSION,
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "_doc_texts": self._doc_texts,
            "_doc_ids": self._doc_ids,
            "_doc_metas": self._doc_metas,
            "_doc_lens": self._doc_lens,
            "_avgdl": self._avgdl,
            "_df": dict(self._df),
            "_idf": self._idf,
            "_postings": dict(self._postings),
            "_N": self._N,
            "_signature": self._signature,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    def load(self, path: str) -> bool:
        """Load a previously saved index from disk.

        Returns ``True`` on success, ``False`` if the file does not exist or
        was written by an incompatible schema version.
        """
        p = Path(path)
        if not p.exists():
            return False

        with open(p, "rb") as fh:
            state = pickle.load(fh)

        if state.get("_schema_version") != _SCHEMA_VERSION:
            return False          # 旧结构：视为未命中，交由调用方重建

        self.k1 = state["k1"]
        self.b = state["b"]
        self.corpus = state["corpus"]
        self._doc_texts = state["_doc_texts"]
        # 兼容旧缓存：缺失时按顺序补齐
        self._doc_ids = list(state["_doc_ids"]) if state.get("_doc_ids") else [
            f"chunk_{i}" for i in range(len(self._doc_texts))
        ]
        self._doc_metas = (
            list(state["_doc_metas"])
            if state.get("_doc_metas") is not None
            else [None] * len(self._doc_texts)
        )
        self._doc_lens = state["_doc_lens"]
        self._avgdl = state["_avgdl"]
        self._df = defaultdict(int, state["_df"])
        self._idf = state["_idf"]
        postings = state.get("_postings")
        self._postings = (
            {t: list(idx_list) for t, idx_list in postings.items()}
            if postings is not None
            else self._rebuild_postings()
        )
        self._N = state["_N"]
        self._signature = state.get("_signature")
        return True

    def _rebuild_postings(self) -> Dict[str, List[int]]:
        """Reconstruct postings lists from the corpus (for caches saved pre-acceleration)."""
        postings: Dict[str, List[int]] = defaultdict(list)
        for idx, counter in enumerate(self.corpus):
            for t in counter:
                postings[t].append(idx)
        return dict(postings)

    # ------------------------------------------------------------------
    #  properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return self._N

    @property
    def signature(self) -> Optional[str]:
        return self._signature
