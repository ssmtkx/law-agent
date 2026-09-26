"""Build the statute index: crawl output → article chunks → Chroma + BM25.

Two entry points:

* :func:`build_law_index` — read ``data/raw/laws/**/*.json``, chunk by
  article, write to Chroma, rebuild BM25 from the whole collection.
* :func:`build_hybrid_from_collection` — attach a BM25 index to an existing
  Chroma collection, using the on-disk cache when it is still valid.
"""

from pathlib import Path

import json
import os
import time

from src.domain import DEFAULT_COLLECTION, DEFAULT_CORPUS_DIR
from src.indexing.bm25_index import BM25Index, _doc_signature
from src.indexing.vector_store import VectorStore
from src.indexing.vector_store import fetch_all
from src.ingestion.law_parser import LawArticleSplitter
from src.retrieval.hybrid_retriever import HybridRetriever

BM25_CACHE_PATH = os.path.join("data", "bm25_index.pkl")

# 索引哪些效力状态的法规。
#
# 只收现行有效 + 尚未生效：
#   · 已废止 / 已修改 —— 同一部法会有多个版本并存（实测民事诉讼法就是），
#     都索引会让同一条号出现两次、互相竞争排名。旧版本保留在磁盘上，
#     留给后续的「法律时效性」特性。
#   · **尚未生效要收** —— 用户完全可能问「医保法怎么规定的」，排除掉就答不了。
#     代价是可能拿还没施行的法去回答当下的问题，所以切分时会在这类条文的
#     正文头部标注「尚未生效」，并在提示词里要求回答时说明施行日期。
INDEXED_STATUS = (3, 4)


def _load_laws(corpus_dir: str) -> list[dict]:
    """All statute JSON records on disk."""
    root = Path(corpus_dir)
    if not root.exists():
        return []
    laws = []
    for path in sorted(root.glob("*/*.json")):
        try:
            laws.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:                     # noqa: BLE001
            print(f"  [!] 跳过损坏的记录 {path.name}: {exc}")
    return laws


def _dedupe(laws: list[dict]) -> tuple[list[dict], int]:
    """Drop duplicate records for the same statute version.

    实测源库里存在同标题、同效力状态、条文完全相同但 ``bbbs`` 不同的条目
    （如公司法、村民委员会组织法各有两份）。索引以 bbbs 为 source，
    两份都会入索引，产生重复块互相竞争排名。按 (标题, 效力状态) 去重，
    保留条文更多的那份。
    """
    best: dict[tuple, dict] = {}
    for law in laws:
        key = (law.get("title", ""), law.get("sxx"))
        prev = best.get(key)
        if prev is None or len(law.get("articles") or []) > len(
                prev.get("articles") or []):
            best[key] = law
    removed = len(laws) - len(best)
    return list(best.values()), removed


def build_law_index(
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    max_chars: int = 800,
) -> tuple[HybridRetriever, int] | tuple[None, int]:
    """Index the crawled statutes. Returns ``(retriever, chunk_count)``.

    Returns ``(None, 0)`` when there is no corpus on disk.
    """
    laws = _load_laws(corpus_dir)
    if not laws:
        print(f"[!] 未在 {corpus_dir} 找到法规 JSON。")
        print("[!] 先运行: python scripts/crawl_laws.py")
        return None, 0

    laws, removed = _dedupe(laws)
    if removed:
        print(f"[*] 去重：跳过 {removed} 份重复条目")

    current = [law for law in laws if law.get("sxx") in INDEXED_STATUS]
    skipped_version = len(laws) - len(current)
    unparsed = [law for law in current if not law.get("articles")]
    indexable = [law for law in current if law.get("articles")]

    n_effective = sum(1 for law in current if law.get("sxx") == 3)
    n_pending = len(current) - n_effective
    print(f"[*] 磁盘上 {len(laws)} 部法规（收录 {len(current)} 部："
          f"现行有效 {n_effective}，尚未生效 {n_pending}；"
          f"已废止/已修改 {skipped_version} 部未索引）")

    splitter = LawArticleSplitter(max_chars=max_chars)
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    total_chunks = 0
    no_article_titles = []

    # 嵌入是这一步的全部耗时（bge-small-zh 在 CPU 上十几到几十条/秒），
    # 而 Chroma 的 add 不输出任何进度 —— 面对 8 万条的规模，没有进度就
    # 无法判断是在跑还是卡死，也估不出剩余时间。
    started = time.time()
    print("[*] 开始嵌入 …（每条法条一个 chunk，CPU 推理）")

    for i, law in enumerate(indexable, 1):
        chunks = splitter.split(law)
        if not chunks:
            continue
        # 以 bbbs 为 source，同一部法重复索引时先清旧块（见 add_documents）
        store.add_documents(
            collection,
            [c["text"] for c in chunks],
            [c["metadata"] for c in chunks],
            source=law.get("bbbs") or law.get("title", ""),
        )
        total_chunks += len(chunks)

        if i % 25 == 0 or i == len(indexable):
            elapsed = time.time() - started
            rate = total_chunks / max(elapsed, 1e-9)          # 条/秒
            eta_min = (elapsed / i) * (len(indexable) - i) / 60
            print(f"    … {i}/{len(indexable)} 部  "
                  f"{total_chunks:,} 条  "
                  f"{rate:.0f} 条/秒  "
                  f"预计还需 {eta_min:.1f} 分钟", flush=True)

    print(f"[OK] Chroma 写入 {total_chunks} 条（{len(indexable)} 部法规），"
          f"用时 {(time.time()-started)/60:.1f} 分钟")
    if unparsed:
        no_article_titles = [law.get("title", "")[:30] for law in unparsed[:5]]
        print(f"[!] {len(unparsed)} 部法规无可切分条文"
              f"（修正案/法律解释类），未索引；例如: {no_article_titles}")

    bm25 = _build_bm25_from_collection(collection)
    bm25.save(BM25_CACHE_PATH)
    print(f"[*] BM25 索引已保存到 {BM25_CACHE_PATH}")

    return HybridRetriever(collection, bm25, alpha=0.3), total_chunks


def _build_bm25_from_collection(collection) -> BM25Index:
    """BM25 over everything currently in the collection.

    Built from the collection rather than from the loop above so the keyword
    index always covers exactly what the vector store holds.
    """
    result = fetch_all(collection)
    docs = result.get("documents", []) or []
    ids = result.get("ids", []) or []
    metas = result.get("metadatas", []) or []

    bm25 = BM25Index()
    bm25.build(docs, doc_ids=ids, metadatas=metas)
    print(f"[OK] BM25 索引 {bm25.document_count} 条")
    return bm25


def build_hybrid_from_collection(
    collection,
    alpha: float = 0.3,
    collection_name: str = DEFAULT_COLLECTION,
) -> HybridRetriever:
    """Attach a BM25 index to an existing Chroma collection.

    Loads the persisted index when it is still valid; rebuilds and saves
    otherwise.  Validity covers both the documents and the cache schema —
    a cache written by older code is rejected outright, because silently
    accepting it loses the chunk ids that hybrid fusion dedupes on.
    """
    bm25 = BM25Index()

    if os.path.exists(BM25_CACHE_PATH) and bm25.load(BM25_CACHE_PATH):
        try:
            docs = fetch_all(collection).get("documents", []) or []
        except Exception:                            # noqa: BLE001
            docs = []
        if _doc_signature(docs) == bm25.signature:
            print(f"[*] BM25 索引从缓存加载（{bm25.document_count} 条）")
            return HybridRetriever(collection, bm25, alpha=alpha)
        print("[!] BM25 缓存与当前集合不一致，重建")
    elif os.path.exists(BM25_CACHE_PATH):
        print("[!] BM25 缓存 schema 过期（旧版本写入），重建")

    # ── rebuild from Chroma ──
    try:
        result = fetch_all(collection)
        docs = result.get("documents", []) or []
        ids = result.get("ids", []) or []
        metas = result.get("metadatas", []) or []
    except Exception:                                # noqa: BLE001
        docs, ids, metas = [], [], []

    if not docs:
        print("[!] 集合为空，BM25 索引也将为空")
        bm25.build([])
        return HybridRetriever(collection, bm25, alpha=alpha)

    print(f"[*] 从集合重建 BM25 索引（{len(docs)} 条）…")
    bm25.build(docs, doc_ids=ids, metadatas=metas)
    bm25.save(BM25_CACHE_PATH)
    print(f"[OK] BM25 索引 {bm25.document_count} 条，已保存")
    return HybridRetriever(collection, bm25, alpha=alpha)
