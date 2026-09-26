"""建立法规检索索引（Chroma 向量库 + BM25）。

从 ``data/raw/laws/`` 读取采集结果，按条文切分后写入向量库。

Usage::

    python scripts/build_index.py                    # 增量索引全部法条
    python scripts/build_index.py --rebuild          # 清空集合后重建
    python scripts/build_index.py --collection x     # 指定集合名
    python scripts/build_index.py --stats            # 只看当前索引状态

索引只收录**现行有效**（sxx=3）的法规。已废止/已修改的版本仍保留在磁盘上，
留给后续的「法律时效性」特性。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.domain import DEFAULT_COLLECTION, DEFAULT_CORPUS_DIR  # noqa: E402
from src.indexing.indexer import _load_laws, build_law_index  # noqa: E402
from src.indexing.vector_store import VectorStore  # noqa: E402
from src.indexing.vector_store import fetch_all


def show_stats(collection_name: str) -> int:
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)
    result = fetch_all(collection)
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    print("=" * 66)
    print(f"  集合: {collection_name}")
    print("=" * 66)
    print(f"  已索引条文: {len(docs):,}")

    if metas:
        laws = {m.get("law_title", "") for m in metas}
        cats: dict = {}
        for m in metas:
            cat = m.get("category", "未知")
            cats[cat] = cats.get(cat, 0) + 1
        print(f"  覆盖法规: {len(laws):,} 部")
        print(f"  分层分布: {cats}")
        missing_meta = sum(1 for m in metas if not m)
        if missing_meta:
            print(f"  ⚠ 缺元数据的块: {missing_meta}")

    if not docs:
        print()
        print("  [i] 集合为空。先采集法规再建索引：")
        print("      python scripts/crawl_laws.py")
        print("      python scripts/build_index.py")
    print("=" * 66)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="建立法规检索索引",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_DIR,
                    help=f"法规 JSON 目录（默认 {DEFAULT_CORPUS_DIR}）")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Chroma 集合名（默认 {DEFAULT_COLLECTION}）")
    ap.add_argument("--max-chars", type=int, default=800,
                    help="单条 chunk 的字数上限，超出才按句切分（默认 800）")
    ap.add_argument("--rebuild", action="store_true",
                    help="先清空集合再建索引")
    ap.add_argument("--stats", action="store_true",
                    help="只显示当前索引状态")
    args = ap.parse_args()

    if args.stats:
        return show_stats(args.collection)

    laws = _load_laws(args.corpus)
    if not laws:
        print(f"[!] {args.corpus} 下没有法规 JSON。先运行 scripts/crawl_laws.py")
        return 1

    store = VectorStore()
    if args.rebuild:
        try:
            store.client.delete_collection(args.collection)
            print(f"[*] 已删除旧集合 {args.collection}")
        except Exception as exc:                     # noqa: BLE001
            print(f"[*] 无旧集合可删（{type(exc).__name__}）")

    print("=" * 66)
    print("  建立法规索引")
    print("=" * 66)
    started = time.time()

    retriever, chunks = build_law_index(
        corpus_dir=args.corpus,
        collection_name=args.collection,
        max_chars=args.max_chars,
    )
    if retriever is None:
        return 1

    elapsed = time.time() - started
    print()
    if chunks:
        print(f"[OK] 共 {chunks:,} 条，用时 {elapsed/60:.1f} 分钟 "
              f"（{chunks/max(elapsed, 1):.0f} 条/秒）")
    print()
    show_stats(args.collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
