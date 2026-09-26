"""诊断召回：正确条文是在哪一步丢掉的。

调参之前必须先知道瓶颈在哪一段。精排和阈值只能"从候选池里挑"，池子里
没有正确答案时它们无能为力 —— 而此前一直在调这两样。

这个脚本回答两个问题：

1. **候选池召回率**：正确条文在 BM25 / 语义 / 融合 三路各自 top-D 里的
   出现比例。分出"找不到"（召回问题）与"排不上"（排序问题）。
2. **嵌入截断**：有多少 chunk 超过了嵌入模型的最大长度而被静默截断 ——
   被截掉的部分等于没进索引。

Usage::

    python scripts/diagnose_recall.py                    # 默认采样 150 条
    python scripts/diagnose_recall.py --sample 300 --depth 20,50,100,500
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.domain import DEFAULT_COLLECTION  # noqa: E402
from src.eval.evaluator import (RAGEvaluator, _match_chunk, chunk_key,  # noqa: E402
                                gold_keys, sample_dataset)
from src.indexing.bm25_index import BM25Index  # noqa: E402
from src.indexing.vector_store import VectorStore  # noqa: E402
from src.indexing.vector_store import fetch_all
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def recall_at(get_metas, dataset: list[dict], corpus_keys: set,
              depth: int) -> tuple[float, int]:
    """Fraction of answerable, in-corpus items whose gold appears in top-*depth*.

    Only counts items the corpus can actually answer — an item whose gold is
    absent scores zero by construction and would drag every mode down equally,
    hiding the differences we are looking for.
    """
    hit = total = 0
    for item in dataset:
        if item.get("expect_refusal"):
            continue
        gold = item.get("relevant") or []
        if not (gold_keys(item) and gold_keys(item) <= corpus_keys):
            continue                      # 语料没收录，不参与比较
        total += 1
        if any(_match_chunk(m, gold) for m in get_metas(item, depth)):
            hit += 1
    return (hit / total if total else 0.0), total


def check_truncation(model_name: str, sample: int = 3000) -> None:
    """Chunk 长度超出嵌入模型上限的比例 —— 超出部分会被静默截断。"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:                          # noqa: BLE001
        print(f"  [!] 无法加载 tokenizer（{type(exc).__name__}），跳过")
        return

    limit = tok.model_max_length
    if limit > 100000:            # 未设置时 transformers 返回一个哨兵值
        limit = 512
    print(f"  嵌入模型: {model_name}   最大长度 {limit} tokens")

    store = VectorStore()
    collection = store.get_or_create_collection(DEFAULT_COLLECTION)
    docs = fetch_all(collection).get("documents") or []
    if not docs:
        print("  [!] 集合为空")
        return

    step = max(1, len(docs) // sample)
    probe = docs[::step][:sample]
    lengths = [len(tok(d, add_special_tokens=True)["input_ids"]) for d in probe]
    over = sum(1 for n in lengths if n > limit)
    lengths.sort()
    print(f"  抽样 {len(probe):,} 块（共 {len(docs):,}）")
    print(f"    token 长度  中位={lengths[len(lengths)//2]}  "
          f"p90={lengths[int(len(lengths)*0.9)]}  最大={lengths[-1]}")
    print(f"    超出 {limit} 被截断: {over}/{len(probe)} = {over/len(probe):.1%}")
    if over:
        print("    ⚠ 这些块尾部内容没有进入向量索引")


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="诊断召回瓶颈",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset", default="data/eval_law.json")
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--depth", default="20,50,100,500")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--fusion", choices=["rrf", "minmax"], default="rrf",
                    help="融合方式：rrf（默认，按名次）或 minmax（旧版，按分数）")
    args = ap.parse_args()

    depths = [int(x) for x in args.depth.split(",") if x.strip()]
    max_depth = max(depths)

    print("=" * 70)
    print("  召回诊断")
    print("=" * 70)

    dataset = sample_dataset(
        json.loads(Path(args.dataset).read_text(encoding="utf-8")), args.sample)
    print(f"  采样 {len(dataset)} 条")

    store = VectorStore()
    collection = store.get_or_create_collection(args.collection)
    result = fetch_all(collection)
    docs = result.get("documents", []) or []
    ids = result.get("ids", []) or []
    metas = result.get("metadatas", []) or []
    corpus_keys = {k for k in (chunk_key(m) for m in metas) if k}
    print(f"  集合 {collection.count():,} 条")

    if not docs:
        print("[!] 集合为空，先建索引")
        return 1

    # ── 三路召回各自的最深处结果，供不同 D 复用 ──
    print(f"[*] 逐路召回 top-{max_depth}（BM25 建索引中）…")
    t0 = time.time()
    bm25 = BM25Index()
    bm25.build(docs, doc_ids=ids, metadatas=metas)
    hybrid = HybridRetriever(collection, bm25, alpha=0.3, fusion=args.fusion)
    print(f"  融合方式: {args.fusion}")
    print(f"  BM25 就绪 {time.time()-t0:.0f}s")

    cache: dict = {}

    def semantic_metas(item, depth):
        key = ("sem", item["id"])
        if key not in cache:
            r = collection.query(query_texts=[item["question"]],
                                 n_results=max_depth)
            cache[key] = r.get("metadatas", [[]])[0] or []
        return cache[key][:depth]

    def bm25_metas(item, depth):
        key = ("bm25", item["id"])
        if key not in cache:
            hits = bm25.search(item["question"], top_k=max_depth)
            cache[key] = [bm25._doc_metas[i] for i, _ in hits]
        return cache[key][:depth]

    def hybrid_metas(item, depth):
        key = ("hyb", item["id"])
        if key not in cache:
            r = hybrid.query([item["question"]], n_results=max_depth,
                             candidate_pool=max_depth)
            cache[key] = r.get("metadatas", [[]])[0] or []
        return cache[key][:depth]

    print()
    print(f"  {'召回方式':<12}" + "".join(f"{'@'+str(d):>10}" for d in depths))
    print("  " + "-" * (12 + 10 * len(depths)))
    summary = {}
    for label, fn in (("BM25", bm25_metas), ("语义", semantic_metas),
                      ("融合", hybrid_metas)):
        row = []
        for d in depths:
            r, n = recall_at(fn, dataset, corpus_keys, d)
            row.append(r)
        summary[label] = row
        print(f"  {label:<12}" + "".join(f"{v:>10.3f}" for v in row))
        if label == "BM25":
            print(f"  {'  (可答条目数)':<12}{n:>10}")

    print()
    print("  怎么读：")
    print("    · 各列都低 → 召回本身不行，问题在嵌入/分词/chunk 文本")
    print("    · @20 低但 @100/@500 高 → 排序问题，融合或精排能救")
    print("    · 三路差异大 → 能定位是哪一路拖后腿")

    print()
    print("─" * 70)
    print("  嵌入截断检查")
    print("─" * 70)
    check_truncation(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
