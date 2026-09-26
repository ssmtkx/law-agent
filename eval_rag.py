"""RAG evaluation CLI — 法规检索评测。

评测集必须来自语料之外的独立来源（照着语料写题会得到循环论证的指标）。
真值按 (法名, 条号) 匹配，报告先给语料覆盖度、再给检索指标。

Usage::

    python eval_rag.py                          # 默认 hybrid，K=1,3,5,10
    python eval_rag.py --rerank                 # hybrid + CrossEncoder 精排
    python eval_rag.py --mode semantic          # 仅语义检索（Chroma）
    python eval_rag.py --mode bm25              # 仅 BM25
    python eval_rag.py --k 1,3,5                # 自定义 K
    python eval_rag.py --in-domain              # 只评测语料确实收录的条目
    python eval_rag.py --verbose                # 打印逐条 HIT/MISS
    python eval_rag.py --save results.json      # 保存结果
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Must be set BEFORE any HF imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.domain import DEFAULT_COLLECTION
from src.eval.evaluator import chunk_key, gold_keys, sample_dataset
from src.indexing.vector_store import VectorStore
from src.indexing.vector_store import fetch_all
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerProcessor
from src.eval.evaluator import RAGEvaluator


class _BM25Adapter:
    """Adapt BM25Index.search() → retriever .query() interface.

    Metadata must be returned alongside documents: relevance is decided from
    ``(law_title, article_no)``, so a plain-text-only adapter would make every
    item score zero.
    """

    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index

    def query(self, query_texts, n_results=5, **_kw):
        all_docs, all_scores, all_metas = [], [], []
        for q in query_texts:
            hits = self._bm25.search(q, top_k=n_results)
            docs = [self._bm25._doc_texts[i] for i, _ in hits]
            scores = [float(s) for _, s in hits]
            metas = [self._bm25._doc_metas[i] for i, _ in hits]
            all_docs.append(docs)
            all_scores.append(scores)
            all_metas.append(metas)
        return {"documents": all_docs, "distances": all_scores,
                "metadatas": all_metas}


def build_retriever(mode: str, collection_name: str = DEFAULT_COLLECTION):
    """Build a retriever and the corpus key set used for coverage reporting.

    Modes:
      - ``hybrid``  : BM25 + semantic fusion (default)
      - ``semantic``: Chroma vector search only
      - ``bm25``    : BM25 keyword search only

    Returns ``(retriever, corpus_keys)`` where ``corpus_keys`` is the set of
    ``(law, article)`` pairs the corpus actually contains. The evaluator uses
    it to separate "the corpus doesn't hold this article" from "retrieval
    failed to find it" — without it the two look identical in the metrics.
    """
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    count = collection.count()
    print(f"[*] 集合: {collection_name}（{count:,} 条）")
    if count == 0:
        print("[!] 知识库为空，评测无意义。请先：")
        print("      python scripts/crawl_laws.py")
        print("      python scripts/build_index.py --rebuild")
        return None, set()

    result = fetch_all(collection)
    docs = result.get("documents", []) or []
    ids = result.get("ids", []) or []
    metas = result.get("metadatas", []) or []
    corpus_keys = {k for k in (chunk_key(m) for m in metas) if k}

    if mode == "semantic":
        return collection, corpus_keys

    # ── BM25 ──
    bm25 = BM25Index()
    bm25.build(docs, doc_ids=ids, metadatas=metas)
    print(f"[*] BM25 索引 {bm25.document_count} 条")

    if mode == "bm25":
        return _BM25Adapter(bm25), corpus_keys

    # ── hybrid (default) ──
    hybrid = HybridRetriever(collection, bm25, alpha=0.3)
    print("[*] HybridRetriever (α=0.3)")
    return hybrid, corpus_keys


def main():
    # Windows 控制台默认 GBK，报告里含 ⚠ 等符号会直接抛 UnicodeEncodeError
    if hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="RAG retrieval evaluation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "semantic", "bm25"], default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="在 hybrid 召回之上接入 CrossEncoder 精排",
    )
    parser.add_argument(
        "--k", type=str, default="1,3,5,10",
        help="Comma-separated K values (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/eval_law.json",
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save full results to a JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-item HIT/MISS verdicts",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="精排相关性下限（默认用 RerankerProcessor 的 0.5）",
    )
    parser.add_argument(
        "--hyde-outer", action="store_true",
        help="把 HyDE 包在精排外层：改写文本同时用于召回与打分",
    )
    parser.add_argument(
        "--hyde", action="store_true",
        help="接入 HyDE 查询改写（把口语问题搬到法条语域，实测 @20 +28 点）",
    )
    parser.add_argument(
        "--in-domain", action="store_true",
        help="只评测语料确实收录了黄金条文的条目",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="随机采样 N 条（带精排时单条约 10 秒，全量跑要十几小时）",
    )
    args = parser.parse_args()

    k_values = tuple(int(x.strip()) for x in args.k.split(","))

    # ── 1. build retriever ──
    print("=" * 64)
    print("  RAG 检索评测")
    print("=" * 64)
    print(f"  模式: {args.mode}{' + Reranker' if args.rerank else ''}")
    print(f"  K 值: {k_values}")
    print()

    retriever, corpus_keys = build_retriever(args.mode)
    if retriever is None:
        return

    # ── optional query expansion（必须包在精排**里层**：改写供召回用，
    #    精排仍按用户原问题打分，这样拒答阈值的标定保持有效）──
    if args.hyde:
        from src.retrieval.query_expander import HyDEExpander
        print("[*] 接入 HyDE 查询改写 …")
        retriever = HyDEExpander(retriever, verbose=True)

    # ── optional reranker ──
    if args.rerank:
        print("[*] 接入 CrossEncoder 精排 …")
        kw = {} if args.min_score is None else {"min_score": args.min_score}
        retriever = RerankerProcessor(retriever, **kw)

    # 想看"改写文本同时用于打分"的效果时，把 HyDE 包到精排外面
    if args.hyde_outer:
        from src.retrieval.query_expander import HyDEExpander
        print("[*] HyDE 置于精排外层（改写文本也用于精排打分）…")
        retriever = HyDEExpander(retriever, verbose=True)

    # ── 2. load dataset ──
    print(f"[*] 载入评测集: {args.dataset}")
    dataset = RAGEvaluator.load_dataset(args.dataset)
    print(f"[*] {len(dataset)} 条")

    # ── optional sample ──
    if args.sample and len(dataset) > args.sample:
        from src.eval.evaluator import sample_dataset
        dataset = sample_dataset(dataset, args.sample)
        print(f"[*] 随机采样至 {len(dataset)} 条"
              f"（均匀采样，保持全集分布；分层采样会超采样小法规）")

    # ── optional filter to in-domain only ──
    if args.in_domain:
        dataset = [it for it in dataset
                   if gold_keys(it) & corpus_keys]
        print(f"[*] 过滤后保留语料可覆盖的 {len(dataset)} 条")
        if not dataset:
            print("[!] 过滤后为空 —— 检查评测集与语料是否对得上。")
            return

    # ── 3. evaluate ──
    evaluator = RAGEvaluator(retriever, corpus_keys=corpus_keys)
    print("[*] 评测中 …\n")
    result = evaluator.evaluate(dataset, k_values=k_values, verbose=args.verbose)

    # ── 4. report ──
    evaluator.print_report(result, dataset)

    # ── 5. optional save ──
    if args.save:
        # strip per_item retrieved docs (can be long) for cleaner JSON
        slim = {
            "config": {
                "mode": args.mode,
                "rerank": args.rerank,
                "k_values": list(k_values),
            },
            "summary": result["summary"],
            "per_item": [
                {"id": it["id"], "question": it["question"],
                 "gold_count": it["gold_count"],
                 "in_corpus": it["in_corpus"],
                 "metrics": it["metrics"]}
                for it in result["per_item"]
            ],
        }
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(slim, fh, ensure_ascii=False, indent=2)
        print(f"[OK] Results saved to {args.save}")

    # ── 覆盖率提示 ──
    # 不再设硬编码的达标线。Hit Rate 的分母里混着"语料根本没有的条文"，
    # 一个固定阈值既可能被语料不全拉低，也可能被评测集偏易抬高 ——
    # 先看覆盖度，再看指标，比盯一个数字有意义。
    coverage = result["summary"].get("corpus_coverage", 0.0)
    if coverage < 0.9:
        print(f"  ⚠️  语料覆盖率仅 {coverage:.1%}："
              f"有相当比例的黄金条文不在语料中，")
        print("      此时 Hit Rate 主要反映语料完整度，不是检索能力。"
              "先补采语料再解读指标。")


if __name__ == "__main__":
    main()
