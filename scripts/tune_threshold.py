"""标定精排的相关性下限（min_score）。

`min_score` 同时控制两件相互拉扯的事：

* 调高 → 域外问题被正确拒答，但**该答的也可能被拒**（误拒）
* 调低 → 该答的都能答，但域外问题会被硬答（幻觉风险）

之前那个 0.5 是用 5 条干净的查询拍出来的，放在真实评测集上误拒率高达 45.7%。
这个脚本给出完整的工作曲线，让阈值有依据可定。

做法：把精排对每个候选的分数**算一次**（下界设为 0，不做任何过滤），
然后离线扫过整个阈值网格 —— 比反复跑评测快几十倍。

Usage::

    python scripts/tune_threshold.py                    # 默认采样 120 条
    python scripts/tune_threshold.py --sample 200
    python scripts/tune_threshold.py --save curve.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.domain import DEFAULT_COLLECTION  # noqa: E402
from src.eval.evaluator import _match_chunk, gold_keys, sample_dataset  # noqa: E402
from src.indexing.bm25_index import BM25Index  # noqa: E402
from src.indexing.vector_store import VectorStore  # noqa: E402
from src.indexing.vector_store import fetch_all
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranker import RerankerProcessor  # noqa: E402

GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def collect_scores(processor: RerankerProcessor, dataset: list[dict],
                   pool: int) -> list[dict]:
    """对每条问题取候选及其精排分数（不做过滤）。"""
    records = []
    started = time.time()
    for i, item in enumerate(dataset, 1):
        # min_score=0 → 所有候选都返回，于是拿到完整的分数分布
        res = processor.query([item["question"]], n_results=pool)
        records.append({
            "item": item,
            "probs": list(res["distances"][0]),
            "metas": list(res["metadatas"][0]),
        })
        if i % 10 == 0 or i == len(dataset):
            el = time.time() - started
            eta = (len(dataset) - i) / (i / max(el, 1e-9)) / 60
            print(f"    … {i}/{len(dataset)} 条  {el:.0f}s  预计还需 {eta:.1f} 分钟",
                  flush=True)
    return records


def evaluate_at(records: list[dict], threshold: float, k: int = 5) -> dict:
    """给定阈值，算一遍关键指标。"""
    hit = false_refusal = 0
    answerable = 0
    kept_total = 0
    refuse_ok = expected_refusals = 0

    for rec in records:
        item = rec["item"]
        gold = item.get("relevant") or []
        expect_refusal = bool(item.get("expect_refusal"))

        kept = [(p, m) for p, m in zip(rec["probs"], rec["metas"])
                if p >= threshold][:k]
        kept_total += len(kept)
        refused = not kept

        if expect_refusal:
            expected_refusals += 1
            refuse_ok += int(refused)
            continue

        answerable += 1
        if refused:
            false_refusal += 1
            continue
        if any(_match_chunk(m, gold) for _p, m in kept):
            hit += 1

    return {
        "threshold": threshold,
        "hit@5": hit / answerable if answerable else 0.0,
        "refusal_rate": refuse_ok / expected_refusals if expected_refusals else 0.0,
        "false_refusal_rate": false_refusal / answerable if answerable else 0.0,
        "avg_kept": kept_total / len(records) if records else 0.0,
        "answerable": answerable,
        "expected_refusals": expected_refusals,
    }


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="标定精排相关性下限",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset", default="data/eval_law.json")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--pool", type=int, default=20,
                    help="每条问题取多少候选（默认 20，与线上一致）")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--save", default=None, help="把曲线存成 JSON")
    args = ap.parse_args()

    print("=" * 68)
    print("  相关性下限标定")
    print("=" * 68)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    dataset = sample_dataset(dataset, args.sample)
    print(f"  采样 {len(dataset)} 条"
          f"（含 {sum(1 for d in dataset if d.get('expect_refusal'))} 条域外问题）")

    store = VectorStore()
    collection = store.get_or_create_collection(args.collection)
    result = fetch_all(collection)
    bm25 = BM25Index()
    bm25.build(result.get("documents", []) or [],
               doc_ids=result.get("ids", []) or [],
               metadatas=result.get("metadatas", []) or [])
    hybrid = HybridRetriever(collection, bm25, alpha=0.3)
    print(f"  集合 {collection.count():,} 条，BM25 {bm25.document_count:,} 条")

    # min_score=0：先不过滤，把完整分数分布取回来
    print("[*] 精排打分中（每条约 10 秒）…")
    processor = RerankerProcessor(hybrid, candidate_pool=args.pool, min_score=0.0)
    records = collect_scores(processor, dataset, args.pool)

    # ── 分数分布 ──
    gold_scores, other_scores = [], []
    for rec in records:
        gold = rec["item"].get("relevant") or []
        for p, m in zip(rec["probs"], rec["metas"]):
            if rec["item"].get("expect_refusal"):
                continue
            (gold_scores if _match_chunk(m, gold) else other_scores).append(p)
    print()
    print("  精排分数分布（用于理解阈值落在哪）")
    if gold_scores:
        gs = sorted(gold_scores)
        print(f"    正确条文  n={len(gs):<5d} "
              f"中位={gs[len(gs)//2]:.3f}  p25={gs[len(gs)//4]:.3f}  "
              f"p75={gs[3*len(gs)//4]:.3f}")
    if other_scores:
        os_ = sorted(other_scores)
        print(f"    其他条文  n={len(os_):<5d} "
              f"中位={os_[len(os_)//2]:.3f}  p75={os_[3*len(os_)//4]:.3f}  "
              f"p90={os_[int(len(os_)*0.9)]:.3f}")

    # ── 阈值曲线 ──
    print()
    print(f"  {'阈值':>6}{'Hit@5':>9}{'拒答率':>9}{'误拒率':>9}{'平均返回':>10}")
    print("  " + "-" * 46)
    curve = []
    for t in GRID:
        row = evaluate_at(records, t)
        curve.append(row)
        print(f"  {t:>6.2f}{row['hit@5']:>9.3f}{row['refusal_rate']:>9.1%}"
              f"{row['false_refusal_rate']:>9.1%}{row['avg_kept']:>10.2f}")

    print()
    print("  怎么读：Hit@5 高、拒答率高、误拒率低，三者不可兼得。")
    print("  误拒率是拿可答问题换来的——它每涨一点，用户就多一批问不到的问题。")

    if args.save:
        Path(args.save).write_text(
            json.dumps({"curve": curve, "sample": len(dataset),
                        "pool": args.pool},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [OK] 曲线已存到 {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
