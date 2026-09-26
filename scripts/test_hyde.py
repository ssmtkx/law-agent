"""检验 HyDE（假设性文档嵌入）能否改善召回。

背景：瓶颈已定位为**排序精度**，不是表征能力 —— 正确条文在语义检索的
top-500 里有 86%，但 top-20 只有 48%。落差 38 个点。

最可能的原因：用户问「公司拖欠我三个月工资我能主张什么」（口语），
条文写「用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付
劳动报酬」（书面），两者语域差得太远。

HyDE 的做法是先用 LLM 生成一段"如果法律要回答这个问题，条文大概会怎么
写"的文字，拿它去检索，从而把查询也搬到书面语域。

比较四种查询表示在**同一批题**上的语义召回率：
    raw   原始问题
    hyde  仅假设性条文
    aug   问题 + 假设性条文拼接（单次嵌入）
    both  分别检索后合并（按 chunk id 取并集）

生成结果缓存在 ``data/external/hyde_cache.json``，重跑不再调用 LLM。

Usage::

    python scripts/test_hyde.py --sample 150 --workers 4
    python scripts/test_hyde.py --sample 30 --generate-only   # 先小批试
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.domain import DEFAULT_COLLECTION  # noqa: E402
from src.eval.evaluator import (_match_chunk, chunk_key, gold_keys,  # noqa: E402
                                sample_dataset)
from src.indexing.vector_store import VectorStore, fetch_all  # noqa: E402

CACHE = Path("data/external/hyde_cache.json")

# 提示词的关键是**要求法条语域**。用"不要解释、不要标题"避免模型写成
# 问答式的解释文 —— 那样语域又跑回口语侧，起不到搬移的作用。
PROMPT = """用中文法条的措辞和句式，把下面的问题改写成一段法条正文。只输出法条文字，不要解释、不要标题、不要 markdown，150-250 字。

问题：{question}"""

# 推理模型的 reasoning_content 长度按问题剧烈波动（实测 1900–6900 字），
# 小预算下会把 token 全部耗在推理上、正文为空（finish_reason="length"）。
# 4096 覆盖大部分，空的再用 8192 重试一次。
MAX_TOKENS = 4096
RETRY_MAX_TOKENS = 8192


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                     encoding="utf-8")


def generate_all(dataset: list[dict], workers: int) -> dict:
    """Generate a hypothetical statute passage per question, cached on disk."""
    from openai import OpenAI

    cache = load_cache()
    # 已缓存但为空的一并重试 —— 那是上一轮预算不足留下的
    todo = [it for it in dataset if not cache.get(it["id"])]
    print(f"  缓存命中 {len(dataset)-len(todo)} 条，需生成 {len(todo)} 条")

    if not todo:
        return cache

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=240,
    )
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    lock = threading.Lock()
    done = 0
    started = time.time()

    def call(item: dict, max_tokens: int) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": PROMPT.format(question=item["question"])}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    def one(item: dict) -> tuple[str, str]:
        text = call(item, MAX_TOKENS)
        if not text:                       # 推理吃光预算 → 加大预算重试
            text = call(item, RETRY_MAX_TOKENS)
        return item["id"], text

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for qid, text in pool.map(one, todo):
            with lock:
                cache[qid] = text
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.time() - started
                    eta = (len(todo) - done) / max(done / el, 1e-9) / 60
                    empty = sum(1 for t in cache.values() if not t)
                    print(f"    … {done}/{len(todo)}  {el:.0f}s  "
                          f"预计还需 {eta:.1f} 分钟  空结果 {empty} 条",
                          flush=True)
                    save_cache(cache)      # 定期落盘，中断不丢
    save_cache(cache)
    return cache


def pool_recall(items: list[dict], texts: dict, query_fn, corpus_keys: set,
                depth: int) -> tuple[float, int]:
    hit = total = 0
    for it in items:
        gold = it.get("relevant") or []
        if not (gold_keys(it) and gold_keys(it) <= corpus_keys):
            continue
        total += 1
        metas = query_fn(texts[it["id"]], depth)
        if any(_match_chunk(m, gold) for m in metas):
            hit += 1
    return (hit / total if total else 0.0), total


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="检验 HyDE 对语义召回的改善",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset", default="data/eval_law.json")
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--depth", default="20,50,100")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--generate-only", action="store_true",
                    help="只生成假设性条文，不做检索对比")
    args = ap.parse_args()

    depths = [int(x) for x in args.depth.split(",") if x.strip()]
    max_depth = max(depths)

    print("=" * 70)
    print("  HyDE 对比实验")
    print("=" * 70)

    dataset = sample_dataset(
        json.loads(Path(args.dataset).read_text(encoding="utf-8")), args.sample)
    n_refuse = sum(1 for d in dataset if d.get("expect_refusal"))
    print(f"  采样 {len(dataset)} 条（{n_refuse} 条域外问题不参与对比）")

    print("[*] 生成假设性条文（LLM，结果缓存）…")
    cache = generate_all(dataset, args.workers)
    empty = [k for k, v in cache.items() if not v]
    if empty:
        print(f"  ⚠ {len(empty)} 条生成为空，将退回用原问题")
    if args.generate_only:
        print(f"  [OK] 已缓存 {len(cache)} 条 → {CACHE}")
        return 0

    store = VectorStore()
    collection = store.get_or_create_collection(args.collection)
    fetched = fetch_all(collection)
    corpus_keys = {k for k in (chunk_key(m) for m in fetched["metadatas"]) if k}
    print(f"  集合 {collection.count():,} 条")

    # 查询文本的四种表示
    texts = {}
    for it in dataset:
        q = it["question"]
        h = cache.get(it["id"]) or ""
        texts[it["id"]] = {"raw": q, "hyde": h or q, "aug": f"{q}\n{h}" if h else q}

    def dense(text: str, depth: int) -> list:
        r = collection.query(query_texts=[text], n_results=depth)
        return r.get("metadatas", [[]])[0] or []

    def dense_both(texts_for_item: dict, depth: int) -> list:
        """两路分别检索后按 chunk id 合并（并集）。"""
        out, seen = [], set()
        for key in ("raw", "hyde"):
            r = collection.query(query_texts=[texts_for_item[key]],
                                 n_results=depth)
            for m in (r.get("metadatas", [[]])[0] or []):
                cid = json.dumps(m, sort_keys=True, ensure_ascii=False)
                if cid not in seen:
                    seen.add(cid)
                    out.append(m)
        return out

    print()
    print(f"  {'查询表示':<14}" + "".join(f"{'@'+str(d):>10}" for d in depths))
    print("  " + "-" * (14 + 10 * len(depths)))
    for label in ("raw", "hyde", "aug", "both"):
        fn = (lambda t, d: dense_both(t, d)) if label == "both" \
            else (lambda t, d: dense(t[label], d))
        row, n = [], 0
        for d in depths:
            r, n = pool_recall(dataset, texts, fn, corpus_keys, d)
            row.append(r)
        tag = "  ← 基线" if label == "raw" else ""
        print(f"  {label:<14}" + "".join(f"{v:>10.3f}" for v in row) + tag)

    print(f"  {'(可答条目数)':<14}{n:>10}")
    print()
    print("  怎么读：与 raw 基线比，@20 涨了就说明口语/书面语的语域差确实是瓶颈。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
