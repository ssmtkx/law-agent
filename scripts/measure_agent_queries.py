"""测量查询改写对**工具式查询**是否还有帮助。

背景：Agent 的每个工具调用都会触发一次 HyDE 改写（3.7 秒/次，ReAct 上限 10 轮
→ 最坏累计 37 秒，且串行无法并行）。一个自然的优化是"只改写用户原问题，工具
查询跳过改写"。

但**先测再改**。这个脚本比较两类查询在改写前后的召回率：

    full   完整问题      —— search_knowledge 可能收到的形态
    kw     关键词式       —— 另两个工具实际发出的形态
                            （如 "劳动合同解除 法律规定 条文 适用 依据 相关法规"）

判据：如果 kw 类在改写前后召回率没有差别，那么工具查询跳过改写是安全的。

Usage::

    python scripts/measure_agent_queries.py --sample 60 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import os
import re
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
from src.retrieval.query_expander import HyDEExpander  # noqa: E402

CACHE = Path("data/external/agent_query_cache.json")

# 与 tools.py 中的增强串保持一致，保证测的是真实形态
_TOOL_TEMPLATES = {
    "kw_tool_list": "{core} 法律规定 条文 适用 依据 相关法规",
    "kw_mistakes": "{core} 常见误区 法律风险 责任 举证 时效 注意事项",
}


def extract_core(question: str, k: int = 6) -> str:
    """把自然语言问题压成关键词串 —— 模拟 Agent 提炼技术点后的查询。"""
    try:
        import jieba.analyse
        words = jieba.analyse.extract_tags(question, topK=k)
    except Exception:                                # noqa: BLE001
        words = re.findall(r"[一-龥]{2,4}", question)[:k]
    return " ".join(words) if words else question


def build_queries(dataset: list[dict]) -> dict:
    """每道题产出多种查询形态。"""
    out = {}
    for it in dataset:
        q = it["question"]
        core = extract_core(q)
        forms = {"full": q}
        for name, tpl in _TOOL_TEMPLATES.items():
            forms[name] = tpl.format(core=core)
        out[it["id"]] = forms
    return out


class _Collector:
    """只记录被传入的查询文本，不做检索。"""

    def __init__(self):
        self.seen: list[str] = []

    def query(self, query_texts, n_results=5, **_kw):
        self.seen.extend(query_texts)
        return {"documents": [[]], "metadatas": [[]], "ids": [[]],
                "distances": [[]]}


def expand_all(texts: list[str], workers: int, verbose: bool) -> dict:
    """对一批文本做改写，结果缓存。"""
    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            cache = {}
    todo = [t for t in dict.fromkeys(texts) if not cache.get(t)]
    print(f"  改写缓存命中 {len(set(texts)) - len(todo)} 条，需生成 {len(todo)} 条")
    if not todo:
        return cache

    exp = HyDEExpander(_Collector(), cache_path=None, verbose=verbose)
    lock = threading.Lock()
    done = 0
    started = time.time()

    def one(t: str) -> tuple[str, str]:
        return t, exp._augment(t)

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for text, out in pool.map(one, todo):
            with lock:
                body = out.split("\n", 1)[1] if "\n" in out else ""
                cache[text] = body          # 只存改写出的条文部分
                done += 1
                if done % 20 == 0 or done == len(todo):
                    el = time.time() - started
                    print(f"    … {done}/{len(todo)}  {el:.0f}s", flush=True)
                    CACHE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False),
                                     encoding="utf-8")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="测量改写对工具式查询的效果",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dataset", default="data/eval_law.json")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--depth", default="20,50,100")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = ap.parse_args()

    depths = [int(x) for x in args.depth.split(",") if x.strip()]
    max_depth = max(depths)

    print("=" * 72)
    print("  工具式查询：改写前后对比")
    print("=" * 72)

    dataset = sample_dataset(
        json.loads(Path(args.dataset).read_text(encoding="utf-8")), args.sample)
    dataset = [d for d in dataset if not d.get("expect_refusal")]
    queries = build_queries(dataset)
    print(f"  采样 {len(dataset)} 条")

    print("[*] 生成改写（仅用于对比，不改生产代码）…")
    all_texts = [t for forms in queries.values() for t in forms.values()]
    cache = expand_all(all_texts, args.workers, verbose=False)

    store = VectorStore()
    collection = store.get_or_create_collection(args.collection)
    fetched = fetch_all(collection)
    corpus_keys = {k for k in (chunk_key(m) for m in fetched["metadatas"]) if k}
    print(f"  集合 {collection.count():,} 条")

    def recall(forms_key: str, use_hyde: bool, depth: int) -> tuple[float, int]:
        hit = total = 0
        for it in dataset:
            gold = it.get("relevant") or []
            if not (gold_keys(it) and gold_keys(it) <= corpus_keys):
                continue
            total += 1
            text = queries[it["id"]][forms_key]
            if use_hyde and cache.get(text):
                text = f"{text}\n{cache[text]}"
            r = collection.query(query_texts=[text], n_results=depth)
            metas = r.get("metadatas", [[]])[0] or []
            if any(_match_chunk(m, gold) for m in metas):
                hit += 1
        return (hit / total if total else 0.0), total

    print()
    print(f"  {'查询形态':<26}" + "".join(f"{'@'+str(d):>11}" for d in depths)
          + f"{'改写增益(@20)':>14}")
    print("  " + "-" * (26 + 11 * len(depths) + 14))
    for key, label in (("full", "完整问题"),
                       ("kw_tool_list", "关键词式（法条清单）"),
                       ("kw_mistakes", "关键词式（风险误区）")):
        before, after, n = [], [], 0
        for d in depths:
            rb, n = recall(key, False, d)
            ra, _ = recall(key, True, d)
            before.append(rb)
            after.append(ra)
        gain = after[0] - before[0]
        row = f"  {label:<26}"
        for b, a in zip(before, after):
            row += f"{b:>4.2f}→{a:<4.2f} "
        row += f"{gain:>+12.3f}"
        print(row)
    print(f"  {'(可答条目数)':<26}{n:>9}")
    print()
    print("  怎么读：")
    print("    · 关键词式那一行增益接近 0 → 工具查询跳过改写是安全的")
    print("    · 增益明显为正 → 不能跳过，那 37 秒得留着")
    return 0


if __name__ == "__main__":
    sys.exit(main())
