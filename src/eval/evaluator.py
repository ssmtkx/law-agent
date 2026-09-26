"""RAG evaluation metrics: Hit Rate, MRR, Recall, Precision, NDCG.

Works with any retriever that exposes a ``.query(query_texts, n_results)``
method returning ``{"documents": [[doc, ...]], "distances": [[score, ...]],
"metadatas": [[{...}, ...]]}``.

Relevance is decided from **chunk metadata**, not from chunk text.  Each
retrieved chunk carries ``law_title`` / ``article_no``, and each eval item
names the articles that should be cited.  This matters: the previous design
compared retrieved text against a ground-truth text list addressed by
position (``relevant_chunk_indices``), which meant the eval set could only
ever be written by looking at the corpus — the questions were paraphrases of
the answers, so the metric measured nothing.

Coverage is reported separately from retrieval quality.  A gold article that
the corpus does not contain is an automatic miss, and conflating that with
"retrieval failed" is how the old benchmark produced a meaningless 84% / 100%
pair (the same 42 hits over two different denominators).
"""

import json
import math
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.utils.cn_numeral import article_number


# ──────────────────────────────────────────────
#  Text normalisation
# ──────────────────────────────────────────────

# 〔〕 are used in 法释〔2020〕1号 style citations and would otherwise leak
# through into comparisons.
_PUNCT_RE = re.compile(
    r"[\s　，。！？；：、（）()【】\[\]《》〈〉〔〕“”‘’\"'\-—…·]+"
)


def _normalize(text: str) -> str:
    """Strip whitespace and punctuation for tolerant matching."""
    return _PUNCT_RE.sub("", text or "")


def normalize_law(name: str) -> str:
    """Normalise a statute name for comparison.

    Drops the 《》 wrapper, the leading 中华人民共和国 and whitespace so that
    ``《中华人民共和国民法典》``, ``中华人民共和国民法典`` and ``民法典``
    all collapse to the same key.
    """
    n = _normalize(name)
    for prefix in ("中华人民共和国", "中华人民共和国最高人民法院"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def law_matches(a: str, b: str) -> bool:
    """Tolerant statute-name equality.

    Eval sets disagree about how much of a statute's name to write; the
    abbreviation is always a suffix of the full name, so a containment test
    in either direction is the right tolerance.
    """
    na, nb = normalize_law(a), normalize_law(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def article_matches(meta: dict, gold: dict) -> bool:
    """True when a retrieved chunk is the gold article (or one of its 之一 variants)."""
    if not law_matches(meta.get("law_title", ""), gold.get("law", "")):
        return False
    want = article_number(gold.get("article"))
    got = article_number(meta.get("article_no"))
    if want is None or got is None:
        return False
    return want == got


def _match_chunk(meta: Optional[dict], gold_list: List[dict]) -> bool:
    """True when *meta* is one of the gold articles."""
    if not meta:
        return False
    return any(article_matches(meta, gold) for gold in gold_list)


def chunk_key(meta: Optional[dict]) -> Optional[Tuple[str, int]]:
    """Comparable identity of a chunk, for building the corpus coverage set."""
    if not meta:
        return None
    num = article_number(meta.get("article_no"))
    if num is None:
        return None
    return (normalize_law(meta.get("law_title", "")), num)


def gold_keys(item: dict) -> Set[Tuple[str, int]]:
    """All (law, article) keys an eval item expects."""
    keys = set()
    for gold in item.get("relevant") or []:
        num = article_number(gold.get("article"))
        if num is not None:
            keys.add((normalize_law(gold.get("law", "")), num))
    return keys


def sample_dataset(dataset: List[dict], n: int,
                   seed: int = 20260925) -> List[dict]:
    """Uniform random sample of exactly *n* items, reproducible.

    **不要按法规分层。** 分层看起来更"均衡"，但会让样本偏离全集：各法规在
    评测集里的条数相差两个数量级（民法典 1935 条 vs 公司法 51 条），
    从每部法各取等量会把小法超采样几十倍。实测这样采出的 72 条样本上
    Hit@5 只有 0.125，而全量是 0.319 —— 样本比全集难得多，拿来对比或调参
    会得出完全错误的结论。

    均匀随机采样带采样误差，但期望上代表全集，这才是"快速迭代"和
    "扫一遍阈值"需要的东西。固定种子保证同一 n 得到同一批题目。
    """
    if n >= len(dataset):
        return list(dataset)
    rng = random.Random(seed)
    return rng.sample(list(dataset), n)


# ──────────────────────────────────────────────
#  Main evaluator
# ──────────────────────────────────────────────

class RAGEvaluator:
    """Evaluate a retriever against a labelled QA dataset.

    Parameters
    ----------
    retriever :
        An object with a ``.query(query_texts, n_results)`` method returning
        ``documents`` and ``metadatas`` parallel lists.
    corpus_keys :
        Optional set of ``(normalised_law, article_number)`` pairs describing
        what the corpus actually contains.  When supplied, the report splits
        "the corpus doesn't have this article" from "retrieval failed to
        find it" — without it those two very different outcomes look alike.
    """

    def __init__(self, retriever, corpus_keys: Optional[Iterable] = None):
        self.retriever = retriever
        self.corpus_keys: Optional[Set] = set(corpus_keys) if corpus_keys else None

    # ── load / save dataset ───────────────────────────────────

    @staticmethod
    def load_dataset(path: str) -> List[dict]:
        """Load eval items from a JSON file.

        Expected schema per item::

            {"id": "q001", "question": "...", "category": "...",
             "type": "knowledge" | "case_analysis",
             "relevant": [{"law": "...", "article": "第五百七十七条"}]}
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("eval dataset must be a JSON array")
        for item in data:
            # 先识别上一版 schema，给出可操作的报错 —— 否则只会看到
            # "missing required key: relevant"，不知道要改成什么
            if "relevant_chunk_indices" in item:
                raise ValueError(
                    f"eval item {item.get('id')!r} uses the retired "
                    f"'relevant_chunk_indices' schema. Positional indices into "
                    f"a hard-coded chunk list are what made the old benchmark "
                    f"self-referential. Replace with 'relevant': "
                    f'[{{"law": "中华人民共和国民法典", "article": "第五百七十七条"}}]'
                )
            for key in ("id", "question", "relevant"):
                if key not in item:
                    raise ValueError(f"eval item missing required key: {key}")
            if not isinstance(item["relevant"], list):
                raise ValueError(
                    f"eval item {item['id']}: 'relevant' must be a list of "
                    f"{{law, article}} objects"
                )
        return data

    # ── static metric helpers ──────────────────────────────────

    @staticmethod
    def _reciprocal_rank(rel: List[float]) -> float:
        """1 / rank of the first relevant item, or 0 if none."""
        for i, r in enumerate(rel, start=1):
            if r > 0:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _ndcg(rel: List[float]) -> float:
        """Normalised Discounted Cumulative Gain."""
        if not rel:
            return 0.0
        dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(rel, reverse=True)
        idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    # ── evaluation logic ──────────────────────────────────────

    def evaluate(
        self,
        dataset: List[dict],
        k_values: Tuple[int, ...] = (1, 3, 5, 10),
        verbose: bool = False,
        progress_every: int = 25,
    ) -> Dict[str, Any]:
        """Run retrieval evaluation across the dataset.

        Returns a dict with per-k metrics, corpus coverage and per-item results.

        ``progress_every`` prints a rate/ETA line every N items. Without it a
        reranked run is minutes-to-hours of total silence — a single query
        costs ~10s with the cross-encoder, so even 200 items is a long wait
        with no way to tell "working" from "hung".
        """
        per_item: List[dict] = []
        aggregate: Dict[int, Dict[str, float]] = {
            k: {"hit": 0, "reciprocal_rank": 0.0, "recall": 0.0,
                "precision": 0.0, "ndcg": 0.0}
            for k in k_values
        }
        started = time.time()
        n = len(dataset) or 1
        n_in_corpus = 0
        gold_total = gold_covered = 0
        metadata_seen = metadata_slots = 0
        # 拒答计分：expect_refusal 的条目应当召回为空；正常条目召回为空则是误拒
        expected_refusals = correct_refusals = false_refusals = 0

        for idx, item in enumerate(dataset, 1):
            qid = item["id"]
            question = item["question"]
            gold = item.get("relevant") or []
            expect_refusal = bool(item.get("expect_refusal"))

            # ── retrieve ──
            result = self.retriever.query(
                query_texts=[question], n_results=max(k_values)
            )
            retrieved_docs: List[str] = result.get("documents", [[]])[0]
            metas: List[Optional[dict]] = result.get("metadatas", [[]])[0] or []
            if len(metas) < len(retrieved_docs):
                metas = list(metas) + [None] * (len(retrieved_docs) - len(metas))
            # 分母要用**实际召回的条数**，不能用 len(dataset)×max_k。
            # 相关性下限会把候选过滤掉，很多查询返回 0 条或两三条 —— 把
            # 空槽位算成"元数据缺失"，会把"拒答生效"误报成"元数据全丢"
            # （实测这曾让缺失率虚高到 81%，而真实值接近 0）。
            slots_this = min(len(retrieved_docs), max(k_values))
            metadata_seen += sum(1 for m in metas[:slots_this] if m)
            metadata_slots += slots_this

            # 召回为空即"拒答"。这只有在检索层设了相关性下限时才会发生
            refused = not retrieved_docs
            if expect_refusal:
                expected_refusals += 1
                correct_refusals += int(refused)
            elif refused:
                false_refusals += 1

            # ── how much of this item's gold does the corpus actually hold? ──
            keys = gold_keys(item)
            gold_total += len(keys)
            if self.corpus_keys is not None:
                covered = keys & self.corpus_keys
                gold_covered += len(covered)
                in_corpus = bool(keys) and covered == keys
            else:
                in_corpus = True
            if in_corpus:
                n_in_corpus += 1

            item_result = {
                "id": qid,
                "question": question,
                "gold_count": len(gold),
                "in_corpus": in_corpus,
                "retrieved": [d[:80] for d in retrieved_docs],
                "metrics": {},
            }

            for k in k_values:
                rel = [
                    1.0 if _match_chunk(m, gold)
                    else 0.0
                    for m in metas[:k]
                ]
                # 检索结果不足 k 条时补零，否则 precision 会被高估
                rel = rel + [0.0] * (k - len(rel))

                hit = 1.0 if any(rel) else 0.0
                rr = self._reciprocal_rank(rel)
                recall = sum(rel) / len(gold) if gold else 0.0
                precision = sum(rel) / k if k > 0 else 0.0
                ndcg = self._ndcg(rel)

                item_result["metrics"][f"hit@{k}"] = hit
                item_result["metrics"][f"mrr@{k}"] = rr
                item_result["metrics"][f"recall@{k}"] = recall
                item_result["metrics"][f"precision@{k}"] = precision
                item_result["metrics"][f"ndcg@{k}"] = ndcg

                aggregate[k]["hit"] += hit
                aggregate[k]["reciprocal_rank"] += rr
                aggregate[k]["recall"] += recall
                aggregate[k]["precision"] += precision
                aggregate[k]["ndcg"] += ndcg

            per_item.append(item_result)

            if verbose:
                status = "HIT " if any(
                    _match_chunk(m, gold) for m in metas[:5]
                ) else "MISS"
                note = "" if in_corpus else "  (corpus lacks gold)"
                print(f"  [{status}] {qid}: {question[:46]}…{note}")
            elif progress_every and (idx % progress_every == 0
                                     or idx == len(dataset)):
                elapsed = time.time() - started
                rate = idx / max(elapsed, 1e-9)
                eta = (len(dataset) - idx) / rate / 60 if rate > 0 else 0.0
                hits = sum(1 for it in per_item
                           if it["metrics"].get("hit@5", 0) >= 0.5)
                print(f"    … {idx}/{len(dataset)} 条  "
                      f"{rate:.1f} 条/秒  "
                      f"当前 Hit@5={hits/idx:.3f}  "
                      f"预计还需 {eta:.1f} 分钟", flush=True)

        summary: Dict[str, float] = {}
        for k in k_values:
            summary[f"hit_rate@{k}"] = aggregate[k]["hit"] / n
            summary[f"mrr@{k}"] = aggregate[k]["reciprocal_rank"] / n
            summary[f"recall@{k}"] = aggregate[k]["recall"] / n
            summary[f"precision@{k}"] = aggregate[k]["precision"] / n
            summary[f"ndcg@{k}"] = aggregate[k]["ndcg"] / n

        summary["items"] = float(len(dataset))
        summary["items_in_corpus"] = float(n_in_corpus)
        summary["corpus_coverage"] = (
            gold_covered / gold_total if gold_total else 0.0
        )
        summary["metadata_present_rate"] = (
            metadata_seen / metadata_slots if metadata_slots else 1.0
        )
        # 拒答率：分母是"应当拒答"的条目数；误拒率：分母是其余条目数
        summary["expected_refusals"] = float(expected_refusals)
        summary["refusal_rate"] = (
            correct_refusals / expected_refusals if expected_refusals else 0.0
        )
        n_answerable = len(dataset) - expected_refusals
        summary["false_refusal_rate"] = (
            false_refusals / n_answerable if n_answerable else 0.0
        )

        return {"summary": summary, "per_item": per_item}

    # ── printing ──────────────────────────────────────────────

    def print_report(self, eval_result: Dict[str, Any], dataset: List[dict]):
        """Print a human-readable evaluation report.

        实例方法而非静态方法：拒答指标的解读取决于当前检索器有没有相关性下限
        （只有 RerankerProcessor 有），报告需要据此说明该指标是否适用。
        """
        summary = eval_result["summary"]
        per_item = eval_result["per_item"]
        n = int(summary.get("items", len(dataset)))

        print("\n" + "=" * 68)
        print("                  RAG 检索评测报告")
        print("=" * 68)
        print(f"  评测集规模: {n} 条")

        # category breakdown
        cats: Dict[str, int] = {}
        for item in dataset:
            cat = item.get("category", "未知")
            cats[cat] = cats.get(cat, 0) + 1
        print(f"  覆盖类别: {len(cats)} 类")

        # ── corpus coverage — reported BEFORE the metrics, because a metric
        #    computed over items the corpus cannot answer is not a retrieval
        #    score no matter how it is averaged ──
        print("\n" + "-" * 68)
        print("  语料覆盖度（先看这个，再看指标）")
        print("-" * 68)
        cov = summary.get("corpus_coverage", 0.0)
        in_corpus = int(summary.get("items_in_corpus", n))
        print(f"  黄金条文覆盖率: {cov:.1%}"
              f"   （评测集要求的条文里，语料实际收录的比例）")
        print(f"  可答条目: {in_corpus}/{n}"
              f"   （其全部黄金条文都在语料中的条目数）")
        meta_rate = summary.get("metadata_present_rate", 1.0)
        if meta_rate < 1.0:
            print(f"  ⚠ 召回结果元数据缺失率: {1 - meta_rate:.1%}"
                  f"   —— 引用来源会退化成「知识库」，且上面的指标不可信")

        # 拒答能力：只有在检索层设了相关性下限时才有非零值
        n_refuse = int(summary.get("expected_refusals", 0))
        if n_refuse:
            has_floor = hasattr(self.retriever, "min_score")
            print(f"  拒答率: {summary.get('refusal_rate', 0.0):.1%}"
                  f"   （{n_refuse} 条域外问题中，正确返回空的比例）")
            print(f"  误拒率: {summary.get('false_refusal_rate', 0.0):.1%}"
                  f"   （可答条目中被错误拒答的比例）")
            if not has_floor:
                print("  · 未接入精排（--mode hybrid），没有相关性下限，"
                      "拒答率必然为 0 —— 该指标需要 --rerank")
            elif summary.get("refusal_rate", 0.0) == 0.0:
                print("  ⚠ 拒答率 0 —— 相关性下限没有生效，域外问题会被硬答。"
                      "检查 reranker 的 min_score")

        # ── metrics ──
        print("\n" + "-" * 68)
        print("  核心指标（越高越好 →）")
        print("-" * 68)
        # 从 summary 里反推实际算过的 k，不要写死 —— 写死会让 --k 自定义的值
        # 算了却不显示（实测传 --k 1,3,5,8,12,16,20,30 时表格只剩前三列）
        k_values = sorted(
            int(key.split("@")[1]) for key in summary
            if key.startswith("hit_rate@")
        )
        header = f"  {'Metric':<20}"
        for k in k_values:
            header += f" {'@'+str(k):>8}"
        print(header)
        print("  " + "-" * (20 + 9 * len(k_values)))

        for label, key in [
            ("Hit Rate", "hit_rate"), ("MRR", "mrr"), ("Recall", "recall"),
            ("Precision", "precision"), ("NDCG", "ndcg"),
        ]:
            row = f"  {label:<20}"
            for k in k_values:
                row += f" {summary.get(f'{key}@{k}', 0.0):>8.3f}"
            print(row)

        # ── split by whether the corpus could answer at all ──
        if 0 < in_corpus < n:
            print("\n" + "-" * 68)
            print("  Hit@5 拆分：语料可答 vs 语料缺失")
            print("-" * 68)
            for label, want in (("语料可答", True), ("语料缺失", False)):
                items = [it for it in per_item if it["in_corpus"] is want]
                if not items:
                    continue
                hits = sum(1 for it in items
                           if it["metrics"].get("hit@5", 0) >= 0.5)
                print(f"  {label:<10} ({len(items):>3}条)  "
                      f"{hits/len(items):.3f}")
            print("  注：语料缺失的条目是评测集与语料的差值，"
                  "不是检索能力的度量。")

        # ── type breakdown ──
        types: Dict[str, List[dict]] = {}
        for it in per_item:
            t = _find_field(dataset, it["id"], "type", "unknown")
            types.setdefault(t, []).append(it)
        if len(types) > 1:
            print("\n" + "-" * 68)
            print("  分题型 Hit@5")
            print("-" * 68)
            for t, items in sorted(types.items()):
                hits = sum(1 for it in items
                           if it["metrics"].get("hit@5", 0) >= 0.5)
                print(f"  {t:<16} ({len(items):>3}条)  {hits/len(items):.3f}")

        # ── failures ──
        print("\n" + "-" * 68)
        print("  未命中列表（Top-5 无相关条文，且语料确实收录）")
        print("-" * 68)
        failures = [
            it for it in per_item
            if it["metrics"].get("hit@5", 1) < 0.5 and it["in_corpus"]
        ]
        if failures:
            for it in failures[:20]:
                print(f"  [{it['id']}] {it['question'][:56]}")
            if len(failures) > 20:
                print(f"  … 另有 {len(failures) - 20} 条")
            print(f"  共 {len(failures)}/{in_corpus} 条可答条目未命中")
        else:
            print("  可答条目全部命中。")

        print("\n" + "=" * 68 + "\n")


def _find_field(dataset: List[dict], qid: str, field: str, default: Any) -> Any:
    for item in dataset:
        if item["id"] == qid:
            return item.get(field, default)
    return default
