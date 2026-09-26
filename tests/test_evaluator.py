"""Unit tests for the eval matcher and evaluator.

Relevance is decided from chunk **metadata** (``law_title`` / ``article_no``)
against an eval item's ``relevant`` list — not from chunk text addressed by
position.  These tests pin that contract, including the regression that a
chunk with missing metadata must count as a miss rather than silently
matching on empty strings.
"""

import unittest

from src.eval.evaluator import (
    RAGEvaluator, _match_chunk, _normalize, article_matches, article_number,
    chunk_key, gold_keys, law_matches, normalize_law,
)

CIVIL_CODE = "中华人民共和国民法典"
CRIMINAL_LAW = "中华人民共和国刑法"


class NormalizeTest(unittest.TestCase):
    def test_strips_whitespace_and_punctuation(self):
        self.assertEqual(_normalize("A，B。 C！D"), "ABCD")
        self.assertEqual(_normalize("（甲）《乙》·丙—丁"), "甲乙丙丁")

    def test_strips_bracket_glyphs_used_in_legal_citations(self):
        # 〔〕 出现在法释〔2020〕1号 这类引用里，不处理会漏进比较
        self.assertEqual(_normalize("法释〔2020〕1号"), "法释20201号")


class LawNameTest(unittest.TestCase):
    def test_normalize_drops_country_prefix_and_book_marks(self):
        self.assertEqual(normalize_law("《中华人民共和国民法典》"), "民法典")
        self.assertEqual(normalize_law(CIVIL_CODE), "民法典")
        self.assertEqual(normalize_law("民法典"), "民法典")

    def test_full_and_short_names_match(self):
        self.assertTrue(law_matches(CIVIL_CODE, "民法典"))
        self.assertTrue(law_matches("《中华人民共和国民法典》", CIVIL_CODE))

    def test_different_statutes_do_not_match(self):
        self.assertFalse(law_matches(CIVIL_CODE, CRIMINAL_LAW))
        # 刑法 vs 刑事诉讼法：后者不以「刑法」结尾，但包含关系会误判，
        # 所以这里断言的是 *子串* 方向也要成立时才命中
        self.assertTrue(law_matches("中华人民共和国刑法", "刑法"))

    def test_empty_names_never_match(self):
        self.assertFalse(law_matches("", CIVIL_CODE))
        self.assertFalse(law_matches(CIVIL_CODE, ""))


class ArticleNumberTest(unittest.TestCase):
    def test_chinese_numeral_to_int(self):
        self.assertEqual(article_number("第五百七十七条"), 577)
        self.assertEqual(article_number("第一条"), 1)
        self.assertEqual(article_number("第十条"), 10)
        self.assertEqual(article_number("第一千二百六十条"), 1260)

    def test_arabic_digits_accepted(self):
        self.assertEqual(article_number(577), 577)
        self.assertEqual(article_number("577"), 577)
        self.assertEqual(article_number("第577条"), 577)

    def test_sub_article_keeps_base_number(self):
        # 「第十七条之一」属于第十七条，评测时按基础条号对齐
        self.assertEqual(article_number("第十七条之一"), 17)

    def test_unparseable_returns_none(self):
        self.assertIsNone(article_number(""))
        self.assertIsNone(article_number(None))


class ArticleMatchTest(unittest.TestCase):
    def test_same_law_and_article(self):
        meta = {"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}
        self.assertTrue(article_matches(meta, {"law": CIVIL_CODE, "article": "第五百七十七条"}))

    def test_short_law_name_still_matches(self):
        meta = {"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}
        self.assertTrue(article_matches(meta, {"law": "民法典", "article": "577"}))

    def test_wrong_article_misses(self):
        meta = {"law_title": CIVIL_CODE, "article_no": "第五百七十八条"}
        self.assertFalse(article_matches(meta, {"law": CIVIL_CODE, "article": "第五百七十七条"}))

    def test_same_article_number_in_another_statute_misses(self):
        meta = {"law_title": CRIMINAL_LAW, "article_no": "第一条"}
        self.assertFalse(article_matches(meta, {"law": CIVIL_CODE, "article": "第一条"}))


class MatchChunkTest(unittest.TestCase):
    GOLD = [{"law": CIVIL_CODE, "article": "第五百七十七条"}]

    def test_metadata_match_hits(self):
        meta = {"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}
        self.assertTrue(_match_chunk(meta, self.GOLD))

    def test_missing_metadata_misses(self):
        # 回归：BM25 缓存陈旧时元数据会退化成 None。旧实现按文本比对，
        # 这种情况下会拿空串去比，行为不可预期；新实现必须判为未命中。
        self.assertFalse(_match_chunk(None, self.GOLD))
        self.assertFalse(_match_chunk({}, self.GOLD))

    def test_unrelated_chunk_misses(self):
        meta = {"law_title": "中华人民共和国海商法", "article_no": "第一百零二条"}
        self.assertFalse(_match_chunk(meta, self.GOLD))


class KeyTest(unittest.TestCase):
    def test_chunk_key_normalises_law(self):
        self.assertEqual(
            chunk_key({"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}),
            ("民法典", 577),
        )

    def test_gold_keys_covers_all_articles(self):
        item = {"relevant": [
            {"law": CIVIL_CODE, "article": "第五百七十七条"},
            {"law": "民法典", "article": "577"},
            {"law": CRIMINAL_LAW, "article": "第二百六十四条"},
        ]}
        self.assertEqual(gold_keys(item), {("民法典", 577), ("刑法", 264)})

    def test_chunk_key_none_without_metadata(self):
        self.assertIsNone(chunk_key(None))


class _FakeRetriever:
    """Returns preset (docs, metas); ignores the query."""

    def __init__(self, docs, metas):
        self._docs, self._metas = docs, metas

    def query(self, query_texts, n_results=5, **_kw):
        return {"documents": [self._docs[:n_results]],
                "metadatas": [self._metas[:n_results]]}


class EvaluatorTest(unittest.TestCase):
    DATASET = [
        {"id": "q1", "question": "违约了要承担什么责任？", "category": "合同",
         "type": "knowledge",
         "relevant": [{"law": CIVIL_CODE, "article": "第五百七十七条"}]},
        {"id": "q2", "question": "语料里没有的条文？", "category": "其他",
         "type": "knowledge",
         "relevant": [{"law": "中华人民共和国某未收录法", "article": "第一条"}]},
    ]

    def test_hit_and_coverage_reported_separately(self):
        docs = ["d1", "d2"]
        metas = [{"law_title": CIVIL_CODE, "article_no": "第五百七十七条"},
                 {"law_title": CRIMINAL_LAW, "article_no": "第一条"}]
        evaluator = RAGEvaluator(
            _FakeRetriever(docs, metas),
            corpus_keys={("民法典", 577)},   # q2 的黄金条文不在语料里
        )
        out = evaluator.evaluate(self.DATASET, k_values=(1, 5), progress_every=0)
        s = out["summary"]

        self.assertEqual(s["hit_rate@1"], 0.5)          # q1 命中，q2 不可能命中
        self.assertEqual(s["corpus_coverage"], 0.5)     # 2 条要求里收录了 1 条
        self.assertEqual(s["items_in_corpus"], 1.0)
        # q2 未被计入「可答条目」，所以不污染检索质量结论
        self.assertTrue([it for it in out["per_item"] if it["id"] == "q2"][0]["in_corpus"] is False)

    def test_metadata_loss_drops_hit_rate(self):
        """元数据全丢时必须拉低指标，而不是靠空串比对蒙混过关。"""
        docs = ["d1", "d2"]
        evaluator = RAGEvaluator(_FakeRetriever(docs, [None, None]))
        out = evaluator.evaluate(self.DATASET, k_values=(1, 5), progress_every=0)
        self.assertEqual(out["summary"]["hit_rate@5"], 0.0)
        self.assertEqual(out["summary"]["metadata_present_rate"], 0.0)

    def test_fewer_results_than_k_does_not_inflate_precision(self):
        evaluator = RAGEvaluator(_FakeRetriever(
            ["d1"], [{"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}]))
        out = evaluator.evaluate(self.DATASET[:1], k_values=(5,), progress_every=0)
        # 只返回 1 条 → precision@5 应为 1/5，而不是 1/1
        self.assertAlmostEqual(out["summary"]["precision@5"], 0.2)


class RefusalTest(unittest.TestCase):
    """拒答计分：这是 reranker 相关性下限是否生效的直接度量。"""

    DATASET = [
        {"id": "q1", "question": "违约了要承担什么责任？",
         "relevant": [{"law": CIVIL_CODE, "article": "第五百七十七条"}]},
        {"id": "q2", "question": "GDPR 对数据跨境传输有什么要求？",
         "relevant": [], "expect_refusal": True},
        {"id": "q3", "question": "美国专利法的新颖性标准？",
         "relevant": [], "expect_refusal": True},
    ]

    def test_all_refused_scores_full(self):
        evaluator = RAGEvaluator(_FakeRetriever([], []))
        out = evaluator.evaluate(self.DATASET, k_values=(5,), progress_every=0)
        # 两条域外问题都被拒，可答条目也被误拒
        self.assertEqual(out["summary"]["refusal_rate"], 1.0)
        self.assertEqual(out["summary"]["false_refusal_rate"], 1.0)

    def test_no_refusal_scores_zero(self):
        docs = ["d1"]
        metas = [{"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}]
        evaluator = RAGEvaluator(_FakeRetriever(docs, metas))
        out = evaluator.evaluate(self.DATASET, k_values=(5,), progress_every=0)
        # 域外问题也拿到了候选 → 拒答率为 0（正是阈值失效的症状）
        self.assertEqual(out["summary"]["refusal_rate"], 0.0)
        self.assertEqual(out["summary"]["false_refusal_rate"], 0.0)
        self.assertEqual(out["summary"]["expected_refusals"], 2.0)

    def test_metrics_ignore_refusal_items(self):
        """域外条目 relevant 为空，不应把 Hit Rate 拉成 0 而掩盖真实表现。"""
        docs = ["d1"]
        metas = [{"law_title": CIVIL_CODE, "article_no": "第五百七十七条"}]
        evaluator = RAGEvaluator(_FakeRetriever(docs, metas))
        out = evaluator.evaluate(self.DATASET, k_values=(5,), progress_every=0)
        # 3 条里只有 q1 能命中 → hit_rate@5 = 1/3，但这是"命中率"而非"能力"
        self.assertAlmostEqual(out["summary"]["hit_rate@5"], 1 / 3)


class LoadDatasetTest(unittest.TestCase):
    def test_rejects_legacy_positional_schema(self):
        import json
        import os
        import tempfile

        legacy = [{"id": "q1", "question": "…", "relevant_chunk_indices": [0]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(legacy, fh)
            with self.assertRaises(ValueError) as ctx:
                RAGEvaluator.load_dataset(path)
        self.assertIn("relevant_chunk_indices", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class AgentCitationFormatTest(unittest.TestCase):
    """Agent 评测的引用计分要接受两种格式。

    提示词要求写 `[来源N]`，但模型偶尔直接写 `[来源：关税法第五条]` ——
    后者**信息量更大**。旧实现只匹配编号正则，导致一条完全正确的回答在
    引用维度上被判 0 分（实测发生在 a020 上）。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "_eval_agent", Path("eval_agent.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.score = staticmethod(mod.score_sources)

    def test_numbered_format(self):
        score, detail = self.score("……[来源1]……[来源2]")
        self.assertEqual(score, 25.0, detail)

    def test_named_format_also_counts(self):
        score, detail = self.score("《关税法》第五条……[来源：关税法第五条]")
        self.assertGreater(score, 0.0, "具名格式不应被判为没有引用")
        self.assertEqual(score, 12.5, detail)

    def test_mixed_formats_add_up(self):
        score, detail = self.score("[来源1] [来源：民法典第五百八十五条]")
        self.assertEqual(score, 25.0, detail)

    def test_no_citation_scores_zero(self):
        score, _ = self.score("这段回答没有任何来源标注。")
        self.assertEqual(score, 0.0)
