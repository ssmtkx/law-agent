"""Unit tests for BM25Index (postings acceleration, ids/metadata, persistence)."""

import os
import pickle
import tempfile
import unittest

from src.indexing.bm25_index import BM25Index

DOCS = [
    "《中华人民共和国民法典》第五百七十七条规定，当事人一方不履行合同义务的，应当承担违约责任。",
    "《中华人民共和国刑法》第二百六十四条规定，盗窃公私财物数额较大的，处三年以下有期徒刑。",
    "《中华人民共和国行政处罚法》第五条规定，行政处罚遵循公正、公开的原则。",
]
IDS = ["d0", "d1", "d2"]
METAS = [
    {"law_title": "中华人民共和国民法典", "article_no": "第五百七十七条"},
    {"law_title": "中华人民共和国刑法", "article_no": "第二百六十四条"},
    {"law_title": "中华人民共和国行政处罚法", "article_no": "第五条"},
]


class BM25BuildTest(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25Index()
        self.bm25.build(DOCS, doc_ids=IDS, metadatas=METAS)

    def test_doc_ids_and_metas_stored(self):
        self.assertEqual(self.bm25._doc_ids, IDS)
        self.assertEqual(self.bm25._doc_metas, METAS)

    def test_default_ids_generated(self):
        bm25 = BM25Index()
        bm25.build(DOCS)
        self.assertEqual(bm25._doc_ids, ["chunk_0", "chunk_1", "chunk_2"])

    def test_postings_only_hit_relevant_docs(self):
        hits = dict(self.bm25.search("盗窃"))
        self.assertIn(1, hits)
        self.assertNotIn(0, hits)
        self.assertNotIn(2, hits)

    def test_unknown_term_returns_empty(self):
        # 用纯 ASCII 无义词，避免 jieba 把短语拆出文档中存在的常见词
        self.assertEqual(self.bm25.search("zzqxvw"), [])

    def test_results_sorted_desc(self):
        scores = [score for _, score in self.bm25.search("违约")]
        self.assertEqual(scores, sorted(scores, reverse=True))


class BM25PersistenceTest(unittest.TestCase):
    def test_save_load_roundtrip(self):
        bm25 = BM25Index()
        bm25.build(DOCS, doc_ids=IDS, metadatas=METAS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bm25.pkl")
            bm25.save(path)
            loaded = BM25Index()
            self.assertTrue(loaded.load(path))
            self.assertEqual(loaded.signature, bm25.signature)
            self.assertEqual(loaded._doc_ids, IDS)
            self.assertEqual(loaded._doc_metas, METAS)
            self.assertEqual(loaded.search("盗窃"), bm25.search("盗窃"))

    def test_legacy_cache_without_ids_and_postings(self):
        """旧 schema 的缓存必须被拒绝，而不是带着退化字段被静默加载。

        这里断言的是修复后的契约。旧实现接受这种缓存，并把 ``_doc_ids``
        退化成 ``chunk_{i}``、元数据全丢 —— 而向量库里的真实 id 是
        ``{source}_chunk_{i}``，两者永不相等，导致融合阶段的去重前提失效：
        同一个块会在 top-k 里出现两次（其中一次没有元数据），引用来源随之
        退化成「知识库」。拒绝并重建才是正确行为。
        """
        bm25 = BM25Index()
        bm25.build(DOCS)
        state = {          # 刻意不带 _schema_version / _doc_ids / _postings
            "k1": bm25.k1,
            "b": bm25.b,
            "corpus": bm25.corpus,
            "_doc_texts": bm25._doc_texts,
            "_doc_lens": bm25._doc_lens,
            "_avgdl": bm25._avgdl,
            "_df": dict(bm25._df),
            "_idf": bm25._idf,
            "_N": bm25._N,
            "_signature": bm25.signature,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.pkl")
            with open(path, "wb") as fh:
                pickle.dump(state, fh)
            loaded = BM25Index()
            self.assertFalse(loaded.load(path))


if __name__ == "__main__":
    unittest.main()
