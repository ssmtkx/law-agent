"""Unit tests for HybridRetriever fusion and metadata propagation."""

import unittest

from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever

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


class FakeCollection:
    """固定返回三块文档的伪 Chroma 集合（语义路命中全部文档）。"""

    def query(self, query_texts, n_results=5):
        k = min(n_results, len(DOCS))
        return {
            "documents": [DOCS[:k]],
            "distances": [[0.1, 0.5, 0.9][:k]],
            "ids": [IDS[:k]],
            "metadatas": [METAS[:k]],
        }


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self):
        bm25 = BM25Index()
        bm25.build(DOCS, doc_ids=IDS, metadatas=METAS)
        self.retriever = HybridRetriever(FakeCollection(), bm25, alpha=0.3)

    def test_result_shape(self):
        res = self.retriever.query(["违约责任"], n_results=3)
        docs, dists, ids, metas = (
            res["documents"][0],
            res["distances"][0],
            res["ids"][0],
            res["metadatas"][0],
        )
        self.assertEqual(len(docs), len(dists))
        self.assertEqual(len(docs), len(ids))
        self.assertEqual(len(docs), len(metas))

    def test_bm25_only_doc_joins_union(self):
        # 语义路返回全部文档，因此 union 应包含 3 块
        res = self.retriever.query(["盗窃 违约责任"], n_results=3)
        self.assertEqual(len(res["documents"][0]), 3)
        self.assertEqual(len(res["ids"][0]), 3)

    def test_metadata_propagated(self):
        res = self.retriever.query(["盗窃"], n_results=3)
        idx = res["ids"][0].index("d1")
        self.assertEqual(res["metadatas"][0][idx]["law_title"], "中华人民共和国刑法")
        self.assertEqual(res["metadatas"][0][idx]["article_no"], "第二百六十四条")

    def test_scores_sorted_desc(self):
        res = self.retriever.query(["盗窃 违约责任"], n_results=3)
        self.assertEqual(
            res["distances"][0],
            sorted(res["distances"][0], reverse=True),
        )


if __name__ == "__main__":
    unittest.main()
