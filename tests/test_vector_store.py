"""Tests for the paginated collection reader.

`fetch_all` exists because `collection.get()` without a limit breaks past a
few tens of thousands of records: Chroma's SQLite backend tries to bind one
variable per row and hits ``SQLITE_MAX_VARIABLE_NUMBER`` (~32,766). That was
observed in production — a 32-minute embedding run finished, and the very
next step died with ``too many SQL variables`` when trying to read the index
back to build BM25.

The failure mode is nasty because it only appears at scale: it worked at
27,672 chunks and broke at 54,618.
"""

import unittest

from src.indexing.vector_store import fetch_all


class _FakeCollection:
    """Mimics Chroma's `get(limit=, offset=)` slicing over a fixed dataset."""

    def __init__(self, n: int, max_batch: int | None = None):
        self.ids = [f"id{i}" for i in range(n)]
        self.documents = [f"doc{i}" for i in range(n)]
        self.metadatas = [{"i": i} for i in range(n)]
        # 模拟底层对单次取数条数的硬上限
        self.max_batch = max_batch
        self.calls = 0

    def get(self, limit=None, offset=0, **_kw):
        self.calls += 1
        if self.max_batch is not None and limit and limit > self.max_batch:
            raise RuntimeError("too many SQL variables")
        end = len(self.ids) if limit is None else offset + limit
        return {
            "ids": self.ids[offset:end],
            "documents": self.documents[offset:end],
            "metadatas": self.metadatas[offset:end],
        }


class FetchAllTest(unittest.TestCase):
    def test_returns_everything(self):
        col = _FakeCollection(12_345)
        out = fetch_all(col, batch=5000)
        self.assertEqual(len(out["ids"]), 12_345)
        self.assertEqual(len(out["documents"]), 12_345)
        self.assertEqual(len(out["metadatas"]), 12_345)
        self.assertEqual(out["ids"][0], "id0")
        self.assertEqual(out["ids"][-1], "id12344")

    def test_result_stays_aligned(self):
        """ids / documents / metadatas 必须同序等长 —— 三者错位会让
        BM25 的关键词索引挂到错误的块上。"""
        out = fetch_all(_FakeCollection(1_000), batch=128)
        for i, (doc, meta) in enumerate(zip(out["documents"], out["metadatas"])):
            self.assertEqual(doc, f"doc{i}")
            self.assertEqual(meta["i"], i)

    def test_pages_around_a_hard_row_limit(self):
        """真实场景：单次取数有上限，一次性全取会直接报错。"""
        col = _FakeCollection(54_618, max_batch=5000)
        out = fetch_all(col, batch=5000)
        self.assertEqual(len(out["ids"]), 54_618)
        self.assertGreater(col.calls, 1)

    def test_empty_collection(self):
        out = fetch_all(_FakeCollection(0))
        self.assertEqual(out["ids"], [])
        self.assertEqual(out["documents"], [])

    def test_exact_multiple_terminates(self):
        """条数正好是批大小的整数倍时也要能停下，不能死循环。"""
        col = _FakeCollection(200)
        out = fetch_all(col, batch=100)
        self.assertEqual(len(out["ids"]), 200)

    def test_no_duplicates_across_pages(self):
        out = fetch_all(_FakeCollection(1_234), batch=100)
        self.assertEqual(len(set(out["ids"])), 1_234)


if __name__ == "__main__":
    unittest.main()
