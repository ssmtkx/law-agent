"""用户反馈落盘的测试。

此前反馈只写 session_state，刷新即丢、也无法复盘。现在每次点击追加一行
JSONL，并**带上问题与回答** —— 只记一个 up/down 事后完全不知道为什么差评。
"""

import json
import os
import tempfile
import unittest

from src.utils.feedback import load_all, record, summarize


class FeedbackRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "fb.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_question_and_answer(self):
        """核心诉求：反馈必须带上问题与回答，否则无法复盘。"""
        record("down", question="违约金过高能调减吗？",
               answer="根据民法典……", mode="rag", path=self.path)
        entries = load_all(self.path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["rating"], "down")
        self.assertEqual(entries[0]["question"], "违约金过高能调减吗？")
        self.assertIn("民法典", entries[0]["answer"])
        self.assertTrue(entries[0]["ts"])

    def test_appends_rather_than_overwrites(self):
        for r in ("up", "down", "up"):
            record(r, question="q", path=self.path)
        self.assertEqual(len(load_all(self.path)), 3)

    def test_rejects_invalid_rating(self):
        with self.assertRaises(ValueError):
            record("meh", path=self.path)

    def test_missing_file_reads_empty(self):
        self.assertEqual(load_all(os.path.join(self.tmp.name, "nope.jsonl")), [])

    def test_skips_malformed_lines(self):
        """写入中断会留下半行，读取时要跳过而不是整份崩掉。"""
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write('{"rating": "up", "ts": "x"}\n')
            fh.write('{"rating": "do')          # 半行
            fh.write('\n{"rating": "down", "ts": "y"}\n')
        self.assertEqual(len(load_all(self.path)), 2)

    def test_answer_is_truncated(self):
        record("up", answer="长" * 5000, path=self.path)
        self.assertLessEqual(len(load_all(self.path)[0]["answer"]), 2100)

    def test_write_failure_does_not_raise(self):
        """反馈是附属功能，写盘失败不能把主流程弄崩。"""
        record("up", question="q", path="/nonexistent-dir/x/y.jsonl")


class FeedbackSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "fb.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_summarize_counts_and_satisfaction(self):
        record("up", question="a", mode="rag", path=self.path)
        record("up", question="b", mode="agent", path=self.path)
        record("down", question="c", mode="rag", path=self.path)
        s = summarize(self.path)
        self.assertEqual((s["total"], s["up"], s["down"]), (3, 2, 1))
        self.assertAlmostEqual(s["satisfaction"], 2 / 3)
        self.assertEqual(s["by_mode"], {"rag": 2, "agent": 1})
        self.assertEqual(len(s["worst"]), 1)
        self.assertEqual(s["worst"][0]["question"], "c")

    def test_empty_summary_is_zero_not_crash(self):
        s = summarize(self.path)
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["satisfaction"], 0.0)


if __name__ == "__main__":
    unittest.main()
