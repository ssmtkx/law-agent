"""Unit tests for PaperSentenceSplitter."""

import unittest

from src.ingestion.splitter import PaperSentenceSplitter


class SplitterSectionTest(unittest.TestCase):
    def setUp(self):
        self.splitter = PaperSentenceSplitter(chunk_size=100, chunk_overlap=20)

    def test_split_by_sections(self):
        text = "一、引言\n这是第一段内容。\n\n二、主体\n这是第二段内容。"
        chunks = self.splitter.split(text)
        self.assertTrue(any("一、引言" in c for c in chunks))
        self.assertTrue(any("二、主体" in c for c in chunks))
        # 章节不应被合并到同一个 chunk（回归：标题曾被正则吞掉导致合并）
        self.assertFalse(any("一、引言" in c and "二、主体" in c for c in chunks))

    def test_intro_before_first_section_is_kept(self):
        text = "本书前言介绍。\n\n一、引言\n这是第一段内容。\n\n二、主体\n这是第二段内容。"
        chunks = self.splitter.split(text)
        self.assertTrue(any("本书前言介绍" in c for c in chunks))
        self.assertTrue(any("一、引言" in c for c in chunks))
        self.assertTrue(any("二、主体" in c for c in chunks))

    def test_no_section_falls_back_to_text(self):
        text = "没有章节标题的普通文本段落。"
        self.assertEqual(self.splitter.split(text), [text])

    def test_long_paragraph_split_is_bounded_and_preserves_content(self):
        text = "第一句。" + "内容" * 80 + "第二句。" + "内容" * 80
        chunks = self.splitter.split(text)
        self.assertGreaterEqual(len(chunks), 2)
        # 单句硬切片段长度恰好为 chunk_size；含重叠的缓冲块最多 chunk_size + overlap + 1（拼接换行符）
        for c in chunks:
            self.assertLessEqual(
                len(c),
                self.splitter.chunk_size + self.splitter.chunk_overlap + 1,
            )
        # 内容不丢失（去空白后拼接应包含原文核心句）
        joined = "".join(chunks)
        self.assertIn("第一句", joined)
        self.assertIn("第二句", joined)


if __name__ == "__main__":
    unittest.main()
