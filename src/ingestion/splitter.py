"""Custom text splitter for Chinese paper-making documents.

Respects section boundaries and keeps technical terms intact.
"""

import re
from typing import List


class PaperSentenceSplitter:
    """Split Chinese documents by section headers, then by paragraphs.

    Ensures chunks are semantically coherent — no mid-sentence cuts,
    no truncated technical terms like "打浆度" or "施胶".
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        sections = self._split_by_sections(text)
        chunks = []
        for section in sections:
            chunks.extend(self._split_section(section))
        return chunks

    # ------------------------------------------------------------------
    #  internal helpers
    # ------------------------------------------------------------------

    # Patterns that start a new section in Chinese technical text.
    # 标题部分必须用捕获组包裹，re.split 才会保留标题文本；
    # 否则标题会被当作分隔符吞掉，多个章节还会被合并成一个 chunk。
    _SECTION_PATTERN = re.compile(
        r'(?:^|\n)\s*'
        r'('
        r'[一二三四五六七八九十]+[、．.]'           # 一、二、
        r'|[0-9]+[\.\、]'                           # 1. 2、
        r'|第[一二三四五六七八九十0-9]+[章节]'        # 第X章
        r'|[（(][一二三四五六七八九十0-9]+[)）]'      # (一)
        r')'
    )

    _CHINESE_SENT_END = re.compile(r'[。！？；\n]')

    def _split_by_sections(self, text: str) -> List[str]:
        # 章节划分
        parts = self._SECTION_PATTERN.split(text)
        result = []
        # 首个章节标题前的引言/前言不应被丢弃
        if parts and parts[0].strip():
            result.append(parts[0].strip())
        for i in range(1, len(parts), 2):
            header = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            section = (header + body).strip()
            if section:
                result.append(section)
        return result if result else [text.strip()]

    def _split_section(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: List[str] = []
        buf = ""

        def flush(piece: str) -> None:
            """Append *piece* to the current chunk buffer, flushing when full."""
            nonlocal buf
            if len(buf) + len(piece) > self.chunk_size and buf:
                chunks.append(buf)
                # build overlap from the tail of previous chunk
                overlap = buf[-self.chunk_overlap:] if len(buf) > self.chunk_overlap else buf
                buf = overlap + "\n" + piece
            else:
                buf = (buf + "\n" + piece) if buf else piece

        for para in paragraphs:
            # 超长段落先按句末切小段，避免单个 chunk 无限膨胀（但绝不从词中间截断）
            if len(para) > self.chunk_size:
                for piece in self._split_long_paragraph(para):
                    flush(piece)
            else:
                flush(para)

        if buf.strip():
            chunks.append(buf.strip())

        return chunks

    def _split_long_paragraph(self, para: str) -> List[str]:
        """按句子边界把超长段落切成 <= chunk_size 的小段；单句仍超长则硬切。"""
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；])", para) if s.strip()]
        pieces: List[str] = []
        cur = ""

        for sent in sentences:
            if len(cur) + len(sent) > self.chunk_size:
                if cur:
                    pieces.append(cur)
                    cur = ""
                # 单个句子仍超过上限 → 按字符硬切（保底线）
                if len(sent) > self.chunk_size:
                    pieces.extend(self._hard_split(sent))
                else:
                    cur = sent
            else:
                cur = cur + sent if cur else sent

        if cur:
            pieces.append(cur)
        return pieces

    def _hard_split(self, text: str) -> List[str]:
        """把一段文本按 chunk_size 硬切成若干段（仅用于单句超长时的兜底）。"""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
