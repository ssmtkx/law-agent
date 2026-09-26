"""Chinese statute parser: docx bytes → structured articles → RAG chunks.

Chinese statutes have a natural, unambiguous chunk boundary — the article
(条).  One article is one semantically complete unit of law, so unlike the
generic :class:`~src.ingestion.splitter.PaperSentenceSplitter` this parser
never has to guess where a section starts.

Two stages:

1. :class:`LawTextParser` — extract text from a ``.docx`` and split it into
   articles, each carrying its article number, numeric value and the
   chapter/section it belongs to.
2. :class:`LawArticleSplitter` — turn those articles into retrieval chunks,
   prefixing each with a context header naming the statute.

The header matters: a bare article body is frequently uninterpretable
("本法所称……", "前款规定的……"), so the statute name and article number
must travel with the text into both the embedding and the prompt.
"""

from __future__ import annotations

import html
import io
import os
import re
import shutil
import zipfile
from typing import List, Optional

from src.utils.cn_numeral import cn_to_int, parse_article_no

# ──────────────────────────────────────────────────────────────
#  Structural patterns
# ──────────────────────────────────────────────────────────────

# 条号必须锚定在行首。不做锚定会把正文里的引用（"依照本法第三百零二条规定"）
# 当成条首 —— 实测海商法宽松匹配 382 处 vs 锚定后 310 条。
_ARTICLE_RE = re.compile(
    r"^(?P<no>第[一二三四五六七八九十百千零〇]+条"
    r"(?:之[一二三四五六七八九十]+)?)"
    r"[\s　]*(?P<body>.*)$"
)

# 编 / 章 / 节 —— 作为条文的容器，同时充当条文的结束边界
_CONTAINER_RE = re.compile(
    r"^(?P<no>第[一二三四五六七八九十百千零〇]+(?P<kind>编|章|节))"
    r"[\s　]*(?P<title>.*)$"
)

# ── 兜底顶层单位 ──
# 一部分现行有效的文件不用「第X条」编号：早期规范性文件（1950–70 年代）与
# 修改/补充决定用「一、二、三、」或「（一）（二）」。这类文件占语料约 18%，
# 没有兜底就完全检索不到。
#
# 只在前述「第X条」一条都没找到时才启用，所以不会干扰正常法规的边界。
_ENUM_RE = re.compile(
    r"^(?P<no>[一二三四五六七八九十]+)、[\s　]*(?P<body>.*)$"
)
_PAREN_RE = re.compile(
    r"^(?P<no>[（(][一二三四五六七八九十]+[)）])[\s　]*(?P<body>.*)$"
)

# 兜底时判定"这段是正文而非版头"的阈值与特征词
_BODY_MIN_CHARS = 40
_HEADER_HINTS = ("公告", "法释〔", "已于", "现予公布", "予以公布")

#: 效力状态标签，与采集器的 STATUS_LABEL 保持一致
STATUS_LABEL = {1: "已废止", 2: "已修改", 3: "现行有效", 4: "尚未生效"}


def _norm_ws(text: str) -> str:
    """Collapse the full-width/normal space runs that docx extraction leaves."""
    return re.sub(r"[\s　]+", " ", text).strip()


_TAG_RE = re.compile(r"<[^>]+>")


def clean_title(title: str) -> str:
    """Strip search-API highlight markup from a statute title.

    The list endpoint wraps matched substrings in
    ``<em class='highlight'>…</em>`` whenever a search term is supplied, e.g.
    ``"<em class='highlight'>中华人民共和国</em><em class='highlight'>刑法</em>"``.
    Left in, it poisons both the citation header and the ``law_title``
    metadata that evaluation matches on, so it is stripped here rather than
    trusted to stay absent.
    """
    return _norm_ws(html.unescape(_TAG_RE.sub("", title or "")))


class LawTextParser:
    """Extract articles from a statute `.docx` file."""

    # ── docx → paragraphs ─────────────────────────────────────

    @staticmethod
    def parse_docx(data: bytes) -> List[str]:
        """Return the non-empty paragraphs of a `.docx`, in document order.

        Reads ``word/document.xml`` directly rather than pulling in
        python-docx — a docx is a zip, and we only need the text runs.
        """
        # 老式 .doc（OLE2 复合文档）会被误当成 docx 下载。它的魔数与 zip
        # 无关，直接交给 zipfile 会得到"没有 word/document.xml"这种含糊报错，
        # 让人以为是解析逻辑的问题。这里先认出来，把原因写清楚。
        if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise ValueError(
                "旧版 .doc 格式（OLE2 复合文档），非 OOXML，需另行转换"
            )

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if "word/document.xml" not in names:
                raise ValueError(
                    f"not an OOXML docx (no word/document.xml; "
                    f"archive contains: {names[:4]})"
                )
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")

        paragraphs = []
        for para_xml in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
            runs = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", para_xml)
            # <w:tab/> and <w:br/> carry no text but do separate words
            text = "".join(runs).strip()
            if text:
                paragraphs.append(text)
        return paragraphs

    @staticmethod
    def parse_pdf(data: bytes) -> List[str]:
        """Return the non-empty lines of a statute PDF.

        Fallback for documents the docx endpoint does not serve as a real
        docx (实测部分最高检司法解释返回的非 zip 内容）。PDF 没有段落结构，
        只能按行切，但条文仍以「第X条」开头，切分不受影响。
        """
        import fitz

        paragraphs: List[str] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                for line in page.get_text().splitlines():
                    line = line.strip()
                    if line:
                        paragraphs.append(line)
        return paragraphs

    # ── paragraphs → articles ─────────────────────────────────

    @classmethod
    def extract_articles(cls, paragraphs: List[str]) -> List[dict]:
        """Split paragraphs into articles.

        Everything before the first article is dropped — that's the title,
        the promulgation order and the table of contents, none of which
        belongs in the index (the title travels in the chunk header instead).

        Returns a list of ``{no, num, text, container}``.
        """
        articles: List[dict] = []
        container = ""          # current 编/章/节 heading path, e.g. "第三章 船员"
        cur: Optional[dict] = None

        def close() -> None:
            if cur is not None and cur["text"].strip():
                cur["text"] = _norm_ws(" ".join(cur["text"].split("\n")))
                articles.append(cur)

        for para in paragraphs:
            m_article = _ARTICLE_RE.match(para)
            if m_article:
                close()
                no = m_article.group("no")
                num, sub = parse_article_no(no)
                cur = {
                    "no": no,
                    "num": num,
                    "sub": sub,
                    "text": m_article.group("body"),
                    "container": container,
                }
                continue

            m_container = _CONTAINER_RE.match(para)
            if m_container:
                close()
                cur = None
                title = _norm_ws(m_container.group("title"))
                container = f"{m_container.group('no')} {title}".strip() if title \
                    else m_container.group("no")
                continue

            # continuation of the current article (wrapped line, sub-item …)
            if cur is not None:
                cur["text"] += "\n" + para
            # else: still in the preamble/TOC — drop

        close()

        if articles:
            for art in articles:
                art.setdefault("chunking", "article")
            return articles

        # 一条「第X条」都没有 → 这份文件不用条文编号，走兜底。
        # 注意不能放宽条首匹配来"救"它：实测这类文件里出现的「第X条」全部是
        # 对**其他法规**的引用（如「将《安排》第一条修改为…」），放宽会把这些
        # 引用误当成本文的条文。
        return cls._extract_fallback(paragraphs)

    # ── convenience ───────────────────────────────────────────

    # ── 兜底：没有「第X条」时按其他顶层单位切 ─────────────────

    @classmethod
    def _extract_fallback(cls, paragraphs: List[str]) -> List[dict]:
        """Fallback chunking for documents that do not use 第X条.

        Three levels, in order of structural reliability:

        1. ``一、`` at line start — used by 修改/补充决定 and early
           regulations. Unambiguous as a top-level unit.
        2. ``（一）`` at line start — same idea, older style.
        3. Paragraph grouping — for 批复 / 复函 that are plain prose with no
           enumeration at all.

        Levels 1 and 2 require at least two markers, otherwise a single
        stray ``一、`` in prose would fabricate a structure that isn't there.
        """
        for pattern, kind in ((_ENUM_RE, "enum"), (_PAREN_RE, "paren")):
            markers = [i for i, p in enumerate(paragraphs)
                       if pattern.match(p)]
            if len(markers) >= 2:
                return cls._units_from_markers(paragraphs, pattern, kind)

        return cls._units_from_paragraphs(paragraphs)

    @classmethod
    def _units_from_markers(cls, paragraphs: List[str], pattern,
                            kind: str) -> List[dict]:
        units: List[dict] = []
        cur: Optional[dict] = None

        def close() -> None:
            if cur is not None and cur["text"].strip():
                cur["text"] = _norm_ws(" ".join(cur["text"].split("\n")))
                units.append(cur)

        for para in paragraphs:
            m = pattern.match(para)
            if m:
                close()
                no = m.group("no")
                cur = {"no": no, "num": cn_to_int(no), "sub": 0,
                       "container": "", "text": m.group("body"),
                       "chunking": kind}
                continue
            if cur is not None:
                cur["text"] += "\n" + para
            # else: 标题 / 文号 / 公布信息 —— 丢弃，chunk 头部已带元数据

        close()
        return units

    @classmethod
    def _units_from_paragraphs(cls, paragraphs: List[str],
                               max_chars: int = 400) -> List[dict]:
        """Last resort: group paragraphs into fixed-budget chunks.

        版头（标题 / 公告 / 法释文号 / 通过日期）不含实质内容，先跳过。
        判定用两个特征：段落过短，或含版头提示词 —— 只看长度会漏掉像
        「最高人民法院《关于…的批复》已于…会议通过，现予公布」这种既长
        又是版头的段落。
        """
        start = 0
        for i, para in enumerate(paragraphs[:12]):
            looks_like_header = (
                len(para) < _BODY_MIN_CHARS
                or any(h in para for h in _HEADER_HINTS)
            )
            if not looks_like_header:
                start = i
                break

        units: List[dict] = []
        buf = ""
        for para in paragraphs[start:]:
            if buf and len(buf) + len(para) > max_chars:
                units.append(buf)
                buf = para
            else:
                buf = f"{buf}\n{para}" if buf else para
        if buf.strip():
            units.append(buf)

        return [
            {"no": f"第{i}段", "num": i, "sub": 0, "container": "",
             "text": _norm_ws(" ".join(text.split("\n"))),
             "chunking": "paragraph"}
            for i, text in enumerate(units, 1)
        ]

    @staticmethod
    def parse_legacy_doc(data: bytes) -> List[str]:
        """Word 97-2003 (``.doc``) → 段落，借助 antiword。

        **为什么需要这条路径**：源站对一部分司法解释返回的是老式 `.doc`
        （OLE2 复合文档），而同时提供的 PDF 是**扫描件**（无文字层）。两条
        常规路径都失败，实测因此漏掉 56 部法规 —— 恰好是最高频引用的那批
        （民间借贷、买卖合同、医疗损害、诉讼时效…… 全是 2020-12-29 为对齐
        民法典统一修订的司法解释）。

        antiword 是**可选依赖**：没装就返回空列表，由调用方记为失败，不影响
        其余流程。安装方式见 README。
        """
        import subprocess
        import tempfile

        exe = shutil.which("antiword")
        if not exe:
            return []

        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as fh:
                fh.write(data)
                tmp = fh.name
            # -m UTF-8.txt 指定字符映射，否则中文会乱码
            out = subprocess.run(
                [exe, "-m", "UTF-8.txt", tmp],
                capture_output=True, timeout=60,
            )
            if out.returncode != 0:
                return []
            text = out.stdout.decode("utf-8", errors="replace")
        except Exception:                            # noqa: BLE001
            return []
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    @classmethod
    def parse(cls, data: bytes, fmt: str = "docx") -> List[dict]:
        """Document bytes → articles. ``fmt`` 为 ``"docx"`` / ``"pdf"`` / ``"doc"``."""
        if fmt == "pdf":
            paragraphs = cls.parse_pdf(data)
        elif fmt == "doc":
            paragraphs = cls.parse_legacy_doc(data)
        else:
            paragraphs = cls.parse_docx(data)
        return cls.extract_articles(paragraphs)


def sanitize_meta(meta: dict) -> dict:
    """Make a metadata dict safe for Chroma.

    Chroma accepts only str/int/float/bool values and raises
    ``TypeError: Cannot convert Python object to MetadataValue`` on anything
    else.  In practice the offender is JSON ``null``: the source API returns
    ``sxrq: null`` for statutes whose effective date is unset (实测 117 部),
    and ``dict.get(k, "")`` returns that ``None`` rather than the default
    because the key exists.  Coercing here keeps every caller safe rather
    than relying on each one remembering.
    """
    clean = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class LawArticleSplitter:
    """Turn statute articles into retrieval chunks with a context header.

    Parameters
    ----------
    max_chars : int
        Articles longer than this are split on sentence boundaries.  Only a
        handful of articles (民法典/刑法 分则) are long enough to trigger it;
        the sub-chunks share the same article number.
    """

    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    def build_header(self, law: dict, article: dict) -> str:
        """Render the '《法名》（日期）第X条' prefix."""
        title = clean_title(law.get("title"))
        bracket = title if title.startswith("《") else f"《{title}》"

        bits = []
        if law.get("gbrq"):
            bits.append(f"{law['gbrq']}公布")
        if law.get("sxrq"):
            bits.append(f"{law['sxrq']}施行")
        # 「尚未生效」必须写进**正文头部**而不只是元数据 —— 生成模型只看得见
        # chunk 文本，看不见 metadata。不显式标注的话，它会拿一部还没施行的
        # 法律去回答当下的问题，而提问者可能正处于旧法适用期。
        if law.get("sxx") == 4:
            bits.append("尚未生效")
        dated = f"（{'，'.join(bits)}）" if bits else ""

        return f"{bracket}{dated}{article['no']}"

    def split(self, law: dict) -> List[dict]:
        """Chunks for one statute.

        Each chunk: ``{text, metadata}``.  ``text`` is header + newline +
        body; ``metadata`` carries the structured fields the evaluator
        matches on (``law_title`` / ``article_no``) and that the retriever
        turns into a citation label.
        """
        chunks: List[dict] = []
        for art in law.get("articles") or []:
            header = self.build_header(law, art)
            for body in self._split_body(art["text"]):
                meta = sanitize_meta({
                    "law_title": clean_title(law.get("title")),
                    "bbbs": law.get("bbbs", ""),
                    "article_no": art["no"],
                    "article_num": art["num"],
                    "article_sub": art.get("sub", 0),
                    "category": law.get("category", ""),
                    "subcategory": law.get("subcategory", ""),
                    "container": art.get("container", ""),
                    "effective_date": law.get("sxrq", ""),
                    "status": law.get("sxx", 3),
                    "status_label": STATUS_LABEL.get(law.get("sxx", 3), ""),
                    # article=按「第X条」切；enum/paren/paragraph=兜底切分。
                    # 标出来是为了让"这段到底是不是一条正式条文"可查证 ——
                    # 兜底块只有编号不同，检索时无法从内容上区分。
                    "chunking": art.get("chunking", "article"),
                })
                chunks.append({"text": f"{header}\n{body}", "metadata": meta})
        return chunks

    def _split_body(self, body: str) -> List[str]:
        """Split an over-long article on sentence boundaries, else return as-is."""
        body = body.strip()
        if not body:
            return []
        if len(body) <= self.max_chars:
            return [body]

        sentences = [s for s in re.split(r"(?<=[。！？；])", body) if s.strip()]
        pieces: List[str] = []
        cur = ""
        for sent in sentences:
            if len(cur) + len(sent) > self.max_chars and cur:
                pieces.append(cur)
                cur = sent
            else:
                cur += sent
        if cur:
            pieces.append(cur)
        return pieces or [body]
