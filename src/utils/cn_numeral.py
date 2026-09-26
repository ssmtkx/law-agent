"""Chinese numeral parsing, shared by statute ingestion and evaluation.

This exists as a single module because the logic is subtle enough that
duplicating it produced the same bug twice: parsing a whole article heading
like ``"第十七条之一"`` as one numeral yields 11 (the trailing 一 pollutes the
result), when the base article number is 17.  Both the crawler-side parser
and the eval-side matcher must agree on article identity, so they must share
one implementation.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

# 条号：基础号 + 可选的「之N」子号。修正案大量使用「第十七条之一」这种形式。
_ARTICLE_NO_RE = re.compile(
    r"^第?(?P<num>[一二三四五六七八九十百千零〇]+)条"
    r"(?:之(?P<sub>[一二三四五六七八九十]+))?$"
)

_PUNCT_RE = re.compile(r"[\s　《》〔〕（）()]+")


def cn_to_int(text: str) -> int:
    """Convert a Chinese numeral to an int. ``一百零二`` → 102, ``十`` → 10.

    Handles the additive/multiplicative form used in statute numbering
    (up to 万, which covers every statute article in the corpus).
    Returns 0 for input with no recognisable numeral.
    """
    total = 0      # accumulated value of completed 万-groups
    section = 0    # value accumulated within the current 万-group
    number = 0     # the pending digit

    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                # 十 on its own means 10, not 0×10
                if number == 0:
                    number = 1
                section += number * unit
            number = 0
    return total + section + number


def parse_article_no(no: str) -> Tuple[int, int]:
    """Split an article heading into ``(base, sub)``.

    ``"第一百零二条"`` → ``(102, 0)``;  ``"第十七条之一"`` → ``(17, 1)``.

    The ``之N`` form must be split off before numeral conversion — passing the
    whole heading to :func:`cn_to_int` folds the trailing ``一`` into the
    value and reports article 17 as article 11.
    """
    m = _ARTICLE_NO_RE.match(_PUNCT_RE.sub("", no or "").strip())
    if not m:
        return 0, 0
    return cn_to_int(m.group("num")), cn_to_int(m.group("sub") or "")


def article_number(value) -> Optional[int]:
    """Best-effort article number from either ``"第五百七十七条"`` or ``577``.

    Returns ``None`` when nothing numeral-like is present, so callers can
    distinguish "unparseable" from "article zero".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value

    text = str(value or "")
    if not text:
        return None

    # 阿拉伯数字优先：评测集常用 "577" / "第577条"
    digits = re.search(r"\d+", text)
    if digits:
        return int(digits.group())

    base, _sub = parse_article_no(text)
    return base or None
