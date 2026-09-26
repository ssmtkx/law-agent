"""用户反馈（👍/👎）的持久化。

此前反馈只写在 Streamlit 的 ``session_state`` 里 —— 刷新页面就丢，也没有任何
办法回看"哪些回答被打了差评"。等于有个按钮，但没有闭环。

现在每次点击追加一行 JSONL 到 ``data/feedback.jsonl``，并**连同问题与回答一起
记录**（此前只存了一个 ``up``/``down``，事后来看完全不知道为什么被差评）。

记录字段::

    {"ts": "2026-09-25T21:10:03", "rating": "down",
     "mode": "rag", "question": "...", "answer": "...",
     "sources": ["《民法典》第五百八十五条"], "iterations": 0}

查看统计::

    python scripts/feedback_stats.py
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PATH = os.getenv("FEEDBACK_PATH", "data/feedback.jsonl")

# 同一进程内串行写，避免多 session 并发追加时交错
_LOCK = threading.Lock()

# 单条记录的长度上限 —— 防止把整篇回答写进日志把文件撑爆
_MAX_TEXT = 2000


def _truncate(text: Optional[str]) -> str:
    text = (text or "").strip()
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…"


def record(
    rating: str,
    question: str = "",
    answer: str = "",
    mode: str = "",
    sources: Optional[List[str]] = None,
    path: str = DEFAULT_PATH,
    **extra: Any,
) -> Dict[str, Any]:
    """Append one feedback entry. Returns the recorded entry.

    ``rating`` 取 ``"up"`` 或 ``"down"``；其余字段用于事后复盘，全部可缺省。
    写盘失败不应影响前端 —— 反馈是附属功能，不能因为它把主流程弄崩。
    """
    if rating not in ("up", "down"):
        raise ValueError(f"rating 必须是 'up' 或 'down'，收到 {rating!r}")

    entry: Dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "mode": mode,
        "question": _truncate(question),
        "answer": _truncate(answer),
        "sources": [_truncate(s) for s in (sources or [])][:20],
    }
    entry.update(extra)

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:                                # noqa: BLE001
        pass          # 反馈写盘失败不影响对话

    return entry


def load_all(path: str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    """Read every feedback entry, skipping malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # 半行（写入中断）直接跳过
    return entries


def summarize(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    """Aggregate counts for a quick health check."""
    entries = load_all(path)
    by_rating = Counter(e.get("rating", "?") for e in entries)
    by_mode = Counter(e.get("mode", "?") for e in entries)
    up, down = by_rating.get("up", 0), by_rating.get("down", 0)
    total = up + down
    return {
        "total": len(entries),
        "up": up,
        "down": down,
        "satisfaction": (up / total) if total else 0.0,
        "by_mode": dict(by_mode),
        "worst": [e for e in entries if e.get("rating") == "down"][-10:],
    }
