"""LLM call tracker: token counting, cost estimation, latency, and logging.

Usage::

    from src.utils.tracker import CallTracker

    tracker = CallTracker()

    with tracker.track("react_agent") as ctx:
        response = client.chat.completions.create(...)
        ctx.record(response)   # or ctx.record_usage(prompt_tokens, completion_tokens)

    tracker.print_summary()    # print a summary table
    tracker.to_dict()          # export as dict
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("law_agent.tracker")

# ── DeepSeek pricing (CNY per 1M tokens) ──
# deepseek-v4-pro: CNY2 / CNY8 (input / output)
# deepseek-v4-pro:   CNY10 / CNY40
_PRICE_MAP: Dict[str, tuple] = {
    "deepseek-v4-pro": (2.0, 8.0),
    "deepseek-v4-pro": (10.0, 40.0),
    "deepseek-chat": (2.0, 8.0),  # legacy alias
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in CNY."""
    prices = _PRICE_MAP.get(model)
    if prices is None:
        # unknown model — guess cheap
        prices = (2.0, 8.0)
    input_price, output_price = prices
    return (prompt_tokens / 1_000_000) * input_price + (
        completion_tokens / 1_000_000
    ) * output_price


@dataclass
class CallRecord:
    """One LLM call."""

    source: str  # e.g. "react_agent", "rag_pipeline"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_cny: float = 0.0
    success: bool = True
    error: str = ""

    def __post_init__(self):
        if self.cost_cny == 0.0:
            self.cost_cny = _estimate_cost(
                self.model, self.prompt_tokens, self.completion_tokens
            )


@dataclass
class TrackContext:
    """Mutable context yielded by ``tracker.track()``."""

    source: str
    model: str
    _start: float = field(default_factory=time.perf_counter)
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _success: bool = True
    _error: str = ""
    _recorded: bool = False

    def record(self, response: Any) -> None:
        """Record tokens from an OpenAI chat completion response object."""
        try:
            usage = response.usage
            self._prompt_tokens = usage.prompt_tokens
            self._completion_tokens = usage.completion_tokens
            self.model = response.model
        except Exception:
            pass
        self._recorded = True

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token counts manually."""
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        if model:
            self.model = model
        self._recorded = True

    def mark_error(self, error: str) -> None:
        self._error = error
        self._success = False
        self._recorded = True

    @property
    def latency_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000


class CallTracker:
    """Accumulate LLM call records and produce summaries.

    Parameters
    ----------
    log_file : str or None
        If set, append each call as a JSON line to this file for persistent logging.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.records: List[CallRecord] = []
        self.log_file = log_file

    # ── context manager ──────────────────────────────────────

    @contextmanager
    def track(self, source: str, model: str = ""):
        """Context manager that builds a ``CallRecord``.

        Usage::

            with tracker.track("react_agent") as ctx:
                resp = client.chat.completions.create(...)
                ctx.record(resp)
        """
        ctx = TrackContext(source=source, model=model)
        try:
            yield ctx
        except Exception as exc:
            ctx.mark_error(str(exc))
            raise
        finally:
            record = CallRecord(
                source=ctx.source,
                model=ctx.model,
                prompt_tokens=ctx._prompt_tokens,
                completion_tokens=ctx._completion_tokens,
                latency_ms=ctx.latency_ms,
                success=ctx._success,
                error=ctx._error,
            )
            self.records.append(record)
            self._maybe_persist(record)

    # ── summary ──────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a human-readable summary table."""
        if not self.records:
            print("[tracker] No calls recorded.")
            return

        total_cost = sum(r.cost_cny for r in self.records)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in self.records)
        total_calls = len(self.records)
        failed = sum(1 for r in self.records if not r.success)
        total_latency = sum(r.latency_ms for r in self.records)

        print("\n" + "=" * 64)
        print("  LLM API 调用统计")
        print("=" * 64)
        print(f"  总调用次数: {total_calls}  (失败: {failed})")
        print(f"  总 Token 数: {total_tokens:,}")
        print(f"  总耗时:      {total_latency/1000:.1f}s")
        print(f"  预估费用:    CNY {total_cost:.4f}")
        print()

        # per-source breakdown
        by_source: Dict[str, dict] = {}
        for r in self.records:
            s = by_source.setdefault(r.source, {"calls": 0, "tokens": 0, "cost": 0.0, "latency": 0.0})
            s["calls"] += 1
            s["tokens"] += r.prompt_tokens + r.completion_tokens
            s["cost"] += r.cost_cny
            s["latency"] += r.latency_ms

        header = f"  {'来源':<20} {'调用':>5} {'Tokens':>10} {'耗时':>8} {'费用':>10}"
        print(header)
        print("  " + "-" * 56)
        for source, s in by_source.items():
            row = f"  {source:<20} {s['calls']:>5} {s['tokens']:>10,} {s['latency']/1000:>7.1f}s CNY{s['cost']:>8.4f}"
            print(row)
        print("  " + "-" * 56)
        print(f"  {'合计':<20} {total_calls:>5} {total_tokens:>10,} {total_latency/1000:>7.1f}s CNY{total_cost:>8.4f}")
        print()

        if failed:
            print(f"  ⚠️  失败调用: {failed}")
            for r in self.records:
                if not r.success:
                    print(f"    [{r.source}] {r.error[:100]}")
            print()

        print("=" * 64 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Export records as a dict (useful for JSON serialisation)."""
        total_cost = sum(r.cost_cny for r in self.records)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in self.records)
        return {
            "total_calls": len(self.records),
            "total_tokens": total_tokens,
            "total_cost_cny": round(total_cost, 6),
            "records": [
                {
                    "source": r.source,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency_ms": round(r.latency_ms, 1),
                    "cost_cny": round(r.cost_cny, 6),
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.records
            ],
        }

    def reset(self) -> None:
        """Clear all records."""
        self.records = []

    # ── internal ─────────────────────────────────────────────

    def _maybe_persist(self, record: CallRecord) -> None:
        if not self.log_file:
            return
        try:
            p = Path(self.log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "source": record.source,
                            "model": record.model,
                            "prompt_tokens": record.prompt_tokens,
                            "completion_tokens": record.completion_tokens,
                            "latency_ms": round(record.latency_ms, 1),
                            "cost_cny": round(record.cost_cny, 6),
                            "success": record.success,
                            "error": record.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass


# ── module-level singleton ──
_default_tracker: Optional[CallTracker] = None


def get_tracker(log_file: Optional[str] = None) -> CallTracker:
    """Get (or create) the module-level CallTracker singleton."""
    global _default_tracker
    if _default_tracker is None:
        log_file = log_file or os.getenv(
            "TRACKER_LOG_FILE", "data/api_calls.jsonl"
        )
        _default_tracker = CallTracker(log_file=log_file)
    return _default_tracker
