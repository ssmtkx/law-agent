"""查看法规采集进度。

采集器只在每个子类别**结束时**打印一次汇总，中途看不到进展；这个脚本直接
读磁盘上已落盘的 JSON，所以对正在运行的采集也有效，不需要重启。

Usage::

    python scripts/crawl_status.py              # 看一眼
    python scripts/crawl_status.py --watch      # 每 30 秒刷新
    python scripts/crawl_status.py --watch 10   # 每 10 秒刷新
    python scripts/crawl_status.py --refresh    # 重新向接口查各类别总数
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.crawl_laws import CATEGORIES, RateLimiter, search_page  # noqa: E402

LAW_DIR = Path("data/raw/laws")
META_DIR = Path("data/meta")
TOTALS_CACHE = META_DIR / "totals.json"

# 每个叶类别在接口里的法规总数。首次运行会实测并缓存到 totals.json。
_STALE_AFTER = 300      # 超过这个秒数没有新文件就提示可能停滞


def load_totals(refresh: bool) -> dict:
    """Per-category expected law counts, cached on disk."""
    if not refresh and TOTALS_CACHE.exists():
        try:
            return json.loads(TOTALS_CACHE.read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            pass

    print("[*] 正在向接口查询各类别法规总数 …")
    limiter = RateLimiter(1.5)
    totals: dict = {}
    for cat_key, (label, subs) in CATEGORIES.items():
        total = 0
        for _sub_label, code in subs:
            try:
                limiter.wait()
                total += search_page(code, 1, size=1).get("total") or 0
            except Exception as exc:                 # noqa: BLE001
                print(f"    [!] {label}/{code} 查询失败: {exc}")
        totals[cat_key] = {"label": label, "total": total}

    META_DIR.mkdir(parents=True, exist_ok=True)
    TOTALS_CACHE.write_text(
        json.dumps(totals, ensure_ascii=False, indent=1), encoding="utf-8")
    return totals


def scan_disk() -> dict:
    """Count what is actually on disk and when it landed."""
    now = time.time()
    per_cat: dict = {}
    articles = 0
    newest = 0.0
    recent: list[float] = []          # mtimes within the last 5 minutes

    for cat_key in CATEGORIES:
        cat_dir = LAW_DIR / cat_key
        files = list(cat_dir.glob("*.json")) if cat_dir.exists() else []
        cat_articles = 0
        for f in files:
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            newest = max(newest, mtime)
            if now - mtime <= 300:
                recent.append(mtime)
            try:
                cat_articles += len(
                    json.loads(f.read_text(encoding="utf-8")).get("articles") or []
                )
            except Exception:                        # noqa: BLE001
                pass
        per_cat[cat_key] = {"laws": len(files), "articles": cat_articles}
        articles += cat_articles

    return {
        "per_cat": per_cat,
        "laws": sum(c["laws"] for c in per_cat.values()),
        "articles": articles,
        "newest": newest,
        "recent": recent,
        "now": now,
    }


def load_failures() -> dict:
    """Failure/warning tallies from the crawler's incremental report.

    The report is only flushed at category boundaries, so during a run it can
    belong to the *previous* run. Report its age and let the caller suppress
    failures that are stale — showing "失败 1976 部" for a run that started
    four seconds ago is worse than showing nothing.
    """
    path = META_DIR / "crawl_report.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - path.stat().st_mtime
    except Exception:                                # noqa: BLE001
        return {}
    reasons: dict = {}
    for w in report.get("warnings") or []:
        reasons[w.get("reason", "unknown")] = reasons.get(w.get("reason", "unknown"), 0) + 1
    return {
        "failures": len(report.get("failures") or []),
        "warn_reasons": reasons,
        "last_error": (report.get("failures") or [{}])[-1].get("error", ""),
        "age": age,
    }


def load_heartbeat() -> dict:
    """Read the crawler's heartbeat, so 'cooling' can be told from 'dead'."""
    path = META_DIR / "crawl_heartbeat.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return {}


def bar(done: int, total: int, width: int = 12) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    filled = min(width, round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(totals: dict, disk: dict, fail: dict) -> None:
    now = disk["now"]
    print("=" * 72)
    print("  法律法规采集进度")
    print("=" * 72)
    print(f"  {'类别':<10}{'已采':>7} / {'目标':>7}   {'进度':<15}{'覆盖率':>8}{'条文':>10}")
    print("  " + "-" * 68)

    grand_done = grand_total = 0
    for cat_key, info in totals.items():
        d = disk["per_cat"].get(cat_key, {"laws": 0, "articles": 0})
        total = info["total"]
        grand_done += d["laws"]
        grand_total += total
        pct = d["laws"] / total if total else 0.0
        print(f"  {info['label']:<10}{d['laws']:>7} / {total:>7}   "
              f"{bar(d['laws'], total):<15}{pct:>7.1%}{d['articles']:>10,}")

    print("  " + "-" * 68)
    gpct = grand_done / grand_total if grand_total else 0.0
    print(f"  {'合计':<10}{grand_done:>7} / {grand_total:>7}   "
          f"{bar(grand_done, grand_total):<15}{gpct:>7.1%}{disk['articles']:>10,}")

    # ── rate + ETA ──
    recent = disk["recent"]
    print()
    if len(recent) >= 2:
        span = max(max(recent) - min(recent), 1.0)
        rate = len(recent) / span * 60.0
        remaining = max(grand_total - grand_done, 0)
        eta_min = remaining / rate if rate > 0 else float("inf")
        print(f"  速度: {rate:.1f} 部/分钟（近 5 分钟）")
        if eta_min != float("inf") and remaining:
            h, m = divmod(int(eta_min), 60)
            print(f"  预计剩余: 约 {h} 小时 {m} 分（不含 WAF 冷却等待）")
        else:
            print("  预计剩余: 已完成")
    else:
        print("  速度: 近 5 分钟没有新文件")

    # ── liveness: 心跳优先，能区分「冷却中」和「已退出」 ──
    beat = load_heartbeat()
    beat_age = now - beat["updated_at"] if beat.get("updated_at") else None

    if beat_age is not None and beat_age < 120:
        state = beat.get("state")
        if state == "cooling":
            resume_in = max(beat.get("resume_at", now) - now, 0)
            print(f"  状态: ⏸ WAF 冷却中（第 {beat.get('waf_hits', '?')} 次），"
                  f"约 {resume_in/60:.1f} 分钟后自动恢复")
        elif state == "fetching":
            print(f"  状态: 采集中 —— 正在抓取《{beat.get('current', '?')}》")
        elif state == "done":
            print("  状态: ✓ 采集已完成")
        else:
            print(f"  状态: {state or '启动中'}")
    elif beat_age is not None:
        print(f"  状态: ⚠ 采集进程可能已退出"
              f"（心跳停在 {beat_age/60:.1f} 分钟前，"
              f"上次状态 {beat.get('state')}）")
        print("        重新运行采集命令即可续跑（已落盘的会跳过）")
    elif disk["newest"]:
        age = now - disk["newest"]
        print(f"  最近落盘: {age:.0f} 秒前（无心跳文件——"
              f"采集器是加心跳之前的版本）")
    else:
        print("  状态: 尚无文件落盘，采集尚未开始")

    if beat_age is not None and beat_age < 120:
        print(f"  本次已采: {beat.get('collected', 0)} 部 / "
              f"{beat.get('articles', 0):,} 条    "
              f"跳过 {beat.get('skipped', 0)}    "
              f"失败 {beat.get('failed', 0)}")

    # 失败数只在报告是新鲜的、或本轮已结束时才有意义
    running = beat_age is not None and beat_age < 120
    if fail and (not running or fail.get("age", 0) < 600):
        print()
        print(f"  失败: {fail.get('failures', 0)} 部")
        for reason, n in sorted((fail.get("warn_reasons") or {}).items()):
            print(f"    - {reason}: {n}")
        if fail.get("last_error"):
            print(f"    最近错误: {fail['last_error'][:90]}")
    elif fail and running:
        print()
        print(f"  （上次运行的报告里有 {fail.get('failures', 0)} 条失败，"
              f"距现在 {fail.get('age', 0)/60:.0f} 分钟，已忽略）")
    print("=" * 72)


def main() -> int:
    # Windows 控制台默认 GBK，中文会变成乱码
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="查看法规采集进度")
    ap.add_argument("--watch", nargs="?", const=30, type=int, default=None,
                    metavar="SECONDS", help="持续刷新（默认 30 秒）")
    ap.add_argument("--refresh", action="store_true",
                    help="重新向接口查询各类别总数")
    args = ap.parse_args()

    totals = load_totals(args.refresh)

    if args.watch is None:
        render(totals, scan_disk(), load_failures())
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")          # 清屏重绘
            render(totals, scan_disk(), load_failures())
            print(f"\n  每 {args.watch}s 刷新，Ctrl-C 退出")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
