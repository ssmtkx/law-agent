"""查看用户反馈。

反馈记录在 ``data/feedback.jsonl``，每次点击 👍/👎 追加一行，**连同问题与回答
一起记录** —— 否则事后看到一个 ``down`` 完全不知道差在哪。

Usage::

    python scripts/feedback_stats.py              # 汇总
    python scripts/feedback_stats.py --bad 20     # 列出最近 20 条差评的问答
    python scripts/feedback_stats.py --json out.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.feedback import DEFAULT_PATH, load_all, summarize  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="查看用户反馈",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--path", default=DEFAULT_PATH, help="feedback.jsonl 路径")
    ap.add_argument("--bad", type=int, default=0,
                    help="列出最近 N 条差评的问答原文")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="把汇总存成 JSON")
    args = ap.parse_args()

    stat = summarize(args.path)
    if stat["total"] == 0:
        print(f"[i] 暂无反馈记录（{args.path}）")
        print("    在 Web 端对回答点 👍/👎 后即可看到统计。")
        return 0

    print("=" * 60)
    print("  用户反馈汇总")
    print("=" * 60)
    print(f"  记录总数 : {stat['total']}")
    print(f"  有帮助   : {stat['up']}")
    print(f"  需改进   : {stat['down']}")
    print(f"  满意度   : {stat['satisfaction']:.1%}")
    print(f"  按模式   : {stat['by_mode']}")

    if stat["down"]:
        print()
        print("  ⚠ 差评是最有价值的信号 —— 用 --bad N 看具体是哪一问哪一答。")

    if args.bad:
        print()
        print("=" * 60)
        print(f"  最近 {args.bad} 条差评")
        print("=" * 60)
        for e in stat["worst"][-args.bad:]:
            print(f"\n[{e.get('ts')}] mode={e.get('mode')}")
            print(f"  Q: {(e.get('question') or '(未记录)')[:120]}")
            print(f"  A: {(e.get('answer') or '')[:200]}…")
            if e.get("sources"):
                print(f"  引用: {e['sources'][:3]}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(stat, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[OK] 已写入 {args.json_out}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
