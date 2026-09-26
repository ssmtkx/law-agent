"""构建法规检索评测集。

把外部独立标注数据转成本项目的评测 schema，并追加一组"域外问题"用于
测量拒答能力。

**评测集必须来自语料之外。** 照着语料自己写问题会得到循环论证的结果：
问题成为答案的改写，任何改动都显示满分。这个项目早期版本就吃过这个亏
（真值是指向硬编码列表的下标，README 上那个 Hit@5=100% 实际是 42/42，
与 84% 是同一个数字的两种写法）。

Usage::

    python scripts/build_eval_set.py                     # 用默认来源构建
    python scripts/build_eval_set.py --verify            # 顺带核对语料覆盖度
    python scripts/build_eval_set.py --limit 800         # 采样，便于快速迭代
    python scripts/build_eval_set.py --out data/eval_law.json

数据来源（需先下载到 data/external/）::

    EQUALS      真实法律问答社区问题 → 法条，含条号标注
                https://github.com/andongBlue/EQUALS
                → data/external/equals/{test,train}_data.json

输出 schema::

    {"id": "equals_103542",
     "question": "我侄子在工地干活时触电死亡，赔偿标准是多少？",
     "relevant": [{"law": "《工伤保险条例》", "article": "第三十九条"}],
     "category": "工伤保险条例", "question_type": "how much",
     "source": "equals"}
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.evaluator import normalize_law  # noqa: E402
from src.utils.cn_numeral import article_number  # noqa: E402

EQUALS_DIR = Path("data/external/equals")
DEFAULT_OUT = Path("data/eval_law.json")

# ── 域外问题：应当拒答 ──
#
# 刻意不用"明天天气""红烧肉怎么做"这类送分题 —— 它们太容易，拒答与否
# 说明不了什么。这里用的是**看起来像正经法律问题、但答案不在中国法规
# 语料里**的题目：域外法、尚未立法的领域、需要专业意见而非条文的情形。
# 系统必须识别出"我检索不到依据"，而不是拿相近的中文法条硬凑。
_OUT_OF_KB = [
    ("欧盟《通用数据保护条例》(GDPR) 对数据跨境传输有哪些要求？", "域外法"),
    ("美国专利法里判断新颖性的标准是什么？", "域外法"),
    ("日本民法典关于法定继承顺位是怎么规定的？", "域外法"),
    ("英美法系下 consideration 对合同成立有什么要求？", "域外法"),
    ("国际货物买卖合同中 CISG 的适用范围是怎样的？", "域外法"),
    ("自动驾驶汽车发生事故时，责任应当由谁承担？", "未立法"),
    ("人工智能生成内容的著作权归属目前是怎么确定的？", "未立法"),
    ("元宇宙里的虚拟财产能否作为遗产继承？", "未立法"),
    ("我给公司起名叫「AI 之神」，能不能通过审核？", "需专业判断"),
    ("我家孩子该上哪所小学？", "非法律"),
    ("这份劳动合同值不值得签，你觉得呢？", "需专业判断"),
    ("你觉得法官会怎么判我这个案子？", "需专业判断"),
]


def load_equals() -> list[dict]:
    """Convert EQUALS items into the project's eval schema."""
    items: list[dict] = []
    files = sorted(EQUALS_DIR.glob("*_data.json"))
    if not files:
        print(f"[!] 未找到 EQUALS 数据。请下载后放到 {EQUALS_DIR}/")
        print("    curl -sL -o eq.zip "
              "https://codeload.github.com/andongBlue/EQUALS/zip/refs/heads/main")
        print("    解压后把 test_data.json / train_data.json 复制到该目录")
        return []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8")).get("data") or []
        for it in data:
            # question 是列表（同一问题可能有多种问法），取第一条
            qs = it.get("question") or []
            question = (qs[0] if isinstance(qs, list) else qs) or ""
            question = str(question).strip()
            law = str(it.get("law_type") or "").strip()
            article = str(it.get("law_text_num") or "").strip()
            if not (question and law and article):
                continue

            items.append({
                "id": f"equals_{it.get('id')}",
                "question": question,
                "relevant": [{"law": law, "article": article}],
                "category": normalize_law(law),
                "question_type": it.get("question_type", ""),
                "source": "equals",
            })
    return items


def build_out_of_kb() -> list[dict]:
    """Items that must be refused — nothing in the corpus answers them."""
    return [
        {
            "id": f"ookb_{i:02d}",
            "question": q,
            "relevant": [],
            "expect_refusal": True,
            "category": tag,
            "source": "out_of_kb",
        }
        for i, (q, tag) in enumerate(_OUT_OF_KB, 1)
    ]


def corpus_keys() -> set:
    """(normalised law, article number) for every currently-effective statute."""
    keys = set()
    for path in glob.glob("data/raw/laws/*/*.json"):
        try:
            law = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            continue
        if law.get("sxx") != 3:                      # 只统计索引会收录的
            continue
        name = normalize_law(law.get("title", ""))
        for art in law.get("articles") or []:
            num = article_number(art.get("no"))
            if num:
                keys.add((name, num))
    return keys


def report_coverage(items: list[dict], keys: set) -> None:
    """Per-law coverage — the number that decides whether metrics mean anything."""
    by_law: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for it in items:
        if it.get("expect_refusal"):
            continue
        for gold in it["relevant"]:
            name = normalize_law(gold["law"])
            num = article_number(gold["article"])
            by_law[gold["law"]][1] += 1
            if num and (name, num) in keys:
                by_law[gold["law"]][0] += 1

    print()
    print(f"  {'法规':<26}{'需要':>7}{'收录':>7}{'覆盖率':>9}")
    print("  " + "-" * 50)
    hit = need = 0
    for law, (ok, total) in sorted(by_law.items(), key=lambda x: -x[1][1]):
        hit += ok
        need += total
        mark = "" if ok == total else ("  ✗ 未采集" if ok == 0 else "  ⚠ 部分")
        print(f"  {law[:24]:<26}{total:>7}{ok:>7}{ok/total:>8.1%}{mark}")
    print("  " + "-" * 50)
    print(f"  {'合计':<26}{need:>7}{hit:>7}{hit/need if need else 0:>8.1%}")
    print()
    if hit < need:
        print("  ⚠ 覆盖率不足 100%：这些条目的黄金条文不在语料中，必然 MISS。")
        print("    此时 Hit Rate 主要反映**语料完整度**，不是检索能力 ——")
        print("    等语料采集完成后重跑本脚本再看指标。")


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="构建法规检索评测集",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出文件")
    ap.add_argument("--limit", type=int, default=None,
                    help="最多取多少条 EQUALS 条目（按法规分层采样）")
    ap.add_argument("--verify", action="store_true",
                    help="构建后核对语料覆盖度")
    args = ap.parse_args()

    print("=" * 66)
    print("  构建评测集")
    print("=" * 66)

    items = load_equals()
    if not items:
        return 1
    print(f"[*] EQUALS: {len(items)} 条")

    if args.limit and len(items) > args.limit:
        # 分层采样，避免采样后只剩一两部法
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for it in items:
            buckets[it["category"]].append(it)
        per = max(1, args.limit // len(buckets))
        picked: list[dict] = []
        for bucket in buckets.values():
            picked.extend(bucket[:per])
        items = picked[:args.limit]
        print(f"[*] 分层采样至 {len(items)} 条"
              f"（覆盖 {len({i['category'] for i in items})} 部法规）")

    ookb = build_out_of_kb()
    print(f"[*] 域外问题（应拒答）: {len(ookb)} 条")
    all_items = items + ookb

    keys = set()
    if args.verify or True:      # 始终核对：这是解读指标的前提
        keys = corpus_keys()
        print(f"[*] 语料: {len(keys):,} 条现行有效条文")
        report_coverage(all_items, keys)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_items, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[OK] 已写入 {out}（{len(all_items)} 条）")
    print()
    print("  下一步:")
    print(f"    python eval_rag.py --dataset {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
