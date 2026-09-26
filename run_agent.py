"""CLI entry point — interactive ReAct Agent for legal analysis.

Usage::

    python run_agent.py              # start interactive agent session

Differences from ``run.py`` (simple RAG):
    - Full ReAct loop with tool calling (Thought → Action → Observation)
    - Three specialised tools: search_knowledge / query_tool_list / check_common_mistakes
    - Structured output format (法律依据 → 适用分析 → 风险提示)
    - Visible thought chain during reasoning
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── HF mirror must be set before any HF-dependent imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()

from src.agent.react_agent import LawReActAgent
from src.utils.tracker import get_tracker
from src.retrieval.factory import build_retriever, rebuild_index


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，打印 emoji 会直接抛 UnicodeEncodeError。

    实测：`print('👋 ...')` 在 GBK 下必然崩，而这两个 CLI 的欢迎语和输入提示
    都带 emoji —— 等于一启动就挂。方框字符（╔═╗）在 GBK 里有，emoji 没有，
    所以光看"能显示中文"会以为没问题。
    """
    import io
    import sys as _sys
    if hasattr(_sys.stdout, "buffer"):
        _sys.stdout = io.TextIOWrapper(
            _sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "buffer"):
        _sys.stderr = io.TextIOWrapper(
            _sys.stderr.buffer, encoding="utf-8", errors="replace")


_WELCOME = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ⚖   法律智能助手 — 法小律 (Agent 模式)                        ║
║                                                                  ║
║     我可以通过以下工具检索法规条文：                               ║
║     · search_knowledge      — 条文检索                            ║
║     · query_tool_list       — 查询适用法条清单                     ║
║     · check_common_mistakes — 常见误区与法律风险                   ║
║                                                                  ║
║     💡 试试问：                                                   ║
║     "公司拖欠工资三个月，我可以主张什么？"                          ║
║     "买的房子有抵押，合同还有效吗？涉及哪些法律？"                   ║
║                                                                  ║
║     /help 查看命令  /clear 清空记忆  /usage 用量  /quit 退出          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _on_thought(step_type: str, content: str) -> None:
    """Print ReAct thought steps to the terminal.""" #打印思考步骤
    if step_type == "action":
        print(f"  🔧 {content}")
    elif step_type == "observation":
        print(f"  📖 {content}")
    # 'answer' is printed separately


def interactive_loop(retriever):
    agent = LawReActAgent(
        retriever,
        max_iterations=10,
        max_history=10,
        top_k=12,
        on_thought=_on_thought,
    )
    print(_WELCOME)

    while True:
        try:
            question = input("🙋 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 法小律：再见！有问题随时来找我。\n")
            break

        if not question:
            continue

        # ── slash commands ──
        if question.startswith("/"):
            cmd = question[1:].strip().lower()
            if cmd in ("quit", "exit", "q"):
                print("👋 法小律：再见！有问题随时来找我。\n")
                break
            elif cmd == "clear":
                agent.clear_history()
                print("🧹 对话记忆已清空\n")
                continue
            elif cmd in ("help", "h", "?"):
                print(
                    "\n  📖 可用命令:\n"
                    "  /quit   退出\n"
                    "  /clear  清空对话记忆\n"
                    "  /help   显示此帮助\n"
                    "  /index  重建知识索引\n"
                    "  /usage  打印token用量"
                    "\n  💡 提问示例:\n"
                    "  · 违约金过高可以请求法院调减吗？\n"
                    "  · 买的房子有抵押，合同还有效吗？涉及哪些法律？\n"
                    "  · 民间借贷要注意什么？有什么常见风险？\n"
                    "  · 公司拖欠工资三个月，我可以主张什么？\n"
                )
                continue
            elif cmd == "usage":
                tracker = get_tracker()
                tracker.print_summary()
                continue
            elif cmd == "index":
                print("🔨 重建索引中（嵌入需要几分钟）...")
                new_hybrid, chunks = rebuild_index()   # 返回 (retriever, 条数)
                if new_hybrid:
                    agent.retriever = new_hybrid       # 精排默认关闭，见 factory.py
                    agent.tool_executor.retriever = agent.retriever
                    print(f"[*] 索引重建完成，共 {chunks:,} 条\n")
                continue
            else:
                print(f"未知命令: /{cmd}，输入 /help 查看可用命令\n")
                continue

        # ── normal turn ──
        print("  🤔 思考中...")
        result = agent.chat(question)

        print(f"\n🤖 法小律：\n{result['answer']}\n")
        print(f"  ⚡ 本轮 ReAct 迭代 {result['iterations']} 次，"
              f"调用 {len(result['tool_calls'])} 个工具\n")


def main():
    _force_utf8_stdout()
    retriever, count = build_retriever()
    if count == 0:
        print("[!] 知识库为空。请先采集语料并建索引：")
        print("      python scripts/crawl_laws.py")
        print("      python scripts/build_index.py --rebuild")
        return

    print(f"[*] 知识库: {count} 条 | 三阶段检索链就绪 (BM25 + 语义 → Reranker)")
    print("[*] ReAct Agent 就绪 (search_knowledge / query_tool_list / check_common_mistakes)\n")
    interactive_loop(retriever)


if __name__ == "__main__":
    main()
