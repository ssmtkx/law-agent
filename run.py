"""CLI entry point — interactive statute Q&A."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── HF mirror must be set before any HF-dependent imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()

from src.retrieval.factory import build_retriever, rebuild_index
from src.domain import ASSISTANT_NAME, ASSISTANT_ROLE
from src.generation.rag_pipeline import LawAgent



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


_WELCOME = rf"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ⚖  {ASSISTANT_ROLE} — {ASSISTANT_NAME}                          ║
║                                                          ║
║     检索范围：                                            ║
║     · 法律（民法典、刑法、公司法、劳动法…）                ║
║     · 行政法规                                            ║
║     · 司法解释                                            ║
║                                                          ║
║     回答只引用检索到的条文，并标注法规与条号               ║
║     检索不到会直说，不编造条号                             ║
║                                                          ║
║     输入 /help 查看命令  |  /clear 清空记忆  |  /quit 退出 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


def interactive_loop(retriever):
    agent = LawAgent(retriever, top_k=12)
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
                    "  /quit  退出\n"
                    "  /clear 清空对话记忆\n"
                    "  /help  显示帮助\n"
                    "  /index 重建知识索引\n"
                    "\n  直接输入问题即可与我对话。\n"
                )
                continue
            elif cmd == "index":
                print("🔨 重建索引中（嵌入需要几分钟）...")
                new_hybrid, chunks = rebuild_index()      # 返回 (retriever, 条数)
                if new_hybrid:
                    agent.retriever = new_hybrid          # 精排默认关闭，见 factory.py
                    print(f"[*] 索引重建完成，共 {chunks:,} 条\n")
                continue
            else:
                print(f"未知命令: /{cmd}，输入 /help 查看可用命令\n")
                continue

        # ── normal turn ──
        result = agent.chat(question)
        print(f"\n🤖 法小律：{result['answer']}\n")

        if result.get("sources"):
            print(f"  📚 引用了 {len(result['sources'])} 条资料\n")


def main():
    _force_utf8_stdout()
    retriever, count = build_retriever()
    if count == 0:
        print("[!] 知识库为空。请先采集语料并建索引：")
        print("      python scripts/crawl_laws.py")
        print("      python scripts/build_index.py --rebuild")
        return

    print(f"[*] 知识库: {count} 条 | 三阶段检索链就绪\n")
    interactive_loop(retriever)


if __name__ == "__main__": #让一个 Python 文件既能被当作“模块”导入使用，又能被当作“脚本”直接运行。
    #当该文件被import时，__name__等于文件名；当该文件直接执行时，__name__=__main__
    main()
