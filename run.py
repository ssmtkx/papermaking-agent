"""CLI entry point — start an interactive conversation with Paper Agent."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── HF mirror must be set before any HF-dependent imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()

from src.retrieval.factory import build_retriever, rebuild_index
from src.generation.rag_pipeline import PaperAgent


_WELCOME = r"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     📜  造纸智能助手 — 小纸  Paper Knowledge Agent       ║
║                                                          ║
║     我可以回答：                                          ║
║     · 制浆工艺（打浆、脱墨、浆料配比…）                    ║
║     · 抄纸流程（网部、压榨、干燥…）                       ║
║     · 施胶与涂布（AKD、表面施胶…）                        ║
║     · 故障排查（气泡、掉粉、定量波动…）                   ║
║     · 质量控制（白度、撕裂度、耐破度…）                   ║
║                                                          ║
║     输入 /help 查看命令  |  /clear 清空记忆  |  /quit 退出 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""


def interactive_loop(retriever):
    agent = PaperAgent(retriever, top_k=5)
    print(_WELCOME)

    while True:
        try:
            question = input("🙋 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 小纸：再见！有问题随时来找我。\n")
            break

        if not question:
            continue

        # ── slash commands ──
        if question.startswith("/"):
            cmd = question[1:].strip().lower()
            if cmd in ("quit", "exit", "q"):
                print("👋 小纸：再见！有问题随时来找我。\n")
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
                print("🔨 重建索引中...")
                from src.retrieval.reranker import RerankerProcessor
                new_hybrid = rebuild_index()
                if new_hybrid:
                    agent.retriever = RerankerProcessor(new_hybrid, candidate_pool=20)
                    print("[*] 索引重建完成\n")
                continue
            else:
                print(f"未知命令: /{cmd}，输入 /help 查看可用命令\n")
                continue

        # ── normal turn ──
        result = agent.chat(question)
        print(f"\n🤖 小纸：{result['answer']}\n")

        if result.get("sources"):
            print(f"  📚 引用了 {len(result['sources'])} 条资料\n")


def main():
    retriever, count = build_retriever()
    if count == 0:
        print("[!] 知识库为空。请先运行: python scripts/seed_data.py")
        return

    print(f"[*] 知识库: {count} 条 | 三阶段检索链就绪\n")
    interactive_loop(retriever)


if __name__ == "__main__": #让一个 Python 文件既能被当作“模块”导入使用，又能被当作“脚本”直接运行。
    #当该文件被import时，__name__等于文件名；当该文件直接执行时，__name__=__main__
    main()
