"""CLI entry point — interactive ReAct Agent for paper-making expert system.

Usage::

    python run_agent.py              # start interactive agent session

Differences from ``run.py`` (simple RAG):
    - Full ReAct loop with tool calling (Thought → Action → Observation)
    - Three specialised tools: search_knowledge / query_tool_list / check_common_mistakes
    - Structured output format (准备 → 分步教学 → 避坑指南)
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

from src.agent.react_agent import PaperReActAgent
from src.utils.tracker import get_tracker
from src.retrieval.factory import build_retriever, rebuild_index

_WELCOME = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🤖  造纸智能助手 — 小纸 (Agent 模式)                         ║
║                                                                  ║
║     我可以通过以下工具查询造纸知识：                               ║
║     · search_knowledge    — 通用工艺知识检索                      ║
║     · query_tool_list     — 设备/工具清单查询                     ║
║     · check_common_mistakes — 常见错误/故障查询                   ║
║                                                                  ║
║     💡 试试问：                                                   ║
║     "纸张出现气泡怎么办？"                                        ║
║     "打浆需要准备什么设备？打浆时容易犯什么错误？"                   ║
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
    agent = PaperReActAgent(
        retriever,
        max_iterations=10,
        max_history=10,
        top_k=5,
        on_thought=_on_thought,
    )
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
                    "\n  📖 可用命令:\n"
                    "  /quit   退出\n"
                    "  /clear  清空对话记忆\n"
                    "  /help   显示此帮助\n"
                    "  /index  重建知识索引\n"
                    "  /usage  打印token用量"
                    "\n  💡 提问示例:\n"
                    "  · AKD施胶的用量和pH条件是什么？\n"
                    "  · 表面施胶需要准备什么工具？\n"
                    "  · 干燥部容易出现什么问题？怎么预防？\n"
                    "  · 废纸脱墨全流程怎么做？有什么注意事项？\n"
                )
                continue
            elif cmd == "usage":
                tracker = get_tracker()
                tracker.print_summary()
                continue
            elif cmd == "index":
                print("🔨 重建索引中...")
                from src.retrieval.reranker import RerankerProcessor
                new_hybrid = rebuild_index()
                if new_hybrid:
                    agent.retriever = RerankerProcessor(new_hybrid, candidate_pool=20)
                    agent.tool_executor.retriever = agent.retriever
                    print("[*] 索引重建完成\n")
                continue
            else:
                print(f"未知命令: /{cmd}，输入 /help 查看可用命令\n")
                continue

        # ── normal turn ──
        print("  🤔 思考中...")
        result = agent.chat(question)

        print(f"\n🤖 小纸：\n{result['answer']}\n")
        print(f"  ⚡ 本轮 ReAct 迭代 {result['iterations']} 次，"
              f"调用 {len(result['tool_calls'])} 个工具\n")


def main():
    retriever, count = build_retriever()
    if count == 0:
        print("[!] 知识库为空。请先运行: python scripts/seed_data.py")
        return

    print(f"[*] 知识库: {count} 条 | 三阶段检索链就绪 (BM25 + 语义 → Reranker)")
    print("[*] ReAct Agent 就绪 (search_knowledge / query_tool_list / check_common_mistakes)\n")
    interactive_loop(retriever)


if __name__ == "__main__":
    main()
