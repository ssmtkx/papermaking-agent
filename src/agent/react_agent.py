"""ReAct Agent for paper-making expert system.

Implements the Phase 3.2 requirement: a ReAct (Reasoning + Acting) loop
with tool calling, paper-expert persona, structured output, and conversation
memory.

Architecture::

    User input
      │
      ▼
    ReAct Loop (max 10 iterations)
      ├─ Thought: LLM decides which tool(s) to call
      ├─ Action:  execute tool(s) against the retrieval pipeline
      ├─ Observation: feed results back to LLM
      └─ ... repeat until Final Answer
      │
      ▼
    Structured output (prep → steps → pitfalls)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from src.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from src.utils.tracker import get_tracker

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """你叫「小纸」，是一位资深造纸工艺专家，拥有 30 年造纸车间一线经验。
你可以通过以下三个工具查询造纸知识库中的专业资料：

## 工具说明

| 工具 | 用途 | 何时使用 |
|------|------|----------|
| `search_knowledge` | 通用知识检索 | 用户问工艺原理、参数、流程等知识性问题 |
| `query_tool_list` | 查询设备/工具清单 | 用户问"需要什么设备""用什么仪器" |
| `check_common_mistakes` | 查询常见错误/故障 | 用户问"容易出什么问题""注意事项""怎么避免" |

## 工作流程

1. **理解问题**：先分析用户到底想知道什么
2. **选择工具**：根据问题类型，选择合适的工具（通常先用 search_knowledge）
3. **检索资料**：调用工具获取知识库中的权威资料
4. **组织回答**：结合检索到的资料，用专业但易懂的语言回答
5. **不足时追问**：如果一次检索不够，可以用不同关键词再检索一次

## 回答格式

当回答技术问题时，按以下三段式结构组织：

### 🛠 准备工作
列出操作前需要准备的材料、设备、环境条件

### 📋 分步教学
按步骤逐一说明，每步标注关键参数和操作要点

### ⚠️ 避坑指南
列出该工序最常见的 2-4 个错误及其预防/解决方法

## 核心规则

1. **优先使用工具检索**知识库，基于资料回答并在关键信息后标注 [来源X]
2. 如果工具检索的结果足以回答，只在知识库范围内作答，不要额外发挥
3. 如果检索结果**不足以完整回答**，可以结合你自己的知识进行补充。来自知识库的内容标注 [来源N]，来自你自身知识的内容直接回答即可，不要加任何 ⚠️ 免责声明或"知识库未找到"之类的提示
4. 如果问题完全超出造纸领域（如问天气、编程等），友好说明你的专业范围并引导回正题
5. 回答保持**简洁专业**，像经验丰富的老师傅带徒弟。用分点和短句
6. 多轮对话时，可以引用之前的对话内容，但每次新问题仍需重新检索确认"""

# ═══════════════════════════════════════════════════════════════════
#  ReAct Agent
# ═══════════════════════════════════════════════════════════════════


class PaperReActAgent:
    """Paper-making expert agent with ReAct reasoning and tool calling.

    Parameters
    ----------
    retriever :
        The retrieval object (e.g. RerankerProcessor wrapping HybridRetriever).
    api_key : str
        DeepSeek API key.  Falls back to ``DEEPSEEK_API_KEY`` env var.
    base_url : str
        DeepSeek API base URL.  Falls back to ``DEEPSEEK_BASE_URL`` env var
        or ``https://api.deepseek.com``.
    model : str
        Model name to use (default ``deepseek-v4-flash``).
    max_iterations : int
        Maximum ReAct loop iterations per turn (default 10).
    max_history : int
        Number of recent conversation turns to retain (default 10).
    top_k : int
        Default number of results per tool call (default 5).
    on_thought : Callable or None
        Optional callback invoked with each Thought/Action/Observation step
        for UI visualisation (Streamlit / CLI).  Signature::
            on_thought(step_type: str, content: str) -> None
    """

    def __init__(
        self,
        retriever,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-v4-flash",
        max_iterations: int = 10,
        max_history: int = 10,
        top_k: int = 5,
        on_thought: Optional[Callable[[str, str], None]] = None,
    ):
        import os

        self.retriever = retriever
        self.model = model
        self.max_iterations = max_iterations
        self.max_history = max_history
        self.on_thought = on_thought

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tool_executor = ToolExecutor(retriever, top_k=top_k)

        # conversation memory: user ↔ assistant pairs only (no tool messages)
        self.history: List[Dict[str, str]] = []

    # ── public API ──────────────────────────────────────────────

    def chat(self, question: str) -> Dict[str, Any]:
        """Handle one conversation turn through the ReAct loop.

        Returns
        -------
        dict with keys:
            ``answer``   — final text answer
            ``sources``  — list of source strings cited
            ``question`` — original question
            ``iterations`` — ReAct loop count for this turn
            ``tool_calls`` — list of tool calls made (for UI visualisation)
        """
        question = question.strip()

        # ── build the message list for this turn ──
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        # include past conversation (compressed: last N user/assistant pairs)
        messages.extend(self.history[-self.max_history * 2 :])
        messages.append({"role": "user", "content": question})

        # ── ReAct loop ──
        tool_calls_log: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            try:
                tracker = get_tracker()
                with tracker.track("react_agent", model=self.model) as ctx:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    ctx.record(response)
            except Exception as exc:
                logger.error("LLM call failed (iteration %d): %s", iterations, exc)
                # retry once after a short description of the error
                if iterations == 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"（系统提示：上一次调用遇到错误，请重试。错误信息：{exc}）",
                        }
                    )
                    continue
                return self._fallback_answer(question, f"API 调用失败: {exc}")

            message = response.choices[0].message

            # ── no tool calls → final answer ──
            if not message.tool_calls:
                answer = message.content or ""
                self._emit("answer", answer)
                # save to conversation memory
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": answer})
                if len(self.history) > self.max_history * 2:
                    self.history = self.history[-self.max_history * 2 :]
                return {
                    "answer": answer,
                    "sources": self._extract_sources(tool_calls_log),
                    "question": question,
                    "iterations": iterations,
                    "tool_calls": tool_calls_log,
                }

            # ── tool calls → execute and continue ──
            # append assistant message (with tool_calls) to the conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                self._emit("action", f"调用工具: {tool_name}({arguments})")

                result = self.tool_executor.execute(tool_name, arguments)
                self._emit("observation", f"返回 {len(result)} 字符")

                tool_calls_log.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result_preview": result[:300],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # ── max iterations reached ──
        # force a final answer
        messages.append(
            {
                "role": "user",
                "content": "（请基于目前已检索到的资料，给出最终回答。如果资料不足，请诚实说明。）",
            }
        )
        try:
            tracker = get_tracker()
            with tracker.track("react_agent_final", model=self.model) as ctx:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
                ctx.record(response)
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = "抱歉，查询过程中遇到了技术问题。请稍后重试或简化您的问题。"

        self._emit("answer", answer)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        return {
            "answer": answer,
            "sources": self._extract_sources(tool_calls_log),
            "question": question,
            "iterations": iterations,
            "tool_calls": tool_calls_log,
        }

    # ── helpers ─────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.history = []

    def _emit(self, step_type: str, content: str) -> None:
        """Notify the optional thought callback."""
        if self.on_thought:
            try:
                self.on_thought(step_type, content)
            except Exception:
                pass

    @staticmethod
    def _extract_sources(tool_calls_log: List[Dict[str, Any]]) -> List[str]:
        """Extract source document previews from the tool-call log."""
        sources: List[str] = []
        for call in tool_calls_log:
            preview = call.get("result_preview", "")
            if preview:
                sources.append(f"[{call['tool']}] {preview}")
        return sources

    @staticmethod
    def _fallback_answer(question: str, reason: str) -> Dict[str, Any]:
        """Return a safe fallback when the agent cannot complete."""
        return {
            "answer": f"抱歉，处理您的问题时遇到了技术问题（{reason}）。请稍后重试。",
            "sources": [],
            "question": question,
            "iterations": 0,
            "tool_calls": [],
        }
