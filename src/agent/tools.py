"""Tool definitions and execution for the Paper Agent ReAct loop.

Three tools corresponding to Phase 3.1 of the development plan:

- ``search_knowledge``  — general-purpose knowledge base search
- ``query_tool_list``   — equipment / instrument lookup for a technique
- ``check_common_mistakes`` — failure-mode / pitfall lookup for a process step

Tool definitions follow the OpenAI function-calling schema so they can be
passed directly to the DeepSeek chat-completions endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════
#  OpenAI-compatible tool definitions
# ═══════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "搜索造纸知识库，获取与问题相关的技术资料。"
                "适用于查询制浆、抄纸、施胶、涂布、质量控制等造纸工艺知识。"
                "这是最通用的检索工具，大部分问题都应先使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的问题或关键词，使用中文表述。例如：'打浆度控制范围'、'AKD施胶用量'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_tool_list",
            "description": (
                "查询某个造纸工艺环节所需的设备、工具和仪器清单。"
                "当用户问'需要什么设备'、'用什么工具'、'需要哪些仪器'时使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "technique": {
                        "type": "string",
                        "description": "造纸工艺环节或技术名称，如'打浆'、'施胶'、'浮选脱墨'、'抄纸'等",
                    }
                },
                "required": ["technique"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_common_mistakes",
            "description": (
                "查询某个造纸工序中常见的错误、故障和注意事项。"
                "当用户问'容易出现什么问题'、'注意事项'、'常见错误'、'怎么避免'时使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "description": "造纸工序名称，如'干燥'、'压榨'、'表面施胶'、'网部脱水'、'打浆'等",
                    }
                },
                "required": ["step"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════
#  Tool executor
# ═══════════════════════════════════════════════════════════════════


class ToolExecutor:
    """Execute tool calls against the retrieval pipeline.

    Parameters
    ----------
    retriever :
        The retrieval object (HybridRetriever or RerankerProcessor) with a
        ``.query(query_texts, n_results)`` method.
    top_k : int
        Number of results to return per tool call (default 5).
    """

    def __init__(self, retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    # ── dispatch ──────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Route a tool call to the right handler and return a string result."""
        handlers = {
            "search_knowledge": self._search_knowledge,
            "query_tool_list": self._query_tool_list,
            "check_common_mistakes": self._check_common_mistakes,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"[ERROR] 未知工具: {tool_name}"

        try:
            return handler(**arguments)
        except Exception as exc:
            return f"[ERROR] 工具执行失败 ({tool_name}): {exc}"

    # ── tool implementations ──────────────────────────────────

    def _search_knowledge(self, query: str) -> str:
        """General-purpose knowledge base search."""
        results = self.retriever.query(
            query_texts=[query],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]

        if not docs:
            return "（知识库检索未命中，请基于专业知识直接回答，不要提及检索失败这件事。）"

        return self._format_docs("知识库检索结果", docs)

    def _query_tool_list(self, technique: str) -> str:
        """Equipment / instrument lookup for a specific technique."""
        # Craft a targeted query for equipment mentions
        augmented_query = f"{technique} 设备 工具 仪器 装置 参数"
        results = self.retriever.query(
            query_texts=[augmented_query],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]

        if not docs:
            return "（知识库检索未命中，请基于专业知识直接回答，不要提及检索失败这件事。）"

        return self._format_docs(f"「{technique}」相关设备与工具", docs)

    def _check_common_mistakes(self, step: str) -> str:
        """Failure-mode / pitfall lookup for a specific process step."""
        augmented_query = f"{step} 常见故障 注意事项 错误 问题 缺陷"
        results = self.retriever.query(
            query_texts=[augmented_query],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]

        if not docs:
            return "（知识库检索未命中，请基于专业知识直接回答，不要提及检索失败这件事。）"

        return self._format_docs(f"「{step}」常见问题与注意事项", docs)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _format_docs(title: str, docs: List[str]) -> str:
        """Format retrieved documents into a readable string block."""
        lines = [f"【{title}】共检索到 {len(docs)} 条资料：\n"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"[来源{i}] {doc}")
        return "\n\n".join(lines)
