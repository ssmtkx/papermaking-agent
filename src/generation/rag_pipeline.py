"""Paper Agent: identity + conversation memory + RAG pipeline.

Architecture:
    User input → intent detection →
      ├─ chat (greeting/identity) → direct LLM response
      └─ knowledge query → retrieve → RAG → LLM response
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.generation.prompts import (
    SYSTEM_PROMPT,
    CHAT_RULES,
    RAG_RULES,
    RAG_QA_PROMPT,
)
from src.utils.tracker import get_tracker

load_dotenv()

# ── Simple intent keywords ──
_CHAT_KEYWORDS = {
    "你好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好",
    "你是谁", "你叫什么", "介绍一下你自己", "你能做什么", "你有什么功能",
    "谢谢", "感谢", "再见", "拜拜", "bye",
    "你是什么",
}


class PaperAgent:
    """A paper-making assistant agent with identity, memory, and RAG."""

    def __init__(self, retriever, top_k: int = 5, model: Optional[str] = None):
        self.retriever = retriever  # HybridRetriever or Chroma collection
        self.top_k = top_k
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        # conversation memory: list of {"role": "user"|"assistant", "content": str}
        self.history: list[dict] = []
        self.max_history = 10  # keep last N turns

    # ── public API ─────────────────────────────────────────

    def chat(self, question: str) -> dict:
        """Handle one turn of conversation.

        Returns {"answer": str, "sources": list, "question": str, "mode": "chat"|"rag"}
        """
        question = question.strip()

        # ── intent detection ──
        if self._is_chat(question):
            result = self._chat_reply(question)
        else:
            result = self._rag_reply(question)

        # ── update memory ──
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return result

    def clear_history(self):
        """Reset conversation memory."""
        self.history = []

    # ── internal ───────────────────────────────────────────

    def _is_chat(self, question: str) -> bool:
        """Quick keyword-based intent detection."""
        q = question.strip().lower()
        for kw in _CHAT_KEYWORDS:
            if kw in q:
                return True
        # 短输入(如"打浆""施胶")很可能是技术术语,交由 RAG 处理,不要当闲聊
        return False

    def _chat_reply(self, question: str) -> dict:
        """Respond to greetings / identity questions without RAG."""
        system = SYSTEM_PROMPT.format(chat_rules=CHAT_RULES)
        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-6:])  # recent context
        messages.append({"role": "user", "content": question})

        tracker = get_tracker()
        with tracker.track("rag_chat", model=self.model) as ctx:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            ctx.record(resp)
        return {
            "answer": resp.choices[0].message.content,
            "sources": [],
            "question": question,
            "mode": "chat",
        }

    def _rag_reply(self, question: str) -> dict:
        """Retrieve → RAG prompt → generate."""
        # ── retrieve ──
        results = self.retriever.query(
            query_texts=[question],
            n_results=self.top_k,
        )
        docs: List[str] = results.get("documents", [[]])[0]

        if not docs:
            # fall back to a polite refusal
            system = SYSTEM_PROMPT.format(chat_rules=RAG_RULES)
            messages = [{"role": "system", "content": system}]
            messages.extend(self.history[-6:])
            messages.append({"role": "user", "content": question})

            tracker = get_tracker()
            with tracker.track("rag_fallback", model=self.model) as ctx:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512,
                )
                ctx.record(resp)
            return {
                "answer": resp.choices[0].message.content,
                "sources": [],
                "question": question,
                "mode": "rag",
            }

        # ── build RAG prompt ──
        context = "\n\n".join(
            f"[来源{i+1}] {doc}" for i, doc in enumerate(docs)
        )
        rag_body = RAG_QA_PROMPT.format(context=context, question=question)

        system = SYSTEM_PROMPT.format(chat_rules=RAG_RULES)
        messages = [{"role": "system", "content": system}]
        # include recent history so agent can reference previous turns
        messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": rag_body})

        tracker = get_tracker()
        with tracker.track("rag_knowledge", model=self.model) as ctx:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            ctx.record(resp)
        return {
            "answer": resp.choices[0].message.content,
            "sources": docs,
            "question": question,
            "mode": "rag",
        }
