"""RAG evaluation metrics: Hit Rate, MRR, Recall, Precision, NDCG.

Works with any retriever that exposes a ``.query(query_texts, n_results)``
method returning ``{"documents": [[doc, ...]], "distances": [[score, ...]]}``.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.reranker import RerankerProcessor


# ──────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Strip whitespace for fuzzy matching."""
    return text.strip().replace("\n", "").replace("\r", "")


def _match_doc(
    retrieved: str,
    ground_truths: List[str],
    match_len: int = 20,
) -> bool:
    """Return True if *retrieved* matches any ground-truth chunk.

    Matches by comparing the first ``match_len`` characters after normalisation.
    """
    needle = _normalize(retrieved)[:match_len]
    for gt in ground_truths:
        if _normalize(gt)[:match_len] == needle:
            return True
    return False


# ──────────────────────────────────────────────
#  Main evaluator
# ──────────────────────────────────────────────

class RAGEvaluator:
    """Evaluate a retriever against a labelled QA dataset.

    Parameters
    ----------
    retriever :
        An object with a ``.query(query_texts, n_results)`` method.
    knowledge_chunks : List[str]
        The full list of text chunks in the knowledge base, in the same
        order referenced by the eval dataset's ``relevant_chunk_indices``.
    """

    def __init__(self, retriever, knowledge_chunks: List[str]):
        self.retriever = retriever
        self.chunks = knowledge_chunks

    # ── load / save dataset ───────────────────────────────────

    @staticmethod
    def load_dataset(path: str) -> List[dict]:
        """Load eval items from a JSON file.

        Expected schema per item:
            id, question, relevant_chunk_indices, category, difficulty
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("eval dataset must be a JSON array")
        for item in data:
            for key in ("id", "question", "relevant_chunk_indices"):
                if key not in item:
                    raise ValueError(f"eval item missing required key: {key}")
        return data

    # ── static metric helpers ──────────────────────────────────

    @staticmethod
    def _reciprocal_rank(rel: List[float]) -> float:
        """1 / rank of the first relevant item, or 0 if none."""
        for i, r in enumerate(rel, start=1):
            if r > 0:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _ndcg(rel: List[float]) -> float:
        """Normalised Discounted Cumulative Gain."""
        if not rel:
            return 0.0
        # DCG
        dcg = sum(
            (2 ** r - 1) / math.log2(i + 2)
            for i, r in enumerate(rel)
        )
        # IDCG (ideal: all relevant first)
        ideal = sorted(rel, reverse=True)
        idcg = sum(
            (2 ** r - 1) / math.log2(i + 2)
            for i, r in enumerate(ideal)
        )
        return dcg / idcg if idcg > 0 else 0.0

    # ── evaluation logic ──────────────────────────────────────

    def evaluate(
        self,
        dataset: List[dict],
        k_values: Tuple[int, ...] = (1, 3, 5, 10),
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run retrieval evaluation across the dataset.

        Returns a dict with per-k metrics and per-item results.
        """
        # build a mapping: chunk_index → normalised prefix for matching
        gt_texts = [_normalize(c) for c in self.chunks]

        per_item: List[dict] = []
        aggregate: Dict[int, Dict[str, float]] = {
            k: {"hit": 0, "reciprocal_rank": 0.0, "recall": 0.0,
                "precision": 0.0, "ndcg": 0.0}
            for k in k_values
        }

        for item in dataset:
            qid = item["id"]
            question = item["question"]
            rel_indices = item["relevant_chunk_indices"]
            rel_texts = [gt_texts[i] for i in rel_indices if i < len(gt_texts)]

            # ── retrieve ──
            result = self.retriever.query(
                query_texts=[question],
                n_results=max(k_values),
            )
            retrieved_docs: List[str] = result.get("documents", [[]])[0]

            # ── per-k metrics ──
            item_result = {
                "id": qid,
                "question": question,
                "relevant_count": len(rel_texts),
                "retrieved": [d[:80] for d in retrieved_docs],  # truncated for report
                "metrics": {},
            }

            for k in k_values:
                top_k_docs = retrieved_docs[:k]

                # relevance vector for top-k
                rel = [
                    1.0 if _match_doc(doc, rel_texts) else 0.0
                    for doc in top_k_docs
                ]

                hit = 1.0 if any(rel) else 0.0
                rr = self._reciprocal_rank(rel)
                # 无相关文档(知识库外)的条目 recall 计 0,避免指标虚高
                recall = sum(rel) / len(rel_texts) if rel_texts else 0.0
                precision = sum(rel) / k if k > 0 else 0.0
                ndcg = self._ndcg(rel)

                item_result["metrics"][f"hit@{k}"] = hit
                item_result["metrics"][f"mrr@{k}"] = rr
                item_result["metrics"][f"recall@{k}"] = recall
                item_result["metrics"][f"precision@{k}"] = precision
                item_result["metrics"][f"ndcg@{k}"] = ndcg

                aggregate[k]["hit"] += hit
                aggregate[k]["reciprocal_rank"] += rr
                aggregate[k]["recall"] += recall
                aggregate[k]["precision"] += precision
                aggregate[k]["ndcg"] += ndcg

            per_item.append(item_result)

            if verbose:
                status = "HIT" if any(
                    _match_doc(d, rel_texts) for d in retrieved_docs[:5]
                ) else "MISS"
                print(f"  [{status}] {qid}: {question[:50]}...")

        # ── average ──
        n = len(dataset)
        summary: Dict[str, float] = {}
        for k in k_values:
            summary[f"hit_rate@{k}"] = aggregate[k]["hit"] / n
            summary[f"mrr@{k}"] = aggregate[k]["reciprocal_rank"] / n
            summary[f"recall@{k}"] = aggregate[k]["recall"] / n
            summary[f"precision@{k}"] = aggregate[k]["precision"] / n
            summary[f"ndcg@{k}"] = aggregate[k]["ndcg"] / n

        return {"summary": summary, "per_item": per_item}

    # ── printing ──────────────────────────────────────────────

    @staticmethod
    def print_report(eval_result: Dict[str, Any], dataset: List[dict]):
        """Print a human-readable evaluation report."""
        summary = eval_result["summary"]
        per_item = eval_result["per_item"]
        n = len(dataset)

        print("\n" + "=" * 64)
        print("                  RAG 检索评测报告")
        print("=" * 64)
        print(f"  评测集规模: {n} 条")

        # category breakdown
        cats: Dict[str, int] = {}
        for item in dataset:
            cat = item.get("category", "未知")
            cats[cat] = cats.get(cat, 0) + 1
        print(f"  覆盖类别: {len(cats)} 类  {cats}")

        print("\n" + "-" * 64)
        print("  核心指标（越高越好 →）")
        print("-" * 64)

        k_values = [k for k in [1, 3, 5, 10] if f"hit_rate@{k}" in summary]
        header = f"  {'Metric':<20}"
        for k in k_values:
            header += f" {'@'+str(k):>8}"
        print(header)
        print("  " + "-" * (20 + 9 * len(k_values)))

        metric_labels = [
            ("Hit Rate", "hit_rate"),
            ("MRR", "mrr"),
            ("Recall", "recall"),
            ("Precision", "precision"),
            ("NDCG", "ndcg"),
        ]
        for label, key in metric_labels:
            row = f"  {label:<20}"
            for k in k_values:
                val = summary.get(f"{key}@{k}", 0.0)
                bar = _mini_bar(val)
                row += f" {val:.3f}"
            print(row)

        # difficulty breakdown
        print("\n" + "-" * 64)
        print("  难度分层 Hit Rate@5")
        print("-" * 64)
        for diff in ["easy", "medium", "hard"]:
            items = [it for it in per_item if _find_diff(dataset, it["id"]) == diff]
            if not items:
                continue
            hits = sum(
                1 for it in items
                if it["metrics"].get("hit@5", 0) >= 0.5
            )
            bar = _mini_bar(hits / len(items))
            print(f"  {diff:<8} ({len(items):>2}条)  {hits/len(items):.3f}  {bar}")

        # list failures (Hit@5 == 0)
        print("\n" + "-" * 64)
        print("  未命中列表（Top-5 无相关文档）")
        print("-" * 64)
        failures = [
            it for it in per_item
            if it["metrics"].get("hit@5", 1) < 0.5
        ]
        if failures:
            for it in failures:
                print(f"  [{it['id']}] {it['question'][:60]}")
            print(f"\n  共 {len(failures)}/{n} 条未命中")
        else:
            print("  全部命中！")

        print("\n" + "=" * 64 + "\n")


def _find_diff(dataset: List[dict], qid: str) -> str:
    for item in dataset:
        if item["id"] == qid:
            return item.get("difficulty", "unknown")
    return "unknown"


def _mini_bar(val: float, width: int = 10) -> str:
    """Draw a tiny ASCII bar for visual scan."""
    filled = round(val * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"
