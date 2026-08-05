"""RAG evaluation CLI — one-shot script to benchmark retrieval quality.

Usage::

    python eval_rag.py                          # default: hybrid mode, K=1,3,5,10
    python eval_rag.py --rerank                 # hybrid + cross-encoder reranker
    python eval_rag.py --mode semantic          # semantic-only (Chroma)
    python eval_rag.py --mode bm25              # BM25-only
    python eval_rag.py --k 1,3,5                # custom K values
    python eval_rag.py --verbose                # print per-item verdicts
    python eval_rag.py --save results.json      # save full results to file
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Must be set BEFORE any HF imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerProcessor
from src.eval.evaluator import RAGEvaluator
from scripts.seed_data import KNOWLEDGE_CHUNKS


class _BM25Adapter:
    """Adapt BM25Index.search() → retriever .query() interface."""

    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index

    def query(self, query_texts, n_results=5, **_kw):
        all_docs, all_scores = [], []
        for q in query_texts:
            hits = self._bm25.search(q, top_k=n_results)
            docs = [self._bm25._doc_texts[i] for i, _ in hits]
            scores = [float(s) for _, s in hits]
            all_docs.append(docs)
            all_scores.append(scores)
        return {"documents": all_docs, "distances": all_scores}


def build_retriever(mode: str, collection_name: str = "paper_knowledge"):
    """Build a retriever based on the selected mode.

    Modes:
      - ``hybrid``  : BM25 + semantic fusion (default)
      - ``semantic``: Chroma vector search only
      - ``bm25``    : BM25 keyword search only
    """
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    # check if collection is empty → seed it
    if collection.count() == 0:
        print("[*] Collection is empty, seeding sample knowledge base ...")
        from scripts.seed_data import seed
        seed()
        collection = store.get_or_create_collection(collection_name)

    print(f"[*] Collection: {collection_name} ({collection.count()} chunks)")

    # ── ground-truth consistency check ──
    # eval 数据集的 relevant_chunk_indices 基于 seed 的 KNOWLEDGE_CHUNKS;
    # 若当前集合不含这些块,检索永远 MISS,评测结果无意义 → 提前明确警告。
    docs_now = collection.get().get("documents", [])
    missing = [c for c in KNOWLEDGE_CHUNKS if c not in docs_now]
    if missing:
        print(f"[!] 警告: 当前集合缺少 {len(missing)}/{len(KNOWLEDGE_CHUNKS)} 个评测基准块")
        print(f"[!] 若评测结果全部 MISS,请先运行: python scripts/seed_data.py 重建知识库")

    if mode == "semantic":
        return collection

    # ── BM25 ──
    result = collection.get()
    docs = result.get("documents", [])
    bm25 = BM25Index()
    bm25.build(docs)
    print(f"[*] BM25 index: {bm25.document_count} documents")

    if mode == "bm25":
        return _BM25Adapter(bm25)

    # ── hybrid (default) ──
    hybrid = HybridRetriever(collection, bm25, alpha=0.3)
    print("[*] HybridRetriever (α=0.3)")
    return hybrid


def main():
    parser = argparse.ArgumentParser(
        description="RAG retrieval evaluation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "semantic", "bm25"], default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply CrossEncoder reranker on top of hybrid retrieval",
    )
    parser.add_argument(
        "--k", type=str, default="1,3,5,10",
        help="Comma-separated K values (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/eval_qa.json",
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save full results to a JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-item HIT/MISS verdicts",
    )
    parser.add_argument(
        "--in-domain", action="store_true",
        help="Only evaluate in-domain questions (exclude out-of-knowledge-base)",
    )
    args = parser.parse_args()

    k_values = tuple(int(x.strip()) for x in args.k.split(","))

    # ── 1. build retriever ──
    print("=" * 64)
    print("  RAG 检索评测")
    print("=" * 64)
    print(f"  模式: {args.mode}{' + Reranker' if args.rerank else ''}")
    print(f"  K 值: {k_values}")
    print()

    retriever = build_retriever(args.mode)

    # ── optional reranker ──
    if args.rerank:
        print("[*] Wrapping with CrossEncoder reranker ...")
        retriever = RerankerProcessor(retriever)

    # ── 2. load dataset ──
    print(f"[*] Loading eval dataset: {args.dataset}")
    dataset = RAGEvaluator.load_dataset(args.dataset)
    print(f"[*] Loaded {len(dataset)} eval items")

    # ── optional filter to in-domain only ──
    if args.in_domain:
        dataset = [it for it in dataset if it.get("relevant_chunk_indices")]
        print(f"[*] Filtered to {len(dataset)} in-domain items")
        if not dataset:
            print("[!] No in-domain items left after filtering — check eval dataset.")
            return

    # ── 3. evaluate ──
    evaluator = RAGEvaluator(retriever, KNOWLEDGE_CHUNKS)
    print("[*] Running evaluation ...\n")
    result = evaluator.evaluate(dataset, k_values=k_values, verbose=args.verbose)

    # ── 4. report ──
    evaluator.print_report(result, dataset)

    # ── 5. optional save ──
    if args.save:
        # strip per_item retrieved docs (can be long) for cleaner JSON
        slim = {
            "config": {
                "mode": args.mode,
                "rerank": args.rerank,
                "k_values": list(k_values),
            },
            "summary": result["summary"],
            "per_item": [
                {"id": it["id"], "question": it["question"],
                 "relevant_count": it["relevant_count"],
                 "metrics": it["metrics"]}
                for it in result["per_item"]
            ],
        }
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(slim, fh, ensure_ascii=False, indent=2)
        print(f"[OK] Results saved to {args.save}")

    # ── check milestone ──
    hit5 = result["summary"].get("hit_rate@5", 0.0)
    if hit5 >= 0.89:
        print("  ✅ 里程碑达成: Hit Rate@5 >= 89%")
    else:
        print(f"  ⚠️  里程碑未达成: Hit Rate@5 = {hit5:.1%} (目标 89%)")


if __name__ == "__main__":
    main()
