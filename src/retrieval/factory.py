"""Shared retriever factory — used by CLI, Streamlit, and eval scripts.

Eliminates duplicated retriever setup logic across ``run_agent.py``,
``app.py``, and ``eval_rag.py``.
"""

from __future__ import annotations

import os

from src.indexing.vector_store import VectorStore
from src.indexing.indexer import build_hybrid_from_collection, build_index
from src.retrieval.reranker import RerankerProcessor

# BM25 索引的磁盘缓存路径。重建索引时会先删除这个缓存文件，强制从全量数据重新构建。
BM25_CACHE = os.path.join("data", "bm25_index.pkl")


def build_retriever(
    collection_name: str = "paper_knowledge",
    alpha: float = 0.3,
    candidate_pool: int = 20,
    auto_seed: bool = True, # 若集合为空，自动灌入示例数据（方便开发调试）
) -> tuple[RerankerProcessor, int]:
    """Build the three-stage retrieval chain and return (retriever, chunk_count).

    Three stages: BM25 + semantic vector → HybridRetriever → CrossEncoder Reranker.

    Parameters
    ----------
    collection_name : str
        Chroma collection to use / create.
    alpha : float
        Hybrid fusion weight (0 = semantic-only, 1 = BM25-only).
    candidate_pool : int
        Number of candidates the reranker rescore.
    auto_seed : bool
        If the collection is empty, seed sample data automatically.
    """
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    if collection.count() == 0 and auto_seed: # 若集合为空，自动灌入示例数据（方便开发调试）
        from scripts.seed_data import seed

        seed()
        collection = store.get_or_create_collection(collection_name)

    hybrid = build_hybrid_from_collection(collection, alpha=alpha) #两路召回的结果
    reranker = RerankerProcessor(hybrid, candidate_pool=candidate_pool) #精排
    return reranker, collection.count()


def rebuild_index(data_dir: str = "data/raw"):
    """Rebuild the full index from PDFs in *data_dir*.

    Returns the new HybridRetriever, or None if no PDFs found.
    ``build_index`` now builds BM25 from the full collection and persists it,
    so this just clears the stale cache and delegates.
    """
    # Clear BM25 cache so a failed rebuild can't leave a stale cache behind
    if os.path.exists(BM25_CACHE):
        os.remove(BM25_CACHE)

    return build_index(data_dir)
