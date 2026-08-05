"""Hybrid retrieval: BM25 (keyword) + Chroma (semantic) with weighted fusion."""

import math
from typing import Dict, List, Optional, Tuple

from src.indexing.bm25_index import BM25Index


class HybridRetriever:
    """Fuse BM25 keyword scores with semantic vector scores.

    Parameters
    ----------
    collection : Chroma collection
        The semantic vector store (must have .query() method).
    bm25_index : BM25Index
        Pre-built BM25 keyword index.
    alpha : float
        Weight for BM25 in fusion.  0 = semantic only, 1 = BM25 only.
        Default 0.3 gives moderate keyword lift while favouring semantics.
    """

    def __init__(
        self,
        collection,
        bm25_index: BM25Index,
        alpha: float = 0.3,
    ):
        self.collection = collection
        self.bm25 = bm25_index
        self.alpha = alpha

    # ------------------------------------------------------------------
    #  public API  (drop-in replacement for collection.query)
    # ------------------------------------------------------------------

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        candidate_pool: int = 20,
    ) -> dict:
        """Run hybrid search for each query string.

        Returns the same shape as ``collection.query()``:
            {"documents": [[doc, ...]], "distances": [[dist, ...]], ...}
        """
        all_docs: List[List[str]] = [] #search返回的文档列表
        all_dists: List[List[float]] = [] #对应的融合分数

        for q in query_texts:
            docs, dists = self._search_one(q, n_results, candidate_pool)
            all_docs.append(docs)
            all_dists.append(dists)

        return {"documents": all_docs, "distances": all_dists}

    # ------------------------------------------------------------------
    #  internal
    # ------------------------------------------------------------------

    def _search_one(
        self, query: str, n_results: int, candidate_pool: int
    ) -> Tuple[List[str], List[float]]:
        """Run BM25 + semantic, fuse, return top N."""

        # ── 1. BM25 keyword search ──
        bm25_hits = self.bm25.search(query, top_k=candidate_pool)
        bm25_map: Dict[int, float] = {idx: score for idx, score in bm25_hits}

        # ── 2. semantic (Chroma) search ──
        chroma_result = self.collection.query(
            query_texts=[query],
            n_results=candidate_pool,
        )
        chroma_docs: List[str] = chroma_result.get("documents", [[]])[0]
        chroma_dists: List[float] = chroma_result.get("distances", [[]])[0]
        # Chroma returns *distance* (lower = better); convert to similarity
        chroma_sims = [1.0 / (1.0 + d) for d in chroma_dists]

        # ── 3. fuse with weighted sum ──
        fused: Dict[str, float] = {}  # doc_text → combined score

        # BM25 contribution
        # BM25命中了就打分
        if bm25_map:
            bm25_scores = list(bm25_map.values())
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min or 1.0
            for i, score in bm25_map.items():
                norm = (score - bm25_min) / bm25_range
                doc_text = self.bm25._doc_texts[i]
                fused[doc_text] = fused.get(doc_text, 0.0) + self.alpha * norm #加上BM25的分数

        # semantic contribution
        # Chroma命中了就打分
        if chroma_sims:
            sem_min, sem_max = min(chroma_sims), max(chroma_sims)
            sem_range = sem_max - sem_min or 1.0
            for doc, sim in zip(chroma_docs, chroma_sims):
                norm = (sim - sem_min) / sem_range
                fused[doc] = fused.get(doc, 0.0) + (1.0 - self.alpha) * norm #加上CHroma的相似度

        # ── 4. sort and return top N ──
        # 实际上取的是并集而不是交集
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:n_results]

        docs = [item[0] for item in ranked]
        scores = [item[1] for item in ranked]

        return docs, scores
