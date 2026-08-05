"""Second-stage reranking with BGE cross-encoder + source weighting."""

from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

# ── Authoritative source keywords for boosting ──
_AUTHORITY_KEYWORDS = [
    "访谈录", "访谈", "官方", "典籍", "标准", "规范",
    "国家标准", "行业标准", "专利", "院士", "专家",
]


class RerankerProcessor:
    """Wraps a retriever and applies cross-encoder reranking.

    The wrapped retriever must expose a ``.query(query_texts, n_results)``
    method returning ``{"documents": [[...]], ...}``.

    Parameters
    ----------
    retriever :
        The first-stage retriever (e.g. HybridRetriever).
    model_name : str
        BGE reranker model id on HuggingFace Hub.
    candidate_pool : int
        How many candidates to fetch from the first stage before reranking.
    source_boost : float
        Multiplier applied to documents from authoritative sources.
    """

    def __init__(
        self,
        retriever,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        candidate_pool: int = 20,
        source_boost: float = 1.3,
    ):
        self.retriever = retriever
        self.candidate_pool = candidate_pool
        self.source_boost = source_boost
        self.model = CrossEncoder(model_name) #交叉编码器

    # ------------------------------------------------------------------
    #  public API  (drop-in replacement for retriever.query)
    # ------------------------------------------------------------------

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> dict:
        """Retrieve → rerank → return top-N."""
        all_docs: List[List[str]] = []
        all_dists: List[List[float]] = []

        for query in query_texts:
            docs, scores = self._rerank_one(query, n_results)
            all_docs.append(docs)
            all_dists.append(scores)

        return {"documents": all_docs, "distances": all_dists}

    # ------------------------------------------------------------------
    #  internal
    # ------------------------------------------------------------------

    def _rerank_one(
        self, query: str, n_results: int
    ) -> Tuple[List[str], List[float]]:
        # ── 1. fetch candidates from first stage ──
        raw = self.retriever.query(
            query_texts=[query],
            n_results=self.candidate_pool,
        )
        docs: List[str] = raw.get("documents", [[]])[0] # 初召回，，取出当前 query 对应的文档列表
        # 例如 ["文档A", "文档B", ...]

        if not docs:
            return [], []

        # ── 2. rerank with cross-encoder ──
        pairs = [[query, doc] for doc in docs]
        raw_scores: List[float] = self.model.predict(pairs).tolist()

        # ── 3. apply source authority boost ──
        boosted = [
            self._apply_source_boost(float(s), doc) #权威来源加权
            for s, doc in zip(raw_scores, docs)
        ]

        # ── 4. sort & return top-N ──
        ranked = sorted(
            zip(docs, boosted),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked = ranked[:n_results]

        return (
            [item[0] for item in ranked],
            [item[1] for item in ranked],
        )

    def _apply_source_boost(self, score: float, doc: str) -> float:
        """Multiply score if the document looks like an authoritative source."""
        for kw in _AUTHORITY_KEYWORDS:
            if kw in doc:
                return score * self.source_boost
        return score
