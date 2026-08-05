"""BM25 keyword index with jieba Chinese tokenization and disk persistence."""

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import jieba


def _doc_signature(documents: List[str]) -> str:
    """Deterministic fingerprint of a document list (for BM25 cache freshness)."""
    import hashlib

    payload = "\x00".join(documents).encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()


class BM25Index:
    """BM25 inverted index for Chinese text retrieval.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b  : float
        Length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # internal state
        self.corpus: List[List[str]] = []   # tokenized documents
        self._doc_texts: List[str] = []     # original strings
        self._doc_lens: List[int] = []       # token count per doc
        self._avgdl: float = 0.0
        self._df: defaultdict = defaultdict(int)  # document frequency
        self._idf: dict = {}
        self._N: int = 0
        self._signature: Optional[str] = None  # fingerprint of _doc_texts

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def build(self, documents: List[str]):
        """Tokenise ``documents`` and construct the BM25 index."""
        self._doc_texts = documents # 原始文档字符串
        self._signature = _doc_signature(documents)  # 用于缓存失效校验
        self.corpus = [list(jieba.cut(doc)) for doc in documents] # 文档的分词结果
        self._doc_lens = [len(tokens) for tokens in self.corpus] # 每篇文档词条数
        self._N = len(self.corpus) # 文档总数
        self._avgdl = sum(self._doc_lens) / self._N if self._N else 0.0 # 所有文档平均词条数

        # document frequencies
        self._df.clear()
        for tokens in self.corpus:
            seen = set()
            for t in tokens:
                if t not in seen:
                    self._df[t] += 1
                    seen.add(t)

        # pre-compute IDF
        # 计算每个词条的idf
        self._idf.clear()
        for term, freq in self._df.items():
            self._idf[term] = math.log(
                (self._N - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        # 根据query搜索top-k个最相关结果
        """Return the top-k *(doc_index, score)* tuples for *query*."""
        if self._N == 0:
            return []

        tokens = list(jieba.cut(query)) # 将query分词
        scores: List[Tuple[int, float]] = [] # 得分

        for i, doc_tokens in enumerate(self.corpus):
            # 对每个文档都记一次分
            score = 0.0
            dl = self._doc_lens[i]
            for term in tokens:
                if term not in self._idf:
                    continue
                tf = doc_tokens.count(term) # 词频
                if tf == 0:
                    continue
                idf = self._idf[term]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * num / den

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True) # 按得分降序排列
        return scores[:top_k]

    # ------------------------------------------------------------------
    #  persistence
    # ------------------------------------------------------------------
    # 进行一个本地保存（持久化）
    def save(self, path: str) -> None:
        """Persist the index to disk so it can be reloaded without re-tokenising.

        Serialises all internal state with :mod:`pickle`.  The resulting file
        can be several MB for large corpora (the full tokenised corpus is
        stored).
        """
        state = {
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "_doc_texts": self._doc_texts,
            "_doc_lens": self._doc_lens,
            "_avgdl": self._avgdl,
            "_df": dict(self._df),
            "_idf": self._idf,
            "_N": self._N,
            "_signature": self._signature,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    def load(self, path: str) -> bool:
        """Load a previously saved index from disk.

        Returns ``True`` on success, ``False`` if the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            return False

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        self.k1 = state["k1"]
        self.b = state["b"]
        self.corpus = state["corpus"]
        self._doc_texts = state["_doc_texts"]
        self._doc_lens = state["_doc_lens"]
        self._avgdl = state["_avgdl"]
        self._df = defaultdict(int, state["_df"])
        self._idf = state["_idf"]
        self._N = state["_N"]
        self._signature = state.get("_signature")
        return True

    # ------------------------------------------------------------------
    #  properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return self._N

    @property
    def signature(self) -> Optional[str]:
        return self._signature
