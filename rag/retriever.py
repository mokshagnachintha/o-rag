"""
retriever.py — Hybrid BM25 + TF-IDF cosine retriever.

* BM25  : classic probabilistic keyword ranking (no deps).
* TF-IDF cosine : dot-product over pre-computed sparse TF-IDF vectors.
* Final score   : alpha * bm25_norm + (1-alpha) * cosine_norm

All maths done with pure Python; numpy is used only when available for
the cosine pass (falls back to pure Python otherwise).
"""
from __future__ import annotations

import math
from typing import List, Dict, Tuple

from .chunker import tokenise

# ------------------------------------------------------------------ #
#  BM25 parameters                                                     #
# ------------------------------------------------------------------ #
K1  = 1.5   # term-frequency saturation
B   = 0.75  # length normalisation weight


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Sparse dot product."""
    if len(a) > len(b):
        a, b = b, a
    return sum(a[t] * b[t] for t in a if t in b)


def _norm(v: Dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values())) or 1.0


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _normalise_scores(scores: List[float]) -> List[float]:
    """Min-max normalise a list to [0, 1]."""
    mn = min(scores)
    mx = max(scores)
    rng = mx - mn or 1.0
    return [(s - mn) / rng for s in scores]


# ------------------------------------------------------------------ #
#  Retriever                                                           #
# ------------------------------------------------------------------ #

class HybridRetriever:
    """
    Loads all chunks into memory once, then answers queries fast.
    Call reload() after new documents are ingested.
    """

    def __init__(self, alpha: float = 0.5):
        """
        alpha=1.0 → pure BM25
        alpha=0.0 → pure TF-IDF cosine
        alpha=0.5 → equal hybrid (default)
        """
        self.alpha = alpha
        self._chunks: List[dict] = []   # [{id, doc_id, text, tokens, tfidf_vec}, ...]
        self._avg_dl: float = 0.0       # average document (chunk) length

    # --- loading ---

    def reload(self) -> None:
        """Re-read all chunks from the database."""
        from .db import load_all_chunks  # lazy import avoids circular dep
        self._chunks = load_all_chunks()
        if self._chunks:
            total = sum(len(c["tokens"]) for c in self._chunks)
            self._avg_dl = total / len(self._chunks)
        else:
            self._avg_dl = 1.0

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    # --- BM25 ---

    def _bm25_scores(self, query_tokens: List[str]) -> List[float]:
        N = len(self._chunks)
        scores: List[float] = []

        # IDF per query token across current corpus
        idf: Dict[str, float] = {}
        for qt in set(query_tokens):
            df = sum(1 for c in self._chunks if qt in c["tokens"])
            idf[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for chunk in self._chunks:
            dl = len(chunk["tokens"]) or 1
            tf_map: Dict[str, int] = {}
            for t in chunk["tokens"]:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                score += idf.get(qt, 0.0) * (
                    tf * (K1 + 1)
                    / (tf + K1 * (1 - B + B * dl / self._avg_dl))
                )
            scores.append(score)
        return scores

    # --- TF-IDF cosine ---

    def _cosine_scores(self, query_tokens: List[str]) -> List[float]:
        from collections import Counter
        tf = Counter(query_tokens)
        total = len(query_tokens) or 1
        q_vec: Dict[str, float] = {t: cnt / total for t, cnt in tf.items()}
        return [_cosine(q_vec, c["tfidf_vec"]) for c in self._chunks]

    # --- public query ---

    def query(self, text: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_text, score) sorted by relevance, top_k results.
        """
        if self.is_empty():
            return []

        q_tokens = tokenise(text)
        if not q_tokens:
            return []

        bm25  = self._bm25_scores(q_tokens)
        cos   = self._cosine_scores(q_tokens)

        bm25_n = _normalise_scores(bm25)
        cos_n  = _normalise_scores(cos)

        combined = [
            (i, self.alpha * b + (1 - self.alpha) * c)
            for i, (b, c) in enumerate(zip(bm25_n, cos_n))
        ]

        combined.sort(key=lambda x: x[1], reverse=True)
        top = combined[:top_k]

        return [
            (self._chunks[i]["text"], score)
            for i, score in top
        ]
