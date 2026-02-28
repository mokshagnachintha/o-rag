"""
retriever.py — Hybrid BM25 + TF-IDF + Semantic retriever.

* BM25     : classic probabilistic keyword ranking (no deps).
* TF-IDF   : sparse cosine over pre-computed chunk vectors.
* Semantic : dense cosine over embeddings from llama-server /embedding
              endpoint (computed in background after every reload).

When semantic embeddings are available the final score is:
    0.30 * bm25_norm  +  0.20 * tfidf_norm  +  0.50 * semantic_norm
Otherwise falls back to the classic hybrid:
    alpha * bm25_norm  +  (1-alpha) * tfidf_norm
"""
from __future__ import annotations

import math
import threading
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


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _cosine_dense(a: list, b: list) -> float:
    """Cosine similarity between two dense float vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1.0
    nb  = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


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
        alpha used for BM25/TF-IDF fallback only (when semantic unavailable):
            alpha=1.0 → pure BM25
            alpha=0.0 → pure TF-IDF cosine
        """
        self.alpha = alpha
        self._chunks: List[dict] = []   # [{id, doc_id, text, tokens, tfidf_vec}, ...]
        self._avg_dl: float = 0.0
        # Semantic embedding cache: chunk_id -> list[float]
        self._embeddings: dict = {}
        self._embed_lock  = threading.Lock()
        self._embed_ready = False

    # --- loading ---

    def reload(self) -> None:
        """Re-read all chunks from the database and trigger embedding computation."""
        from .db import load_all_chunks
        self._chunks = load_all_chunks()
        with self._embed_lock:
            self._embeddings  = {}
            self._embed_ready = False
        if self._chunks:
            total = sum(len(c["tokens"]) for c in self._chunks)
            self._avg_dl = total / len(self._chunks)
            # Compute dense embeddings in background — doesn't block the UI
            threading.Thread(
                target=self._compute_embeddings,
                daemon=True,
            ).start()
        else:
            self._avg_dl = 1.0

    def _compute_embeddings(self) -> None:
        """
        Background thread: call llama-server /embedding for every chunk
        and cache the result.  Gracefully no-ops if the server is down
        or embeddings are unsupported.
        """
        try:
            from .llm import get_embedding
            computed = {}
            for c in self._chunks:
                cid  = c["id"]
                text = c["text"]
                # Keep each request short to stay within context window
                emb = get_embedding(text[:1024])
                if emb is None:
                    print("[retriever] embedding endpoint unavailable — "
                          "falling back to BM25+TF-IDF only")
                    return
                computed[cid] = emb
            with self._embed_lock:
                self._embeddings  = computed
                self._embed_ready = True
            print(f"[retriever] semantic embeddings ready "
                  f"({len(computed)} chunks)")
        except Exception as e:
            print(f"[retriever] embedding computation failed: {e}")

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
        return [_cosine_sparse(q_vec, c["tfidf_vec"]) for c in self._chunks]

    # --- semantic cosine ---

    def _semantic_scores(self, query_text: str) -> "List[float] | None":
        """
        Returns per-chunk cosine similarity against a fresh query embedding,
        or None if embeddings are not yet ready / unavailable.
        """
        with self._embed_lock:
            if not self._embed_ready:
                return None
            embeddings = dict(self._embeddings)  # snapshot

        try:
            from .llm import get_embedding
            q_emb = get_embedding(query_text[:512])
            if q_emb is None:
                return None
            scores = []
            for c in self._chunks:
                cid = c["id"]
                chunk_emb = embeddings.get(cid)
                scores.append(
                    _cosine_dense(q_emb, chunk_emb) if chunk_emb else 0.0
                )
            return scores
        except Exception as e:
            print(f"[retriever] semantic query failed: {e}")
            return None

    # --- public query ---

    def query(self, text: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_text, score) sorted by relevance, top_k results.
        Uses semantic embeddings when available (best quality), otherwise
        falls back to pure BM25 + TF-IDF hybrid.
        """
        if self.is_empty():
            return []

        q_tokens = tokenise(text)
        if not q_tokens:
            return []

        bm25 = self._bm25_scores(q_tokens)
        cos  = self._cosine_scores(q_tokens)
        sem  = self._semantic_scores(text)

        bm25_n = _normalise_scores(bm25)
        cos_n  = _normalise_scores(cos)

        if sem is not None:
            sem_n = _normalise_scores(sem)
            # Semantic-weighted blend: 30 BM25 + 20 TF-IDF + 50 semantic
            combined = [
                (i, 0.30 * b + 0.20 * c + 0.50 * s)
                for i, (b, c, s) in enumerate(zip(bm25_n, cos_n, sem_n))
            ]
        else:
            # Fallback: classic BM25 + TF-IDF hybrid
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
