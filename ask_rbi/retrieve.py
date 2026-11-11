"""Hybrid retrieval: BM25 (exact regulatory terms) + embeddings (meaning), fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunk import Chunk
from .embed import Embedder
from .index import Index, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused reciprocal-rank score
    dense: float  # cosine similarity
    lexical: float  # BM25 score


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, candidates: int = 40, rrf_k: int = 60):
        if embedder.name != index.embedder_name:
            raise ValueError(f"Index was built with '{index.embedder_name}' but the embedder is '{embedder.name}'.")
        self.index = index
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 6, doc_ids: set[str] | None = None) -> list[Hit]:
        n = len(self.index.chunks)
        allowed = np.ones(n, dtype=bool)
        if doc_ids:
            allowed = np.array([c.doc_id in doc_ids for c in self.index.chunks])

        dense = self.index.embeddings @ self.embedder.embed_query(query)
        lexical = np.asarray(self.index.bm25.get_scores(tokenize(query)), dtype=np.float32)

        def ranked(scores: np.ndarray) -> list[int]:
            masked = np.where(allowed, scores, -np.inf)
            order = np.argsort(-masked)[: self.candidates]
            return [int(i) for i in order if np.isfinite(masked[i])]

        dense_rank = ranked(dense)
        lexical_rank = [i for i in ranked(lexical) if lexical[i] > 0]
        fused = reciprocal_rank_fusion([dense_rank, lexical_rank], k=self.rrf_k)
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [Hit(self.index.chunks[i], fused[i], float(dense[i]), float(lexical[i])) for i in best]
