"""Text embedders.

`BGEEmbedder` is the real one (BAAI/bge-small-en-v1.5, 384 dims, runs on CPU).
`HashingEmbedder` needs no downloads; tests and CI use it so they run offline.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from typing import Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=32, show_progress_bar=len(texts) > 64, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


class HashingEmbedder:
    """Bag of word and bigram hashes. Crude, deterministic and dependency-free."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vec(text)[None, :])[0]


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "bge":
        return BGEEmbedder(model_name)
    raise ValueError(f"Unknown embedder '{kind}' (use 'bge' or 'hashing').")
