"""Build, save and load the search index (chunks + embeddings + BM25)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .embed import Embedder

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "may", "of", "on", "or", "shall", "should", "that", "the", "their", "there", "this", "to", "under", "what", "when", "where", "which", "who", "will", "with"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()) if t not in STOPWORDS]


def _search_text(c: Chunk) -> str:
    # Titles help both BM25 and embeddings: "Credit Cards ..." is a strong signal.
    return f"{c.doc_short}. {c.chapter or ''}\n{c.text}"


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedder_name: str
    bm25: BM25Okapi

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> Index:
        if not chunks:
            raise ValueError("No chunks to index.")
        texts = [_search_text(c) for c in chunks]
        emb = embedder.embed_documents(texts)
        return cls(chunks, emb, embedder.name, BM25Okapi([tokenize(t) for t in texts]))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings.astype(np.float16))
        meta = {"embedder": self.embedder_name, "n_chunks": len(self.chunks),
                "docs": sorted({c.doc_id for c in self.chunks})}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Index:
        path = directory / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No index at {directory}. Run: python -m ask_rbi.cli build")
        chunks = [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
        emb = np.load(directory / "embeddings.npy").astype(np.float32)
        meta = json.loads((directory / "meta.json").read_text())
        bm25 = BM25Okapi([tokenize(_search_text(c)) for c in chunks])
        return cls(chunks, emb, meta["embedder"], bm25)
