"""End-to-end pipeline: build the index from PDFs, and answer questions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk, chunk_pages
from .config import Settings
from .embed import Embedder, make_embedder
from .generate import LLM, REFUSAL, Answer, answer_question, make_llm
from .index import Index
from .manifest import load_manifest
from .parse import parse_pdf
from .retrieve import Hit, Retriever

# Cosine floor below which, with zero keyword overlap, we skip the LLM and refuse.
MIN_DENSE = {"bge": 0.45, "hashing": 0.0}


def build_chunks(settings: Settings) -> list[Chunk]:
    docs = load_manifest(settings.manifest_path)
    sources_path = settings.raw_dir / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    chunks: list[Chunk] = []
    for doc in docs:
        pdf = settings.raw_dir / f"{doc.id}.pdf"
        if not pdf.exists():
            print(f"  missing {pdf.name}; run `python -m ask_rbi.cli fetch` first (skipping)")
            continue
        url = sources.get(doc.id, {}).get("url") or doc.pdf_url or doc.notification_url or ""
        pages = parse_pdf(pdf)
        doc_chunks = chunk_pages(pages, doc_id=doc.id, doc_title=doc.title, doc_short=doc.short, url=url,
                                 max_words=settings.chunk_words, overlap_words=settings.chunk_overlap_words)
        with_para = sum(c.para is not None for c in doc_chunks)
        print(f"  {doc.id:<20} {len(pages):>4} pages -> {len(doc_chunks):>4} chunks ({with_para} with para numbers)")
        chunks.extend(doc_chunks)
    return chunks


def build_index(settings: Settings, embedder: Embedder | None = None) -> Index:
    chunks = build_chunks(settings)
    embedder = embedder or make_embedder(settings.embedder, settings.embed_model)
    index = Index.build(chunks, embedder)
    index.save(settings.index_dir)
    print(f"Saved {len(chunks)} chunks to {settings.index_dir}")
    return index


@dataclass
class Result:
    answer: Answer
    hits: list[Hit]
    seconds: float


class AskRBI:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None,
                 index: Index | None = None):
        self.settings = settings
        self.embedder = embedder or make_embedder(settings.embedder, settings.embed_model)
        self.index = index or Index.load(settings.index_dir)
        self.retriever = Retriever(self.index, self.embedder, settings.candidates, settings.rrf_k)
        self.llm = llm or make_llm(settings)
        self.min_dense = MIN_DENSE.get(settings.embedder, 0.0)

    def ask(self, question: str, doc_ids: set[str] | None = None) -> Result:
        start = time.perf_counter()
        question = question.strip()
        if not question:
            return Result(Answer(question, "Please type a question.", refused=True), [], 0.0)
        hits = self.retriever.search(question, top_k=self.settings.top_k, doc_ids=doc_ids)
        if not hits or (max(h.lexical for h in hits) <= 0 and max(h.dense for h in hits) < self.min_dense):
            answer = Answer(question, REFUSAL, refused=True, model="(skipped: nothing relevant retrieved)")
        else:
            answer = answer_question(question, hits, self.llm)
        return Result(answer, hits, time.perf_counter() - start)

    @property
    def documents(self) -> list[tuple[str, str]]:
        seen: dict[str, str] = {}
        for c in self.index.chunks:
            seen.setdefault(c.doc_id, c.doc_short)
        return sorted(seen.items(), key=lambda kv: kv[1])


def default_index_dir() -> Path:
    return Settings().index_dir
