"""Paragraph-aware chunking that keeps RBI paragraph numbers for citations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .parse import Page

# Matches RBI paragraph numbers at the start of a block: "5.", "12.3", "4.2.1".
PARA_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+(?=\S)")
CHAPTER_RE = re.compile(r"^(chapter|section)\s+([ivxlc0-9]+)\b[\s:–-]*(.*)$", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_short: str
    url: str
    para: str | None  # first RBI paragraph number in the chunk
    chapter: str | None
    page_start: int
    page_end: int
    text: str
    para_end: str | None = None  # last RBI paragraph number in the chunk

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        if not self.para:
            pages = f"p. {self.page_start}" if self.page_end == self.page_start else f"pp. {self.page_start}–{self.page_end}"
            return f"{self.doc_short}, {pages}"
        if self.para_end and self.para_end != self.para:
            return f"{self.doc_short}, paras {self.para}–{self.para_end}"
        return f"{self.doc_short}, para {self.para}"


@dataclass
class _Block:
    text: str
    para: str | None
    chapter: str | None
    page: int


def to_blocks(pages: list[Page]) -> list[_Block]:
    blocks: list[_Block] = []
    para: str | None = None
    chapter: str | None = None
    for page in pages:
        for line in page.text.split("\n"):
            line = line.strip()
            if not line:
                continue
            ch = CHAPTER_RE.match(line)
            if ch:
                chapter = f"{ch.group(1).title()} {ch.group(2).upper()}" + (f" – {ch.group(3).strip()}" if ch.group(3).strip() else "")
            m = PARA_RE.match(line)
            if m and _looks_like_para(m.group(1), para):
                para = m.group(1)
            blocks.append(_Block(line, para, chapter, page.number))
    return blocks


def _looks_like_para(candidate: str, previous: str | None) -> bool:
    """Reject numbers that are clearly amounts or years, e.g. '2025 the bank...'."""
    head = candidate.split(".")[0]
    return int(head) < 400


def chunk_pages(pages: list[Page], *, doc_id: str, doc_title: str, doc_short: str, url: str,
                max_words: int = 260, overlap_words: int = 40) -> list[Chunk]:
    blocks = to_blocks(pages)
    chunks: list[Chunk] = []
    current: list[_Block] = []
    words = 0
    fresh = 0  # blocks added since the last flush (excludes carried-over overlap)

    def flush() -> None:
        nonlocal current, words, fresh
        if not current or fresh == 0:
            return
        text = "\n".join(b.text for b in current).strip()
        para = next((b.para for b in current if b.para), None)
        para_end = next((b.para for b in reversed(current) if b.para), None)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}-{len(chunks):04d}",
            doc_id=doc_id, doc_title=doc_title, doc_short=doc_short, url=url,
            para=para, chapter=current[0].chapter,
            page_start=current[0].page, page_end=current[-1].page, text=text, para_end=para_end,
        ))
        # Carry a short tail forward so answers spanning a boundary still match.
        tail: list[_Block] = []
        tail_words = 0
        for b in reversed(current):
            n = len(b.text.split())
            if tail_words + n > overlap_words:
                break
            tail.insert(0, b)
            tail_words += n
        current, words, fresh = tail, tail_words, 0

    for block in blocks:
        n = len(block.text.split())
        if n > max_words:  # a single huge block: split it on sentence-ish boundaries
            flush()
            sentences = re.split(r"(?<=[.;:])\s+", block.text)
            piece: list[str] = []
            for s in sentences:
                piece.append(s)
                if len(" ".join(piece).split()) >= max_words:
                    current = [_Block(" ".join(piece), block.para, block.chapter, block.page)]
                    words, fresh = max_words, 1
                    flush()
                    piece = []
            if piece:
                current.append(_Block(" ".join(piece), block.para, block.chapter, block.page))
                words += len(" ".join(piece).split())
                fresh += 1
            continue
        if words + n > max_words and words > overlap_words:
            flush()
        current.append(block)
        words += n
        fresh += 1
    flush()
    return [c for c in chunks if len(c.text.split()) >= 12]
