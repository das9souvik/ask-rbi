"""Turn RBI PDFs into clean page text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    number: int  # 1-based
    text: str


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _edge_positions(lines: list[str], edge: int) -> list[int]:
    """Indexes of the first and last `edge` non-empty lines: where headers and footers live."""
    filled = [i for i, ln in enumerate(lines) if ln]
    return sorted(set(filled[:edge] + filled[-edge:]))


def strip_repeated_lines(pages: list[list[str]], min_share: float = 0.5, edge: int = 3) -> list[list[str]]:
    """Remove headers and footers that repeat on most pages (e.g. 'RBI/DOR/2025-26/170', 'Page 4').

    Only the top and bottom few lines of each page are candidates, so repeated phrases
    in the body text are never removed.
    """
    if len(pages) < 2:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line)  # page numbers vary

    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({key(lines[i]) for i in _edge_positions(lines, edge)})
    limit = max(2, int(len(pages) * min_share))
    repeated = {k for k, c in counts.items() if c >= limit and len(k) < 120}
    out = []
    for lines in pages:
        edges = set(_edge_positions(lines, edge))
        out.append([ln for i, ln in enumerate(lines) if not (i in edges and key(ln) in repeated)])
    return out


def join_lines(lines: list[str]) -> str:
    """Rejoin hard-wrapped PDF lines while keeping real paragraph breaks."""
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        starts_block = bool(re.match(r"^(\d{1,3}(\.\d{1,3}){0,3}\.?|\([a-z0-9ivx]{1,5}\)|Chapter\b|CHAPTER\b|Section\b|Annex)", line))
        if out and not starts_block and not out[-1].endswith((".", ":", ";")):
            if out[-1].endswith("-") and line[:1].islower():
                out[-1] = out[-1][:-1] + line
            else:
                out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return "\n".join(out)


def parse_pdf(path: Path) -> list[Page]:
    import pymupdf  # imported lazily so tests can run without it

    raw_pages: list[list[str]] = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            raw_pages.append([_clean_line(ln) for ln in page.get_text("text").splitlines()])
    cleaned = strip_repeated_lines(raw_pages)
    return [Page(i + 1, join_lines(lines)) for i, lines in enumerate(cleaned)]
