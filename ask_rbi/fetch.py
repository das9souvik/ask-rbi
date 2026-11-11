"""Download the RBI Master Directions listed in the manifest.

Resolution order for each document:
1. `pdf_url` from the manifest, if given.
2. The RBI notification page (`notification_id`): check its title, then take the PDF link.
3. Otherwise, find the title on RBI's Master Directions index and follow it.

Every page is checked against the expected title before anything is downloaded,
so a wrong id in the manifest fails loudly instead of indexing the wrong rules.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

from .manifest import DocSpec

INDEX_URLS = (
    "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx",
    "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx",
)
USER_AGENT = "ask-rbi/0.1 (educational project; contact via GitHub)"
PDF_LINK_RE = re.compile(r'href\s*=\s*["\']([^"\']+?\.pdf)["\']', re.IGNORECASE)
ANCHOR_RE = re.compile(r'<a\b[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    doc_id: str
    path: Path
    url: str
    sha256: str
    title_check: float


def normalise(text: str) -> str:
    text = unescape(text).lower()
    text = re.sub(r"[‐-―\-]", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_overlap(expected: str, page_text: str) -> float:
    """Share of the expected title's words that appear in the page text."""
    want = [w for w in normalise(expected).split() if len(w) > 2]
    have = set(normalise(page_text).split())
    if not want:
        return 0.0
    return sum(w in have for w in want) / len(want)


def _get(url: str, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # network errors vary by platform
            last = exc
            time.sleep(2 * (attempt + 1))
    raise FetchError(f"Could not download {url}: {last}")


def find_pdf_link(html: str, base_url: str) -> str | None:
    links = [urljoin(base_url, unescape(m)) for m in PDF_LINK_RE.findall(html)]
    if not links:
        return None
    # Prefer RBI's notification PDFs over any annex or press-release PDFs.
    links.sort(key=lambda u: (0 if "/notification/" in u.lower() else 1, len(u)))
    return links[0]


def find_title_link(index_html: str, title: str, base_url: str, threshold: float = 0.9) -> str | None:
    best: tuple[float, str] | None = None
    for href, inner in ANCHOR_RE.findall(index_html):
        text = TAG_RE.sub(" ", inner)
        score = title_overlap(title, text)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, urljoin(base_url, unescape(href)))
    return best[1] if best else None


def resolve_pdf_url(doc: DocSpec, min_overlap: float = 0.85) -> tuple[str, float]:
    if doc.pdf_url:
        if doc.notification_url:
            page = _get(doc.notification_url).decode("utf-8", errors="ignore")
            overlap = title_overlap(doc.title, page)
            if overlap < min_overlap:
                raise FetchError(f"[{doc.id}] notification page title does not match ({overlap:.0%}).")
            return doc.pdf_url, overlap
        return doc.pdf_url, 1.0

    page_url = doc.notification_url
    if page_url is None:
        for index_url in INDEX_URLS:
            index_html = _get(index_url).decode("utf-8", errors="ignore")
            page_url = find_title_link(index_html, doc.title, index_url)
            if page_url:
                break
        if page_url is None:
            raise FetchError(f"[{doc.id}] title not found on the RBI Master Directions index.")

    page = _get(page_url).decode("utf-8", errors="ignore")
    overlap = title_overlap(doc.title, page)
    if overlap < min_overlap:
        raise FetchError(
            f"[{doc.id}] page {page_url} does not look like '{doc.title}' (title match {overlap:.0%}). "
            "Fix notification_id in data/manifest.toml."
        )
    pdf = find_pdf_link(page, page_url)
    if pdf is None:
        raise FetchError(f"[{doc.id}] no PDF link found on {page_url}.")
    return pdf, overlap


def fetch_all(docs: list[DocSpec], raw_dir: Path, force: bool = False) -> list[FetchResult]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources_path = raw_dir / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    results: list[FetchResult] = []
    for doc in docs:
        target = raw_dir / f"{doc.id}.pdf"
        if target.exists() and not force and doc.id in sources:
            meta = sources[doc.id]
            results.append(FetchResult(doc.id, target, meta["url"], meta["sha256"], meta.get("title_check", 1.0)))
            print(f"  skip  {doc.id} (already downloaded)")
            continue
        url, overlap = resolve_pdf_url(doc)
        data = _get(url)
        if not data.startswith(b"%PDF"):
            raise FetchError(f"[{doc.id}] {url} did not return a PDF.")
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        sources[doc.id] = {
            "title": doc.title,
            "url": url,
            "sha256": digest,
            "bytes": len(data),
            "title_check": round(overlap, 3),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        sources_path.write_text(json.dumps(sources, indent=2))
        results.append(FetchResult(doc.id, target, url, digest, overlap))
        print(f"  ok    {doc.id}: {len(data) / 1e6:.1f} MB from {url}")
        time.sleep(1.0)  # be polite to rbi.org.in
    return results
