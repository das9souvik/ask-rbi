"""Load the document manifest (data/manifest.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocSpec:
    id: str
    title: str
    short: str
    notification_id: int | None = None
    pdf_url: str | None = None
    topics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def notification_url(self) -> str | None:
        if self.notification_id is None:
            return None
        return f"https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={self.notification_id}&Mode=0"


def load_manifest(path: Path) -> list[DocSpec]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    docs = []
    seen: set[str] = set()
    for raw in data.get("doc", []):
        doc = DocSpec(
            id=raw["id"],
            title=raw["title"],
            short=raw.get("short", raw["title"]),
            notification_id=raw.get("notification_id"),
            pdf_url=raw.get("pdf_url"),
            topics=tuple(raw.get("topics", [])),
        )
        if doc.id in seen:
            raise ValueError(f"Duplicate document id in manifest: {doc.id}")
        seen.add(doc.id)
        docs.append(doc)
    return docs
