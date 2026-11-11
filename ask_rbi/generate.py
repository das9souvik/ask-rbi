"""Answer generation with strict, checkable citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .retrieve import Hit

REFUSAL = "I couldn't find this in the RBI Master Directions I cover."
DISCLAIMER = "Educational tool, not legal or compliance advice. Always check the official RBI text."

SYSTEM_PROMPT = f"""You answer questions about Reserve Bank of India (RBI) regulations for commercial banks and payment companies.

Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge, and never guess numbers, limits or dates.
2. Cite every factual sentence with its source tag, like [S2]. Use only tags that appear in the sources.
3. If the sources do not answer the question, reply with exactly: "{REFUSAL}" and nothing else.
4. If the sources answer only part of it, answer that part and say what is not covered.
5. Be concise: a direct answer first, then up to 4 short bullet points with key conditions or exceptions.
6. Plain English. Keep RBI's exact figures and terms (e.g. "seven working days")."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class HFInferenceLLM:
    """Hugging Face Inference Providers (free monthly credits on a free HF account)."""

    def __init__(self, model: str, token: str | None, provider: str = "auto", fallback: str | None = None,
                 max_tokens: int = 700, temperature: float = 0.1):
        from huggingface_hub import InferenceClient

        self.model = model
        self.fallback = fallback
        self._client = InferenceClient(provider=provider, api_key=token)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call(self, model: str, system: str, user: str) -> str:
        out = self._client.chat_completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (out.choices[0].message.content or "").strip()

    def complete(self, system: str, user: str) -> str:
        try:
            return self._call(self.model, system, user)
        except Exception:
            if not self.fallback:
                raise
            return self._call(self.fallback, system, user)


@dataclass
class Citation:
    tag: str
    doc_id: str
    doc_title: str
    label: str
    url: str
    page_start: int
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    invalid_tags: list[str] = field(default_factory=list)
    model: str = ""

    @property
    def grounded(self) -> bool:
        """True when the answer refused, or cites at least one real source and no fake ones."""
        return self.refused or (bool(self.citations) and not self.invalid_tags)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        parts.append(f"[S{i}] {c.doc_title} | {c.citation} | page {c.page_start}\n{c.text}")
    return "Sources:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQuestion: {question}\n\nAnswer (cite sources like [S1]):"


TAG_RE = re.compile(r"\[S(\d+)\]")


def is_refusal(text: str) -> bool:
    t = text.strip().strip('"').lower()
    return t.startswith(REFUSAL.lower()[:40]) or (len(t) < 200 and "couldn't find" in t)


def parse_citations(text: str, hits: list[Hit]) -> tuple[list[Citation], list[str]]:
    used: list[Citation] = []
    invalid: list[str] = []
    seen: set[int] = set()
    for m in TAG_RE.finditer(text):
        n = int(m.group(1))
        if not 1 <= n <= len(hits):
            invalid.append(m.group(0))
            continue
        if n in seen:
            continue
        seen.add(n)
        c = hits[n - 1].chunk
        snippet = c.text[:280] + ("…" if len(c.text) > 280 else "")
        used.append(Citation(f"S{n}", c.doc_id, c.doc_title, c.citation, c.url, c.page_start, snippet))
    return used, invalid


def answer_question(question: str, hits: list[Hit], llm: LLM) -> Answer:
    if not hits:
        return Answer(question, REFUSAL, refused=True, model=llm.model)
    raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
    if is_refusal(raw):
        return Answer(question, REFUSAL, refused=True, model=llm.model)
    citations, invalid = parse_citations(raw, hits)
    text = TAG_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", raw)
    return Answer(question, text.strip(), citations, refused=False, invalid_tags=invalid, model=llm.model)
