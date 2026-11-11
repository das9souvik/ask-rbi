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


class GroqLLM:
    """Groq's OpenAI-compatible API. Its free tier serves openai/gpt-oss-20b with no card needed."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, api_key: str, fallback: str | None = None,
                 max_tokens: int = 1500, temperature: float = 0.1, timeout: int = 60,
                 max_wait_retries: int = 8):
        self.model = model
        self.fallback = fallback
        self._key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_wait_retries = max_wait_retries

    def _call(self, model: str, system: str, user: str) -> str:
        import json
        import urllib.request

        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"  # keep the token budget for the answer itself
        req = urllib.request.Request(
            self.URL, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "User-Agent": "ask-rbi/0.1"},
        )
        import time
        import urllib.error

        for attempt in range(self.max_wait_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                return (out["choices"][0]["message"].get("content") or "").strip()
            except urllib.error.HTTPError as exc:
                # Free tier: 429 means "tokens per minute" used up. Wait it out, don't switch model.
                if exc.code != 429 or attempt == self.max_wait_retries:
                    raise
                time.sleep(_retry_after_seconds(exc.headers, attempt))
        raise RuntimeError("unreachable")

    def complete(self, system: str, user: str) -> str:
        try:
            return self._call(self.model, system, user)
        except Exception as exc:
            if not self.fallback or getattr(exc, "code", None) == 429:
                raise
            return self._call(self.fallback, system, user)


def _retry_after_seconds(headers, attempt: int) -> float:
    """Seconds to wait after a 429, from Retry-After if present, else exponential backoff."""
    value = headers.get("retry-after") if headers else None
    try:
        wait = float(value) if value is not None else 2.0 * 2**attempt
    except ValueError:
        wait = 2.0 * 2**attempt
    return min(max(wait, 1.0), 60.0) + 0.5


def make_llm(settings) -> LLM:
    """Groq when GROQ_API_KEY is set (free tier), otherwise Hugging Face Inference Providers."""
    backend = settings.llm_backend
    if backend == "auto":
        backend = "groq" if settings.groq_api_key else "hf"
    if backend == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("ASK_RBI_LLM_BACKEND=groq but GROQ_API_KEY is not set.")
        return GroqLLM(settings.llm_model, settings.groq_api_key, fallback=settings.groq_fallback_model)
    if backend == "hf":
        return HFInferenceLLM(settings.llm_model, settings.hf_token, settings.llm_provider,
                              fallback=settings.llm_fallback_model)
    raise ValueError(f"Unknown ASK_RBI_LLM_BACKEND '{backend}' (use auto, groq or hf).")


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
