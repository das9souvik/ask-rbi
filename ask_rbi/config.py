"""Settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def use_system_certificates() -> bool:
    """Verify HTTPS with the operating system's trust store (macOS Keychain, Windows store).

    Standalone Python builds ship their own certificate bundle, which doesn't include
    certificates your OS already trusts (e.g. from a security app or network that
    inspects HTTPS). Verification stays fully on; we just use the same trust list
    as your browser. Set ASK_RBI_SYSTEM_CERTS=0 to disable.
    """
    if os.environ.get("ASK_RBI_SYSTEM_CERTS", "1") == "0":
        return False
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    return True


use_system_certificates()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    manifest_path: Path = field(default_factory=lambda: Path(_env("ASK_RBI_MANIFEST", str(ROOT / "data" / "manifest.toml"))))
    raw_dir: Path = field(default_factory=lambda: Path(_env("ASK_RBI_RAW_DIR", str(ROOT / "data" / "raw"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("ASK_RBI_INDEX_DIR", str(ROOT / "index"))))

    # Embeddings: "bge" uses sentence-transformers locally; "hashing" is a
    # dependency-free fallback used in tests and CI.
    embedder: str = field(default_factory=lambda: _env("ASK_RBI_EMBEDDER", "bge"))
    embed_model: str = field(default_factory=lambda: _env("ASK_RBI_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # Answer model: same open model on either backend (Groq free tier or HF Inference Providers).
    llm_model: str = field(default_factory=lambda: _env("ASK_RBI_LLM_MODEL", "openai/gpt-oss-20b"))
    llm_fallback_model: str = field(default_factory=lambda: _env("ASK_RBI_LLM_FALLBACK", "Qwen/Qwen2.5-7B-Instruct"))
    llm_provider: str = field(default_factory=lambda: _env("ASK_RBI_LLM_PROVIDER", "auto"))
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    # "auto" uses Groq's free tier when GROQ_API_KEY is set, else Hugging Face.
    llm_backend: str = field(default_factory=lambda: _env("ASK_RBI_LLM_BACKEND", "auto"))
    groq_api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY") or None)
    groq_fallback_model: str = field(default_factory=lambda: _env("ASK_RBI_GROQ_FALLBACK", "llama-3.1-8b-instant"))

    # Chunking and retrieval.
    chunk_words: int = 260
    chunk_overlap_words: int = 40
    top_k: int = 6
    candidates: int = 40
    rrf_k: int = 60
    # Below this fused-relevance threshold we treat the question as out of scope.
    min_best_score: float = 0.0


def get_settings() -> Settings:
    return Settings()
