import dataclasses

import pytest

from ask_rbi.config import Settings
from ask_rbi.generate import GroqLLM, HFInferenceLLM, make_llm


def settings(**kw):
    return dataclasses.replace(Settings(), **kw)


def test_auto_prefers_groq_when_key_set():
    llm = make_llm(settings(llm_backend="auto", groq_api_key="gsk_test"))
    assert isinstance(llm, GroqLLM)
    assert llm.model == "openai/gpt-oss-20b"


def test_auto_falls_back_to_hf_without_key():
    llm = make_llm(settings(llm_backend="auto", groq_api_key=None, hf_token="hf_test"))
    assert isinstance(llm, HFInferenceLLM)


def test_forcing_groq_without_key_fails_clearly():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        make_llm(settings(llm_backend="groq", groq_api_key=None))


class _FakeResp:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_groq_waits_out_rate_limit_instead_of_failing(monkeypatch):
    import io
    import json
    import urllib.error

    from ask_rbi import generate

    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"retry-after": "0"}, io.BytesIO())
        return _FakeResp(json.dumps({"choices": [{"message": {"content": "ok [S1]"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = generate.GroqLLM("openai/gpt-oss-20b", "gsk_test", fallback="other-model")
    assert llm.complete("sys", "user") == "ok [S1]"
    assert calls["n"] == 3


def test_no_fallback_model_by_default():
    assert make_llm(settings(llm_backend="groq", groq_api_key="gsk_test")).fallback is None
