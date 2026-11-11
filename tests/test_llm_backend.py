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
