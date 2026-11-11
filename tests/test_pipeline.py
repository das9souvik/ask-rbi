import json
from dataclasses import replace

from ask_rbi.config import Settings
from ask_rbi.embed import HashingEmbedder
from ask_rbi.pipeline import AskRBI, build_index

from .conftest import CARDS_PAGES, KYC_PAGES, FakeLLM, make_pdf


def make_settings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "cards.pdf", [p.text for p in CARDS_PAGES])
    make_pdf(raw / "kyc.pdf", [p.text for p in KYC_PAGES])
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        '[[doc]]\nid = "cards"\ntitle = "Credit Cards Directions, 2025"\nshort = "Credit and Debit Cards"\n\n'
        '[[doc]]\nid = "kyc"\ntitle = "Know Your Customer Directions, 2025"\nshort = "Know Your Customer"\n'
    )
    (raw / "sources.json").write_text(json.dumps({"cards": {"url": "https://example.org/cards.pdf"}}))
    return replace(Settings(), manifest_path=manifest, raw_dir=raw, index_dir=tmp_path / "index",
                   embedder="hashing", chunk_words=60, chunk_overlap_words=10)


def test_build_then_ask_end_to_end(tmp_path):
    settings = make_settings(tmp_path)
    build_index(settings, HashingEmbedder())
    llm = FakeLLM("Requests to close a card must be honoured within seven working days [S1].")
    app = AskRBI(settings, embedder=HashingEmbedder(), llm=llm)
    res = app.ask("How fast must a bank close my credit card?")
    assert not res.answer.refused
    assert res.answer.citations[0].doc_id == "cards"
    assert res.answer.citations[0].url == "https://example.org/cards.pdf"
    assert ("cards", "Credit and Debit Cards") in app.documents


def test_blank_question(tmp_path):
    settings = make_settings(tmp_path)
    build_index(settings, HashingEmbedder())
    app = AskRBI(settings, embedder=HashingEmbedder(), llm=FakeLLM())
    assert app.ask("   ").answer.refused


def test_eval_runs_offline(tmp_path, monkeypatch):
    from ask_rbi import evaluate

    settings = make_settings(tmp_path)
    build_index(settings, HashingEmbedder())
    monkeypatch.setattr(evaluate, "EVAL_DIR", tmp_path / "eval")
    cases = [
        evaluate.Case("a", "credit card closure", ["cards"], ["closure"]),
        evaluate.Case("b", "poem please", [], [], True),
    ]
    app = AskRBI(settings, embedder=HashingEmbedder(), llm=FakeLLM("Seven working days [S1]."))
    summary = evaluate.run_eval(settings, use_llm=True, app=app, cases=cases)
    assert summary["doc_hit@k"] == "100%"
    assert summary["grounded_rate"] == "100%"
    assert (tmp_path / "eval" / "results.md").exists()
