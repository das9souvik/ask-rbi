from ask_rbi.embed import HashingEmbedder
from ask_rbi.generate import REFUSAL, answer_question, parse_citations
from ask_rbi.index import Index
from ask_rbi.retrieve import Retriever, reciprocal_rank_fusion

from .conftest import FakeLLM


def make_retriever(chunks):
    emb = HashingEmbedder()
    return Retriever(Index.build(chunks, emb), emb, candidates=20)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in {1, 2}
    assert fused[4] < fused[1]


def test_hybrid_search_finds_right_document(synthetic_chunks):
    r = make_retriever(synthetic_chunks)
    assert r.search("closing a credit card account", top_k=1)[0].chunk.doc_id == "cards"
    assert r.search("how often periodic updation KYC high risk", top_k=1)[0].chunk.doc_id == "kyc"


def test_document_filter(synthetic_chunks):
    hits = make_retriever(synthetic_chunks).search("consent", top_k=5, doc_ids={"kyc"})
    assert hits and all(h.chunk.doc_id == "kyc" for h in hits)


def test_index_roundtrip(tmp_path, synthetic_chunks):
    idx = Index.build(synthetic_chunks, HashingEmbedder())
    idx.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert len(loaded.chunks) == len(synthetic_chunks)
    assert loaded.embedder_name == "hashing-512"


def test_answer_keeps_valid_citations_and_flags_fake_ones(synthetic_chunks):
    hits = make_retriever(synthetic_chunks).search("unsolicited credit card", top_k=3)
    llm = FakeLLM("No unsolicited cards [S1]. Closure in seven working days [S2]. Made up [S9].")
    ans = answer_question("q", hits, llm)
    assert [c.tag for c in ans.citations] == ["S1", "S2"]
    assert ans.invalid_tags == ["[S9]"]
    assert "[S9]" not in ans.text
    assert not ans.grounded  # a fake citation means not fully grounded


def test_refusal_and_empty_hits(synthetic_chunks):
    hits = make_retriever(synthetic_chunks).search("unsolicited credit card", top_k=2)
    assert answer_question("q", hits, FakeLLM(REFUSAL)).refused
    llm = FakeLLM()
    assert answer_question("q", [], llm).refused and llm.calls == 0


def test_parse_citations_dedupes(synthetic_chunks):
    hits = make_retriever(synthetic_chunks).search("credit card", top_k=2)
    cites, invalid = parse_citations("A [S1] B [S1] C [S2]", hits)
    assert [c.tag for c in cites] == ["S1", "S2"] and not invalid


def test_answer_without_tags_is_repaired_once():
    from ask_rbi.chunk import Chunk
    from ask_rbi.generate import answer_question
    from ask_rbi.retrieve import Hit

    chunk = Chunk("c1", "cards", "Credit Cards", "Cards", "https://x", "7", None, 1, 1, "Close within seven working days.")
    hits = [Hit(chunk, 1.0, 1.0, 1.0)]

    class TwoStep:
        model = "fake"
        calls = 0

        def complete(self, system, user):
            self.calls += 1
            return "Close it within seven working days." if self.calls == 1 else "Close it within seven working days [S1]."

    llm = TwoStep()
    a = answer_question("How fast must a card be closed?", hits, llm)
    assert llm.calls == 2 and a.grounded and a.citations[0].doc_id == "cards"
