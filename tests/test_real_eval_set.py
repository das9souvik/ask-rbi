from ask_rbi.config import get_settings
from ask_rbi.evaluate import load_cases
from ask_rbi.manifest import load_manifest


def test_eval_set_references_real_documents():
    doc_ids = {d.id for d in load_manifest(get_settings().manifest_path)}
    cases = load_cases()
    assert len(cases) >= 25
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert set(c.expected_docs) <= doc_ids, c.id
        if c.out_of_scope:
            assert not c.expected_docs
        else:
            assert c.expected_docs and c.keywords, c.id
