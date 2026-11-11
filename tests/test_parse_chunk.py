from ask_rbi.chunk import chunk_pages
from ask_rbi.parse import join_lines, parse_pdf, strip_repeated_lines

from .conftest import CARDS_PAGES, make_pdf

BODY = ["Banks shall not issue unsolicited cards.", "Closure within seven working days.", "Interest is reset quarterly.",
        "KYC must be updated periodically.", "Recovery agents must not harass borrowers."]


def test_strip_repeated_headers_and_page_numbers():
    pages = [["RBI/DOR/2025-26/170", "a", "b", BODY[i], "c", "d", f"Page {i + 1}"] for i in range(5)]
    cleaned = strip_repeated_lines(pages)
    for i, lines in enumerate(cleaned):
        assert "RBI/DOR/2025-26/170" not in lines and f"Page {i + 1}" not in lines
        assert BODY[i] in lines


def test_repeated_body_lines_are_kept():
    # "Provided that" repeats on every page, but in the middle, so it must survive.
    pages = [["Header X", "l1", "l2", "Provided that the bank informs the customer.", "l3", "l4", f"Page {i}"] for i in range(5)]
    cleaned = strip_repeated_lines(pages)
    assert all("Provided that the bank informs the customer." in lines for lines in cleaned)


def test_join_lines_rejoins_wrapped_text_but_keeps_paragraphs():
    text = join_lines(["7. Banks shall not issue", "unsolicited cards.", "8. Closure of a card", "within seven work-", "ing days."])
    assert text.split("\n") == ["7. Banks shall not issue unsolicited cards.", "8. Closure of a card within seven working days."]


def test_parse_real_pdf_drops_header(tmp_path):
    pdf = make_pdf(tmp_path / "t.pdf", [p.text for p in CARDS_PAGES])
    pages = parse_pdf(pdf)
    assert len(pages) == 3
    joined = "\n".join(p.text for p in pages)
    assert "RBI/DOR/2025-26/170" not in joined
    assert "unsolicited" in joined


def test_chunks_keep_paragraph_numbers_and_pages(synthetic_chunks):
    cards = [c for c in synthetic_chunks if c.doc_id == "cards"]
    assert cards, "expected chunks"
    closure = next(c for c in cards if "seven working days" in c.text)
    assert closure.para in {"7", "8"}
    assert closure.page_start == 2
    assert closure.citation.startswith("Credit and Debit Cards, para")
    # A chunk spanning several paragraphs cites the whole range, not just the first.
    multi = next(c for c in cards if c.para_end and c.para_end != c.para)
    assert f"paras {multi.para}–{multi.para_end}" in multi.citation


def test_two_page_documents_lose_headers_too():
    pages = [["RBI/DOR/2025-26/170", "x1", "x2", "Banks shall act fairly.", "x3", "x4", "Page 1"],
             ["RBI/DOR/2025-26/170", "y1", "y2", "Customers may complain.", "y3", "y4", "Page 2"]]
    cleaned = strip_repeated_lines(pages)
    assert all("RBI/DOR/2025-26/170" not in lines for lines in cleaned)


def test_chunk_ids_unique_and_no_tiny_chunks(synthetic_chunks):
    ids = [c.chunk_id for c in synthetic_chunks]
    assert len(ids) == len(set(ids))
    assert all(len(c.text.split()) >= 12 for c in synthetic_chunks)


def test_long_paragraph_is_split():
    from ask_rbi.parse import Page

    long = "5. " + " ".join(f"Sentence {i} about deposits." for i in range(200))
    chunks = chunk_pages([Page(1, long)], doc_id="d", doc_title="D", doc_short="D", url="", max_words=50, overlap_words=10)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 80 for c in chunks)
