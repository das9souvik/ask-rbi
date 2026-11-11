from __future__ import annotations

from pathlib import Path

import pytest

from ask_rbi.chunk import chunk_pages
from ask_rbi.parse import Page

CARDS_PAGES = [
    Page(1, "Chapter I – Preliminary\n1. Short title. These Directions shall be called the Credit Cards Directions.\n"
            "2. Applicability. These Directions apply to all commercial banks issuing cards."),
    Page(2, "Chapter III – Issue of cards\n7. Unsolicited cards. Banks shall not issue unsolicited credit cards. "
            "A card shall be issued only with the explicit consent of the applicant.\n"
            "8. Closure of credit card. Any request for closure of a credit card shall be honoured within seven working days, "
            "subject to payment of all dues by the cardholder."),
    Page(3, "9. Billing cycle. Cardholders shall be given a one-time option to modify the billing cycle of the credit card."),
]

KYC_PAGES = [
    Page(1, "Chapter VI – Customer Due Diligence\n18. Video based Customer Identification Process (V-CIP). "
            "Banks may undertake V-CIP to carry out customer due diligence with the informed consent of the customer.\n"
            "19. Officially valid document means the passport, the driving licence, proof of possession of Aadhaar number "
            "and the voter identity card."),
    Page(2, "38. Periodic updation. Banks shall carry out periodic updation of KYC at least once every two years for high "
            "risk customers, once every eight years for medium risk and once every ten years for low risk customers."),
]


@pytest.fixture
def synthetic_chunks():
    chunks = chunk_pages(CARDS_PAGES, doc_id="cards", doc_title="Credit Cards and Debit Cards Directions, 2025",
                         doc_short="Credit and Debit Cards", url="https://example.org/cards.pdf", max_words=40, overlap_words=8)
    chunks += chunk_pages(KYC_PAGES, doc_id="kyc", doc_title="Know Your Customer Directions, 2025",
                          doc_short="Know Your Customer", url="https://example.org/kyc.pdf", max_words=40, overlap_words=8)
    return chunks


def make_pdf(path: Path, pages: list[str], header: str = "RBI/DOR/2025-26/170") -> Path:
    import pymupdf

    doc = pymupdf.open()
    for i, body in enumerate(pages, 1):
        page = doc.new_page()
        page.insert_text((72, 40), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 760), body, fontsize=11)
        page.insert_text((290, 800), f"Page {i}", fontsize=9)
    doc.save(path)
    return path


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str = "Banks must not issue unsolicited cards [S1]."):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply
