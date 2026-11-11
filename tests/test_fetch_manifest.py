from ask_rbi.config import get_settings
from ask_rbi.fetch import find_pdf_link, find_title_link, title_overlap
from ask_rbi.manifest import load_manifest


def test_manifest_loads_with_unique_ids():
    docs = load_manifest(get_settings().manifest_path)
    assert len(docs) >= 10
    assert len({d.id for d in docs}) == len(docs)
    assert all(d.title for d in docs)


def test_title_overlap_handles_dashes_and_case():
    title = "Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025"
    page = "<h2>RESERVE BANK OF INDIA (COMMERCIAL BANKS - KNOW YOUR CUSTOMER) DIRECTIONS, 2025</h2>"
    assert title_overlap(title, page) == 1.0
    assert title_overlap(title, "Credit Cards and Debit Cards Directions") < 0.5


def test_find_pdf_link_prefers_notification_pdf():
    html = ('<a href="/rdocs/PressRelease/PDFs/PR1.PDF">press</a>'
            '<a href="https://rbidocs.rbi.org.in/rdocs/notification/PDFs/170MD.pdf">main</a>')
    assert find_pdf_link(html, "https://www.rbi.org.in/x").endswith("/notification/PDFs/170MD.pdf")


def test_find_title_link_matches_exact_document():
    html = ('<a href="NotificationUser.aspx?Id=1">Reserve Bank of India (Commercial Banks – Governance) Directions, 2025</a>'
            '<a href="NotificationUser.aspx?Id=2">Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025</a>')
    link = find_title_link(html, "Reserve Bank of India (Commercial Banks – Know Your Customer) Directions, 2025",
                           "https://www.rbi.org.in/Scripts/")
    assert link == "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=2"
