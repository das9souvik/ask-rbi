"""Ask RBI: Gradio demo (runs locally and on Hugging Face Spaces)."""

from __future__ import annotations

import html
import os

import gradio as gr

from ask_rbi.config import get_settings
from ask_rbi.generate import DISCLAIMER
from ask_rbi.pipeline import AskRBI

GITHUB_URL = os.environ.get("ASK_RBI_GITHUB_URL", "https://github.com/das9souvik/ask-rbi")
MAX_QUESTION_CHARS = 500

EXAMPLES = [
    "How quickly must a bank close my credit card after I ask?",
    "Can a bank charge foreclosure charges on my floating rate home loan?",
    "At what times can recovery agents call a borrower?",
    "How often must banks update KYC for high-risk customers?",
    "When does a loan become a non-performing asset?",
    "Do payment aggregators have to keep funds in an escrow account?",
]

settings = get_settings()
engine = AskRBI(settings)
DOC_CHOICES = [(short, doc_id) for doc_id, short in engine.documents]


def _sources_html(result) -> str:
    cited = {c.tag for c in result.answer.citations}
    cards = []
    for i, hit in enumerate(result.hits, 1):
        c = hit.chunk
        tag = f"S{i}"
        link = f"{c.url}#page={c.page_start}" if c.url else ""
        badge = "cited" if tag in cited else "retrieved"
        title = html.escape(c.citation)
        text = html.escape(c.text[:600]) + ("…" if len(c.text) > 600 else "")
        open_link = f' · <a href="{html.escape(link)}" target="_blank" rel="noopener">open PDF p.{c.page_start}</a>' if link else ""
        cards.append(
            f'<div class="src {badge}"><div class="src-head"><b>[{tag}]</b> {title}'
            f'<span class="badge">{badge}</span>{open_link}</div><div class="src-body">{text}</div></div>'
        )
    return "".join(cards) or "<p>No sources retrieved.</p>"


def ask(question: str, docs: list[str] | None):
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        return "Type a question about RBI rules for banks, cards, KYC, loans or payments.", ""
    try:
        result = engine.ask(question, doc_ids=set(docs) if docs else None)
    except Exception as exc:  # the free model endpoint can be busy or rate-limited
        return f"The answer model is busy right now ({type(exc).__name__}). Please try again in a minute.", ""
    a = result.answer
    note = f"\n\n<sub>{len(a.citations)} cited source(s) · {result.seconds:.1f}s · {html.escape(a.model)}</sub>"
    if a.invalid_tags:
        note += "\n\n<sub>⚠️ The model referenced a source that doesn't exist; that reference was removed.</sub>"
    return a.text + note, _sources_html(result)


CSS = """
.src {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; margin:8px 0;}
.src.cited {border-left:4px solid #2f7d5b;}
.src-head {font-size:0.92em; margin-bottom:6px;}
.src-body {font-size:0.85em; opacity:0.85; white-space:pre-wrap;}
.badge {margin-left:8px; font-size:0.75em; padding:1px 7px; border-radius:999px; background:var(--background-fill-secondary);}
footer {display:none !important;}
"""

with gr.Blocks(title="Ask RBI") as demo:
    gr.Markdown(
        "# Ask RBI\n"
        "Plain-English answers from the Reserve Bank of India's **2025 consolidated Master Directions**, "
        "with every claim linked to the exact paragraph.\n\n"
        f"<sub>{DISCLAIMER} · [Code and evaluation on GitHub]({GITHUB_URL})</sub>"
    )
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Your question", placeholder="e.g. Can my bank charge for closing a credit card?",
                                  lines=2, max_lines=4)
            docs = gr.Dropdown(DOC_CHOICES, multiselect=True, label="Limit to documents (optional)")
            go = gr.Button("Ask", variant="primary")
            gr.Examples(EXAMPLES, inputs=question)
        with gr.Column(scale=4):
            answer = gr.Markdown(label="Answer")
            gr.Markdown("### Sources")
            sources = gr.HTML()
    go.click(ask, [question, docs], [answer, sources], concurrency_limit=2)
    question.submit(ask, [question, docs], [answer, sources], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(css=CSS, theme=gr.themes.Soft())
