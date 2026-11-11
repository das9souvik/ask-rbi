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

EXTERNAL_ICON = (
    '<svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/>'
    '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>'
)
EMPTY_ANSWER = (
    '<div class="empty"><p class="empty-title">Ask a question to get started</p>'
    "<p>Answers quote the RBI Master Directions and link each claim to the exact paragraph and page.</p></div>"
)
EMPTY_SOURCES = '<p class="muted">Sources used for the answer will appear here.</p>'


def _sources_html(result) -> str:
    cited = {c.tag for c in result.answer.citations}
    cards = []
    for i, hit in enumerate(result.hits, 1):
        c = hit.chunk
        tag = f"S{i}"
        is_cited = tag in cited
        text = html.escape(c.text[:600]) + ("…" if len(c.text) > 600 else "")
        link = ""
        if c.url:
            href = html.escape(f"{c.url}#page={c.page_start}")
            link = (f'<a class="src-link" href="{href}" target="_blank" rel="noopener">'
                    f"PDF p. {c.page_start} {EXTERNAL_ICON}</a>")
        badge = '<span class="badge cited">Cited</span>' if is_cited else '<span class="badge">Retrieved</span>'
        cards.append(
            f'<article class="src{" is-cited" if is_cited else ""}">'
            f'<header class="src-head"><span class="tag">{tag}</span>'
            f'<span class="src-title">{html.escape(c.citation)}</span>{badge}{link}</header>'
            f'<p class="src-body">{text}</p></article>'
        )
    return "".join(cards) or '<p class="muted">No sources retrieved.</p>'


def ask(question: str, docs: list[str] | None):
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        return '<div class="notice">Type a question about RBI rules for banks, cards, KYC, loans or payments.</div>', EMPTY_SOURCES
    try:
        result = engine.ask(question, doc_ids=set(docs) if docs else None)
    except Exception as exc:  # the free model endpoint can be busy or rate-limited
        return (f'<div class="notice">The answer model is busy right now ({html.escape(type(exc).__name__)}). '
                "Please try again in a minute.</div>"), EMPTY_SOURCES
    a = result.answer
    meta = (f'<div class="meta"><span>{len(a.citations)} cited source{"s" if len(a.citations) != 1 else ""}</span>'
            f"<span>{result.seconds:.1f}s</span><span>{html.escape(a.model)}</span></div>")
    warn = ('<div class="notice">The model referenced a source that doesn\'t exist; that reference was removed.</div>'
            if a.invalid_tags else "")
    return a.text + "\n\n" + meta + warn, _sources_html(result)


HEADER = f"""
<div class="hero">
  <p class="eyebrow">Independent project · not affiliated with the Reserve Bank of India</p>
  <h1>Ask RBI</h1>
  <p class="lede">Plain-English answers from the Reserve Bank of India's 2025 consolidated Master Directions,
  with every claim linked to the exact paragraph.</p>
  <ul class="stats">
    <li><b>{len(DOC_CHOICES)}</b> Master Directions</li>
    <li><b>{len(engine.index.chunks):,}</b> indexed passages</li>
    <li><b>Paragraph-level</b> citations</li>
  </ul>
</div>
"""

FOOTER = (f'<p class="foot">{html.escape(DISCLAIMER)} · '
          f'<a href="{html.escape(GITHUB_URL)}" target="_blank" rel="noopener">Code and evaluation on GitHub</a></p>')

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#F6F7F9",
    body_background_fill_dark="#0B1220",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#111A2C",
    block_border_color="#E2E6EC",
    block_border_color_dark="#1F2A3F",
    block_shadow="none",
    block_shadow_dark="none",
    button_primary_background_fill="#1E3A5F",
    button_primary_background_fill_hover="#16304F",
    button_primary_background_fill_dark="#3B6EA8",
    button_primary_background_fill_hover_dark="#4A7DB8",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    input_border_color_focus="#1E3A5F",
    input_border_color_focus_dark="#6B9BD1",
)

CSS = """
:root { --ink:#0F1B2D; --muted:#5B6678; --line:#E2E6EC; --navy:#1E3A5F; --gold:#A87B1F;
        --cited:#1F7A4D; --cited-bg:#E8F4EE; --chip:#EEF1F5; --warn-bg:#FFF6E5; --warn:#7A5300; }
.dark { --ink:#E6EBF2; --muted:#9AA6B8; --line:#1F2A3F; --navy:#6B9BD1; --gold:#D6A84A;
        --cited:#5CC08C; --cited-bg:#11291E; --chip:#18233A; --warn-bg:#2A2110; --warn:#E8C27A; }

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }

.hero { padding: 28px 4px 8px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }
.hero h1 { font-size: 2rem; line-height: 1.15; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); margin: 4px 0 8px; }
.hero .eyebrow { font-size: .78rem; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; color: var(--gold); margin: 0; }
.hero .lede { font-size: 1.02rem; line-height: 1.55; color: var(--muted); max-width: 62ch; margin: 0 0 16px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 24px; list-style: none; padding: 0; margin: 0 0 16px; }
.stats li { font-size: .88rem; color: var(--muted); }
.stats b { color: var(--ink); font-weight: 600; }

.section-label { font-size: .78rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); margin: 4px 0 6px; }

#ask-btn { min-height: 44px; font-weight: 600; cursor: pointer; }
.gallery button, .gallery-item { text-align: left !important; justify-content: flex-start !important; }

#answer { min-height: 180px; padding: 20px 22px !important; }
#answer, #answer p, #answer li { font-size: 1rem; line-height: 1.6; color: var(--ink); }
.empty { color: var(--muted); }
.empty-title { font-weight: 600; color: var(--ink); margin-bottom: 4px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.meta span { font-size: .78rem; padding: 3px 9px; border-radius: 999px; background: var(--chip); color: var(--muted); font-family: var(--font-mono); }
.notice { margin-top: 10px; padding: 10px 12px; border-radius: 8px; background: var(--warn-bg); color: var(--warn); font-size: .92rem; }
.muted { color: var(--muted); font-size: .92rem; }

.src { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin: 0 0 10px; background: var(--block-background-fill); }
.src.is-cited { border-left: 3px solid var(--cited); }
.src-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 6px; }
.tag { font-family: var(--font-mono); font-size: .78rem; font-weight: 600; color: var(--navy); }
.src-title { font-size: .92rem; font-weight: 500; color: var(--ink); flex: 1 1 220px; }
.badge { font-size: .72rem; font-weight: 500; padding: 2px 8px; border-radius: 999px; background: var(--chip); color: var(--muted); }
.badge.cited { background: var(--cited-bg); color: var(--cited); }
.src-link { display: inline-flex; align-items: center; gap: 4px; font-size: .82rem; color: var(--navy) !important; text-decoration: none; }
.src-link:hover { text-decoration: underline; }
.src-link:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; border-radius: 4px; }
.src-body { font-size: .88rem; line-height: 1.55; color: var(--muted); white-space: pre-wrap; margin: 0; }

.foot { font-size: .82rem; color: var(--muted); text-align: center; padding: 18px 0 8px; border-top: 1px solid var(--line); margin-top: 12px; }
.foot a { color: var(--navy); }

@media (max-width: 768px) { .hero h1 { font-size: 1.6rem; } #answer { padding: 16px !important; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

with gr.Blocks(title="Ask RBI") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            question = gr.Textbox(label="Your question", placeholder="e.g. Can my bank charge for closing a credit card?",
                                  lines=3, max_lines=6, max_length=MAX_QUESTION_CHARS)
            docs = gr.Dropdown(DOC_CHOICES, multiselect=True, label="Limit to specific directions (optional)",
                               info="Leave empty to search all of them.")
            go = gr.Button("Ask", variant="primary", elem_id="ask-btn")
            gr.Examples(EXAMPLES, inputs=question, label="Try an example")
        with gr.Column(scale=7, min_width=320):
            gr.HTML('<p class="section-label">Answer</p>')
            answer = gr.Markdown(EMPTY_ANSWER, elem_id="answer", container=True)
            gr.HTML('<p class="section-label">Sources</p>')
            sources = gr.HTML(EMPTY_SOURCES)
    gr.HTML(FOOTER)
    go.click(ask, [question, docs], [answer, sources], concurrency_limit=2)
    question.submit(ask, [question, docs], [answer, sources], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(css=CSS, theme=THEME)
