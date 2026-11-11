"""Evaluation harness.

Retrieval (no model calls):
  doc_hit@k      an expected document appears in the top-k results
  keyword_hit@k  an expected key term appears in the top-k text
  MRR            how high the first correct document ranks

Answers (calls the answer model):
  answer_rate         in-scope questions answered instead of refused
  grounded_rate       answers that cite real sources and invent none
  correct_doc_cited   answers citing at least one expected document
  refusal_rate        out-of-scope questions correctly refused
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import ROOT, Settings

EVAL_DIR = ROOT / "eval"


@dataclass
class Case:
    id: str
    question: str
    expected_docs: list[str]
    keywords: list[str]
    out_of_scope: bool = False


def load_cases(path: Path = EVAL_DIR / "questions.jsonl") -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cases.append(Case(d["id"], d["question"], d["expected_docs"], d["keywords"], d.get("out_of_scope", False)))
    return cases


def retrieval_metrics(case: Case, hits) -> dict:
    docs = [h.chunk.doc_id for h in hits]
    text = " ".join(h.chunk.text for h in hits).lower()
    rank = next((i + 1 for i, d in enumerate(docs) if d in case.expected_docs), None)
    return {
        "doc_hit": rank is not None,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "keyword_hit": any(k.lower() in text for k in case.keywords) if case.keywords else None,
        "retrieved_docs": docs,
    }


def _pct(xs: list[bool]) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"


def run_eval(settings: Settings, use_llm: bool = True, limit: int | None = None, app=None,
             cases: list[Case] | None = None) -> dict:
    from .pipeline import AskRBI

    cases = cases if cases is not None else load_cases()
    cases = cases[:limit] if limit else cases
    if app is None:
        if use_llm:
            app = AskRBI(settings)
        else:
            class _NoLLM:
                model = "none"

                def complete(self, system, user):
                    raise RuntimeError("LLM disabled")

            app = AskRBI(settings, llm=_NoLLM())

    rows = []
    for case in cases:
        hits = app.retriever.search(case.question, top_k=settings.top_k)
        row = {"id": case.id, "question": case.question, "out_of_scope": case.out_of_scope}
        if not case.out_of_scope:
            row.update(retrieval_metrics(case, hits))
        if use_llm:
            res = app.ask(case.question)
            a = res.answer
            row.update({
                "refused": a.refused,
                "grounded": a.grounded,
                "cited_docs": sorted({c.doc_id for c in a.citations}),
                "correct_doc_cited": bool(set(case.expected_docs) & {c.doc_id for c in a.citations}),
                "seconds": round(res.seconds, 2),
                "answer": a.text,
            })
        rows.append(row)
        print(f"  {case.id} {'OOS ' if case.out_of_scope else ''}" +
              (f"doc_hit={row.get('doc_hit')} " if not case.out_of_scope else "") +
              (f"refused={row.get('refused')}" if use_llm else ""))

    ins = [r for r in rows if not r["out_of_scope"]]
    oos = [r for r in rows if r["out_of_scope"]]
    summary = {
        "k": settings.top_k,
        "n_in_scope": len(ins),
        "n_out_of_scope": len(oos),
        "doc_hit@k": _pct([r["doc_hit"] for r in ins]),
        "keyword_hit@k": _pct([r["keyword_hit"] for r in ins if r["keyword_hit"] is not None]),
        "MRR": round(statistics.mean(r["reciprocal_rank"] for r in ins), 3) if ins else None,
    }
    if use_llm:
        answered = [r for r in ins if not r["refused"]]
        summary.update({
            "answer_rate": _pct([not r["refused"] for r in ins]),
            "grounded_rate": _pct([r["grounded"] for r in rows]),
            "correct_doc_cited": _pct([r["correct_doc_cited"] for r in answered]),
            "refusal_rate_out_of_scope": _pct([r["refused"] for r in oos]),
            "median_seconds": round(statistics.median(r["seconds"] for r in rows), 2) if rows else None,
            "model": app.llm.model,
        })
    summary["embedder"] = app.index.embedder_name
    summary["run_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    (EVAL_DIR / "results.md").write_text(_markdown(summary, rows))
    print(json.dumps(summary, indent=2))
    return summary


def _markdown(summary: dict, rows: list[dict]) -> str:
    lines = ["# Evaluation results", "", f"Run: {summary['run_at']} · embedder: `{summary['embedder']}`"
             + (f" · model: `{summary['model']}`" if "model" in summary else ""), "",
             "| Metric | Value |", "|---|---|"]
    for key in ["doc_hit@k", "keyword_hit@k", "MRR", "answer_rate", "grounded_rate", "correct_doc_cited",
                "refusal_rate_out_of_scope", "median_seconds"]:
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    lines += ["", f"k = {summary['k']}; {summary['n_in_scope']} in-scope and {summary['n_out_of_scope']} out-of-scope questions.",
              "", "## Misses", ""]
    misses = [r for r in rows if (not r["out_of_scope"] and not r.get("doc_hit"))
              or (r["out_of_scope"] and r.get("refused") is False)]
    lines += [f"- **{r['id']}** {r['question']}" + (f" (retrieved: {', '.join(r.get('retrieved_docs', [])[:3])})" if not r["out_of_scope"] else " (answered, should refuse)") for r in misses] or ["None."]
    return "\n".join(lines) + "\n"
