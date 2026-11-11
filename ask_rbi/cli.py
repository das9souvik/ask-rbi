"""Command line: fetch, build, ask, eval.

    python -m ask_rbi.cli fetch            # download PDFs listed in data/manifest.toml
    python -m ask_rbi.cli build            # parse, chunk, embed, save to index/
    python -m ask_rbi.cli ask "question"   # one answer with sources
    python -m ask_rbi.cli eval             # run the evaluation set, write eval/results.md
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask-rbi")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="Download the RBI PDFs")
    f.add_argument("--force", action="store_true", help="Re-download even if present")
    f.add_argument("--only", nargs="*", help="Only these document ids")
    sub.add_parser("build", help="Build the search index")
    a = sub.add_parser("ask", help="Ask one question")
    a.add_argument("question")
    e = sub.add_parser("eval", help="Run the evaluation set")
    e.add_argument("--no-llm", action="store_true", help="Retrieval metrics only (no answer model calls)")
    e.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "fetch":
        from .fetch import FetchError, fetch_all
        from .manifest import load_manifest

        docs = load_manifest(settings.manifest_path)
        if args.only:
            docs = [d for d in docs if d.id in set(args.only)]
        failed = []
        for d in docs:
            try:
                fetch_all([d], settings.raw_dir, force=args.force)
            except FetchError as exc:
                failed.append(d.id)
                print(f"  FAIL  {exc}", file=sys.stderr)
        print(f"Done: {len(docs) - len(failed)} ok, {len(failed)} failed {failed or ''}")
        return 1 if failed else 0

    if args.cmd == "build":
        from .pipeline import build_index

        build_index(settings)
        return 0

    if args.cmd == "ask":
        from .pipeline import AskRBI

        res = AskRBI(settings).ask(args.question)
        print(res.answer.text, "\n")
        for c in res.answer.citations:
            print(f"[{c.tag}] {c.label} (page {c.page_start}) {c.url}")
        print(f"\n{res.seconds:.1f}s · model: {res.answer.model}")
        return 0

    if args.cmd == "eval":
        from .evaluate import run_eval

        run_eval(settings, use_llm=not args.no_llm, limit=args.limit)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
