"""Command-line interface for DocRouter.

Examples:
  python cli.py "How often must we inspect fire extinguishers?"
  python cli.py            # interactive loop
"""
from __future__ import annotations

import sys

from docrouter.config import CATEGORIES, Config
from docrouter.rag import DocRouterPipeline


def _print_answer(ans) -> None:
    label = CATEGORIES.get(ans.route.category, {}).get("label", "Out of scope")
    print("\n" + "=" * 72)
    if not ans.in_scope:
        print("OUT OF SCOPE — refused")
        print(f"  reason: {ans.route.reasoning}")
        print("-" * 72)
        print(ans.answer)
        print("=" * 72)
        return
    print(f"ROUTED TO: {label}  "
          f"(via {ans.route.method}, confidence {ans.route.confidence})")
    print(f"  reason: {ans.route.reasoning}")
    print("-" * 72)
    print(ans.answer)
    print("-" * 72)
    print(f"SOURCES (answered by {ans.answered_by}):")
    for s in ans.sources:
        print(f"  [{s['n']}] {s['title']}  (score {s['score']})")
        print(f"       {s['url']}")
    print("=" * 72)


def main() -> None:
    cfg = Config.from_env()
    pipe = DocRouterPipeline(cfg)
    router_mode = "LLM router" if pipe.llm.available else "lexical router"
    print(f"DocRouter ready — retrieval: {pipe.retrieval_mode}; {router_mode}; "
          f"{len(pipe.corpus['records'])} chunks loaded.")

    if len(sys.argv) > 1:
        _print_answer(pipe.ask(" ".join(sys.argv[1:])))
        return

    print("Type a question (or 'quit'):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        _print_answer(pipe.ask(q))


if __name__ == "__main__":
    main()
