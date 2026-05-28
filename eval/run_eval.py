"""Evaluate routing accuracy + out-of-scope detection, and tune OOS_COSINE_MIN.

Usage:
  ./venv/bin/python -m eval.run_eval            # full report
  ./venv/bin/python -m eval.run_eval --sweep    # also sweep the cosine threshold

The threshold sweep answers: "what cosine cutoff best separates in-scope from
out-of-scope questions for the CURRENT embedding model?" Re-run it whenever you
change the embedding model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docrouter.config import CATEGORIES, Config
from docrouter.rag import DocRouterPipeline
from docrouter.router import OUT_OF_SCOPE

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
LABELS = list(CATEGORIES.keys())


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(pipe: DocRouterPipeline, questions: list[dict]) -> None:
    in_scope_q = [q for q in questions if q["expected"] != OUT_OF_SCOPE]
    oos_q = [q for q in questions if q["expected"] == OUT_OF_SCOPE]

    routing_correct = 0
    routing_total = 0
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in LABELS + [OUT_OF_SCOPE]} for a in LABELS}
    # OOS binary classification: positive class = out_of_scope
    tp = fp = fn = tn = 0

    print(f"\nEvaluating {len(questions)} questions "
          f"({len(in_scope_q)} in-scope, {len(oos_q)} out-of-scope)")
    print(f"Mode: retrieval={pipe.retrieval_mode}; "
          f"router={'LLM' if pipe.llm.available else 'lexical/embedding'}\n")

    for q in questions:
        ans = pipe.ask(q["question"])
        predicted = OUT_OF_SCOPE if not ans.in_scope else ans.route.category
        expected = q["expected"]

        if expected == OUT_OF_SCOPE:
            if predicted == OUT_OF_SCOPE:
                tp += 1
            else:
                fn += 1
        else:
            if predicted == OUT_OF_SCOPE:
                fp += 1
            else:
                tn += 1
            routing_total += 1
            if predicted == expected:
                routing_correct += 1
            confusion[expected][predicted] += 1

        mark = "ok " if predicted == expected else "MISS"
        print(f"  [{mark}] exp={expected:<13} pred={predicted:<13} | {q['question'][:60]}")

    print("\n--- Routing accuracy (in-scope questions only) ---")
    acc = routing_correct / routing_total if routing_total else 0.0
    print(f"  {routing_correct}/{routing_total} = {acc:.1%}")
    print("  confusion [expected -> predicted]:")
    header = "    " + " ".join(f"{c[:5]:>6}" for c in LABELS + [OUT_OF_SCOPE])
    print(header)
    for a in LABELS:
        row = " ".join(f"{confusion[a][b]:>6}" for b in LABELS + [OUT_OF_SCOPE])
        print(f"    {a[:5]:>4}{row}")

    print("\n--- Out-of-scope detection (positive = out_of_scope) ---")
    p, r, f1 = _prf(tp, fp, fn)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  precision={p:.2f}  recall={r:.2f}  F1={f1:.2f}")
    if fp:
        print(f"  ! {fp} in-scope question(s) were wrongly rejected (raise recall cost).")
    if fn:
        print(f"  ! {fn} out-of-scope question(s) slipped through (dangerous).")


def sweep_threshold(pipe: DocRouterPipeline, questions: list[dict]) -> None:
    if pipe.dense is None:
        print("\n[sweep] No dense/embedding retriever available; skipping threshold sweep.")
        return

    print("\n--- OOS cosine-threshold sweep (semantic gate only) ---")
    sims = []  # (top_cosine, is_oos)
    for q in questions:
        s = pipe.dense.top_similarity(q["question"])
        sims.append((s, q["expected"] == OUT_OF_SCOPE))

    in_scope_sims = sorted(s for s, oos in sims if not oos)
    oos_sims = sorted(s for s, oos in sims if oos)
    print(f"  in-scope top-cosine: min={in_scope_sims[0]:.3f} "
          f"median={in_scope_sims[len(in_scope_sims)//2]:.3f} max={in_scope_sims[-1]:.3f}")
    print(f"  out-of-scope top-cosine: min={oos_sims[0]:.3f} "
          f"median={oos_sims[len(oos_sims)//2]:.3f} max={oos_sims[-1]:.3f}")

    best = None
    print(f"\n  {'thresh':>7} {'prec':>6} {'recall':>7} {'F1':>6} {'kept_in':>8}")
    t = 0.05
    while t <= 0.7001:
        # predict OOS when top cosine < t
        tp = sum(1 for s, oos in sims if oos and s < t)
        fn = sum(1 for s, oos in sims if oos and s >= t)
        fp = sum(1 for s, oos in sims if not oos and s < t)
        kept_in = sum(1 for s, oos in sims if not oos and s >= t)
        p, r, f1 = _prf(tp, fp, fn)
        flag = ""
        if best is None or f1 > best[1]:
            best = (t, f1)
            flag = "  <- best F1"
        print(f"  {t:>7.2f} {p:>6.2f} {r:>7.2f} {f1:>6.2f} {kept_in:>8}{flag}")
        t += 0.05

    print(f"\n  Recommended OOS_COSINE_MIN ≈ {best[0]:.2f} (max F1={best[1]:.2f}). "
          f"Current config = {pipe.cfg.oos_cosine_min:.2f}.")
    print("  Tip: bias slightly LOWER than max-F1 if missing an off-topic question "
          "is worse than rejecting a borderline real one (plant-safety context).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="sweep the OOS cosine threshold")
    args = ap.parse_args()

    pipe = DocRouterPipeline(Config.from_env())
    questions = load_questions()
    evaluate(pipe, questions)
    if args.sweep:
        sweep_threshold(pipe, questions)


if __name__ == "__main__":
    main()
