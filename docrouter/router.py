"""Intelligent routing: pick the documentation category for a question.

Strategy:
  1. Compute a key-free lexical signal (BM25 category scores) as evidence.
  2. If an LLM is configured, ask it to choose a category given the definitions
     AND the lexical evidence, returning structured JSON with reasoning/confidence.
  3. Otherwise (or on LLM failure), fall back to the lexical signal.
This makes routing robust with zero keys, and "intelligent" when a model is present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import CATEGORIES
from .llm import LLMError, PortkeyClient
from .retriever import BM25Retriever

OUT_OF_SCOPE = "out_of_scope"
VALID = set(CATEGORIES.keys()) | {OUT_OF_SCOPE}


@dataclass
class RouteResult:
    category: str
    method: str  # "llm" | "lexical"
    confidence: float
    reasoning: str
    scores: dict[str, float] = field(default_factory=dict)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


def _lexical_route(retriever: BM25Retriever, question: str) -> RouteResult:
    scores = retriever.category_scores(question)
    norm = _normalize_scores(scores)
    if not norm or max(norm.values()) == 0:
        # No lexical signal at all -> default to safety (most safety-critical).
        return RouteResult("safety", "lexical", 0.0,
                           "No strong keyword match; defaulting to safety.", norm)
    best = max(norm, key=norm.get)
    return RouteResult(best, "lexical", round(norm[best], 3),
                       "Chosen by BM25 keyword overlap with each corpus.", norm)


def _build_prompt(question: str, lexical: dict[str, float]) -> list[dict]:
    cat_lines = "\n".join(
        f"- {key}: {meta['label']} — {meta['description']}"
        for key, meta in CATEGORIES.items()
    )
    evidence = ", ".join(f"{k}={v:.2f}" for k, v in lexical.items())
    system = (
        "You are a routing assistant for a manufacturing plant's documentation "
        "system. Route a floor supervisor's question to exactly ONE category, OR "
        "mark it 'out_of_scope' if it is NOT about plant safety, equipment "
        "maintenance, or quality control (e.g. trivia, food, small talk, general "
        "knowledge). Respond ONLY with JSON: "
        '{"category": "<safety|maintenance|quality|out_of_scope>", '
        '"confidence": <0..1>, "reasoning": "<one sentence>"}.'
    )
    user = (
        f"Categories:\n{cat_lines}\n\n"
        f"Retrieval-overlap evidence (normalized): {evidence}\n\n"
        f"Question: {question}\n\n"
        "Pick the single best category, or 'out_of_scope' if the question does not "
        "belong to any of them. Use the evidence as a hint, but rely on the meaning "
        "of the question. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def route(question: str, retriever: BM25Retriever, llm: PortkeyClient) -> RouteResult:
    lexical = _normalize_scores(retriever.category_scores(question))

    if not llm.available:
        return _lexical_route(retriever, question)

    try:
        raw = llm.chat(_build_prompt(question, lexical), temperature=0.0,
                       max_tokens=200, response_json=True)
        data = json.loads(raw)
        category = str(data.get("category", "")).strip().lower()
        if category not in VALID:
            raise ValueError(f"LLM returned invalid category: {category!r}")
        confidence = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", "")).strip()
        return RouteResult(category, "llm", round(confidence, 3), reasoning, lexical)
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        result = _lexical_route(retriever, question)
        result.reasoning = f"LLM routing unavailable ({exc}); used keyword fallback."
        return result
