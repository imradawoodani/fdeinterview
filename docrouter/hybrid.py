"""Hybrid retrieval: fuse BM25 (lexical) and dense (embedding) results via RRF.

Why hybrid? Floor supervisors mix exact terminology (part numbers, procedure
codes like "1910.147", chemical names) with paraphrased intent ("what do I do if
someone is exposed to fumes"). BM25 nails the former; dense embeddings nail the
latter. Reciprocal Rank Fusion (RRF) combines their ranked lists without needing
the two score scales to be comparable.

Scope gating uses the dense cosine threshold, with a strong-lexical override so
exact-terminology queries that dense under-ranks are not wrongly rejected.
"""
from __future__ import annotations

from collections import defaultdict

from .config import Config
from .retriever import BM25Retriever, Hit
from .textutil import tokenize


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, records: list[dict], bm25: BM25Retriever, dense, cfg: Config):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.bm25 = bm25
        self.dense = dense
        self.cfg = cfg

    def _ranked_ids(self, retriever, query: str, category: str | None, n: int) -> list[int]:
        return [h.record["id"] for h in retriever.search(query, k=n, category=category)]

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[Hit]:
        depth = max(k * 6, 50)
        lists = [
            self._ranked_ids(self.bm25, query, category, depth),
            self._ranked_ids(self.dense, query, category, depth),
        ]
        fused: dict[int, float] = defaultdict(float)
        for ranked in lists:
            for rank, rid in enumerate(ranked):
                fused[rid] += 1.0 / (self.cfg.rrf_k + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.by_id[rid], fused[rid]) for rid in top]

    def category_scores(self, query: str, top_n: int = 3) -> dict[str, float]:
        b = _normalize(self.bm25.category_scores(query, top_n))
        d = _normalize(self.dense.category_scores(query, top_n))
        cats = set(b) | set(d)
        return {c: 0.5 * b.get(c, 0.0) + 0.5 * d.get(c, 0.0) for c in cats}

    def is_in_scope(self, query: str) -> bool:
        # Primary semantic gate.
        if self.dense.is_in_scope(query):
            return True
        # Strong-lexical override: keep exact-terminology queries dense misses
        # (part numbers, "1910.147", chemical names). Requires a high term overlap
        # so off-topic keyword coincidences don't slip through.
        terms = set(tokenize(query))
        hits = self.bm25.search(query, k=1)
        if not hits:
            return False
        coverage = len(terms & set(tokenize(hits[0].record["text"])))
        return coverage >= self.cfg.hybrid_override_coverage

    def top_similarity(self, query: str) -> float:
        return self.dense.top_similarity(query)
