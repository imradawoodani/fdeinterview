"""Lexical BM25 retriever over the chunked corpus (numpy, no external services).

Why BM25 instead of embeddings? It needs no API key, is deterministic, fast on a
small corpus, and is genuinely strong for keyword-heavy technical/regulatory text.
The embedding path can be added later behind the same interface.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .textutil import tokenize


@dataclass
class Hit:
    record: dict
    score: float


class BM25Retriever:
    K1 = 1.5
    B = 0.75

    def __init__(self, records: list[dict]):
        self.records = records
        self.docs_tokens: list[list[str]] = [tokenize(r["text"]) for r in records]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.N = len(records)

        # document frequency + per-doc term frequency
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[Counter] = []
        for tokens in self.docs_tokens:
            counts = Counter(tokens)
            self.tf.append(counts)
            for term in counts:
                self.df[term] += 1

        # precompute idf
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

        # index of record positions by category for scoped search
        self.by_category: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            self.by_category[r["category"]].append(i)

    def _score_doc(self, idx: int, query_terms: list[str]) -> float:
        if self.avg_len == 0:
            return 0.0
        tf = self.tf[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = freq + self.K1 * (1 - self.B + self.B * dl / self.avg_len)
            score += idf * (freq * (self.K1 + 1)) / denom
        return score

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[Hit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        candidates = self.by_category[category] if category else range(self.N)
        scored = [(i, self._score_doc(i, query_terms)) for i in candidates]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [Hit(self.records[i], s) for i, s in scored[:k]]

    def is_in_scope(self, query: str, min_terms: int = 3, min_coverage: int = 2) -> bool:
        """Lexical out-of-scope heuristic.

        BM25 alone can't tell topical relevance from incidental rare-word matches,
        so we use term coverage of the top chunk: a genuinely on-topic question
        shares several distinct terms with its best match, while an off-topic one
        usually shares just one. We only gate "wordy" queries to avoid penalizing
        short but valid domain queries like "Cpk" or "PPE for grinding".
        """
        terms = set(tokenize(query))
        hits = self.search(query, k=1)
        if not hits:
            return False
        coverage = len(terms & set(tokenize(hits[0].record["text"])))
        if len(terms) >= min_terms and coverage < min_coverage:
            return False
        return True

    def category_scores(self, query: str, top_n: int = 3) -> dict[str, float]:
        """Aggregate signal per category: mean of its top-N chunk scores.

        Used by the heuristic router as a strong, key-free routing signal.
        """
        query_terms = tokenize(query)
        result: dict[str, float] = {}
        for category, idxs in self.by_category.items():
            scores = sorted(
                (self._score_doc(i, query_terms) for i in idxs), reverse=True
            )
            top = [s for s in scores[:top_n] if s > 0]
            result[category] = (sum(top) / len(top)) if top else 0.0
        return result
