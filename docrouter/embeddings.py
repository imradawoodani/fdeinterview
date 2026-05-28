"""Embedding-based retrieval with FAISS.

We embed every chunk via Portkey at ingest time, L2-normalize, and store them in a
FAISS `IndexFlatIP` (inner product on normalized vectors == cosine similarity).
At query time we embed the question and search. Because the index is small, we
search the whole index and derive both scoped results and per-category signals
from one pass. Cosine similarity also gives us a clean out-of-scope threshold.
"""
from __future__ import annotations

import numpy as np

from .config import EMBEDDINGS_PATH, FAISS_PATH, Config
from .llm import LLMError, PortkeyClient
from .retriever import Hit


def _normalize(mat: np.ndarray) -> np.ndarray:
    import faiss

    mat = np.ascontiguousarray(mat, dtype="float32")
    faiss.normalize_L2(mat)
    return mat


def build_index(records: list[dict], client: PortkeyClient) -> np.ndarray:
    """Embed all chunks and persist embeddings + a FAISS index. Returns the matrix."""
    import faiss

    texts = [r["text"] for r in records]
    print(f"  embedding {len(texts)} chunks via {client.cfg.embed_model}…")
    mat = client.embed(texts)
    mat = _normalize(mat)

    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    np.save(EMBEDDINGS_PATH, mat)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  saved embeddings ({mat.shape}) and FAISS index.")
    return mat


def index_exists() -> bool:
    return EMBEDDINGS_PATH.exists() and FAISS_PATH.exists()


class EmbeddingRetriever:
    """FAISS cosine retriever. Requires a client to embed the query at search time."""

    def __init__(self, records: list[dict], client: PortkeyClient, cfg: Config):
        import faiss

        self.records = records
        self.client = client
        self.cfg = cfg
        self.index = faiss.read_index(str(FAISS_PATH))
        if self.index.ntotal != len(records):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) != corpus size ({len(records)}). "
                "Re-run ingestion with --embed."
            )
        self.categories = [r["category"] for r in records]
        self._cache: tuple[str, list[tuple[int, float]]] | None = None

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.client.embed([query])
        return _normalize(vec)

    def _search_all(self, query: str) -> list[tuple[int, float]]:
        # Cache the most recent query so route/scope/search share one embed call.
        if self._cache and self._cache[0] == query:
            return self._cache[1]
        vec = self._embed_query(query)
        sims, idxs = self.index.search(vec, self.index.ntotal)
        ranked = [(int(i), float(s)) for i, s in zip(idxs[0], sims[0]) if i != -1]
        self._cache = (query, ranked)
        return ranked

    def search(self, query: str, k: int = 5, category: str | None = None) -> list[Hit]:
        ranked = self._search_all(query)
        hits: list[Hit] = []
        for idx, sim in ranked:
            if category and self.categories[idx] != category:
                continue
            hits.append(Hit(self.records[idx], sim))
            if len(hits) >= k:
                break
        return hits

    def category_scores(self, query: str, top_n: int = 3) -> dict[str, float]:
        ranked = self._search_all(query)
        buckets: dict[str, list[float]] = {}
        for idx, sim in ranked:
            buckets.setdefault(self.categories[idx], []).append(sim)
        return {
            cat: float(np.mean(sorted(sims, reverse=True)[:top_n]))
            for cat, sims in buckets.items()
        }

    def is_in_scope(self, query: str) -> bool:
        ranked = self._search_all(query)
        if not ranked:
            return False
        return ranked[0][1] >= self.cfg.oos_cosine_min

    def top_similarity(self, query: str) -> float:
        ranked = self._search_all(query)
        return ranked[0][1] if ranked else 0.0
