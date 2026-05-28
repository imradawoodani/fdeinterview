"""End-to-end pipeline: route -> retrieve -> answer (grounded, with citations)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CATEGORIES, Config
from .ingest import load_corpus
from .llm import LLMError, PortkeyClient
from .retriever import BM25Retriever, Hit
from .router import OUT_OF_SCOPE, RouteResult, route


@dataclass
class Answer:
    question: str
    route: RouteResult
    answer: str
    hits: list[Hit]
    sources: list[dict] = field(default_factory=list)
    answered_by: str = "extractive"  # "llm" | "extractive" | "refused"
    in_scope: bool = True


SYSTEM_PROMPT = (
    "You are a documentation assistant for floor supervisors at a manufacturing "
    "plant. Answer ONLY using the provided context passages. If the context does "
    "not contain the answer, say so plainly and suggest which document to consult. "
    "Be concise and practical. Cite sources inline using bracketed numbers like "
    "[1], [2] that correspond to the numbered context passages."
)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        r = hit.record
        blocks.append(f"[{i}] ({r['source_title']})\n{r['text']}")
    return "\n\n".join(blocks)


def _extractive_answer(question: str, hits: list[Hit], category: str) -> str:
    """Key-free fallback: present the most relevant passage(s) with citations."""
    if not hits:
        label = CATEGORIES[category]["label"]
        return (f"I couldn't find a relevant passage in the **{label}** "
                f"documentation for this question. Try rephrasing, or check "
                f"whether it belongs to a different category.")
    top = hits[0].record
    snippet = top["text"]
    if len(snippet) > 700:
        snippet = snippet[:700].rsplit(" ", 1)[0] + "…"
    return (
        f"Based on **{top['source_title']}** [1]:\n\n{snippet}\n\n"
        f"_(Add a Portkey API key to enable synthesized, multi-source answers. "
        f"Showing the top-matching passage and its citations.)_"
    )


def _llm_answer(llm: PortkeyClient, question: str, hits: list[Hit]) -> str:
    context = _format_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"},
    ]
    return llm.chat(messages, temperature=0.1, max_tokens=700).strip()


def _sources_from_hits(hits: list[Hit]) -> list[dict]:
    seen: dict[str, dict] = {}
    ordered: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        r = hit.record
        entry = {
            "n": i,
            "title": r["source_title"],
            "url": r["url"],
            "category": r["category"],
            "score": round(hit.score, 3),
            "snippet": r["text"][:300] + ("…" if len(r["text"]) > 300 else ""),
        }
        ordered.append(entry)
    return ordered


def _refusal_text() -> str:
    cats = "\n".join(f"- **{m['label']}** — {m['description']}"
                     for m in CATEGORIES.values())
    return (
        "I can only answer questions about this plant's documentation. "
        "That question doesn't appear to relate to any of the covered areas:\n\n"
        f"{cats}\n\nTry rephrasing it around safety, maintenance, or quality."
    )


class DocRouterPipeline:
    """Loads the corpus + builds the retriever once; reusable across questions."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()
        self.corpus = load_corpus()
        records = self.corpus["records"]
        self.llm = PortkeyClient(self.cfg)
        self.bm25 = BM25Retriever(records)

        # Build the dense (FAISS) retriever if an index exists and a backend is
        # available, then fuse with BM25 (hybrid) unless disabled.
        self.dense = None
        self.embedder = None  # the active embedding-capable retriever (dense or hybrid)
        self.retrieval_mode = "lexical (BM25)"
        if self.cfg.embed_enabled:
            try:
                from .embedders import make_embedder
                from .embeddings import EmbeddingRetriever, index_exists

                if index_exists():
                    embedder = make_embedder(self.cfg, self.llm)
                    self.dense = EmbeddingRetriever(records, embedder, self.cfg)
                    if self.cfg.hybrid:
                        from .hybrid import HybridRetriever

                        self.embedder = HybridRetriever(
                            records, self.bm25, self.dense, self.cfg)
                        self.retrieval_mode = f"hybrid BM25 + {embedder.name} (RRF)"
                    else:
                        self.embedder = self.dense
                        self.retrieval_mode = f"embeddings ({embedder.name})"
            except Exception as exc:  # malformed index, missing lib, etc.
                print(f"[WARN] embedding retriever unavailable: {exc}")

    @property
    def stats(self) -> dict:
        return self.corpus.get("stats", {})

    @property
    def active_retriever(self):
        return self.embedder or self.bm25

    def _retrieve(self, question: str, k: int, category: str):
        """Use the embedding retriever, falling back to BM25 on any runtime error."""
        if self.embedder is not None:
            try:
                return self.embedder.search(question, k=k, category=category)
            except Exception as exc:
                print(f"[WARN] embedding search failed ({exc}); using BM25.")
        return self.bm25.search(question, k=k, category=category)

    def _in_scope(self, question: str) -> bool:
        if self.embedder is not None:
            try:
                return self.embedder.is_in_scope(question)
            except Exception:
                pass
        return self.bm25.is_in_scope(
            question, self.cfg.oos_min_terms, self.cfg.oos_min_coverage
        )

    def ask(self, question: str, k: int | None = None) -> Answer:
        k = k or self.cfg.retrieval_k
        retriever = self.active_retriever

        # Two independent scope gates: a relevance check on the corpus, and the
        # router's own judgment. Either one can reject an off-topic question.
        in_scope = self._in_scope(question)
        route_result = route(question, retriever, self.llm)

        if not in_scope or route_result.category == OUT_OF_SCOPE:
            if route_result.category != OUT_OF_SCOPE:
                route_result.category = OUT_OF_SCOPE
                route_result.reasoning = (
                    route_result.reasoning
                    + " | Rejected: no sufficiently relevant documentation found."
                )
            return Answer(
                question=question, route=route_result, answer=_refusal_text(),
                hits=[], sources=[], answered_by="refused", in_scope=False,
            )

        hits = self._retrieve(question, k=k, category=route_result.category)

        answered_by = "extractive"
        if self.llm.available and hits:
            try:
                answer_text = _llm_answer(self.llm, question, hits)
                answered_by = "llm"
            except LLMError:
                answer_text = _extractive_answer(question, hits, route_result.category)
        else:
            answer_text = _extractive_answer(question, hits, route_result.category)

        return Answer(
            question=question, route=route_result, answer=answer_text, hits=hits,
            sources=_sources_from_hits(hits), answered_by=answered_by, in_scope=True,
        )
