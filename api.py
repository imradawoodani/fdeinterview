"""FastAPI service wrapping the DocRouter pipeline.

Run:  ./venv/bin/uvicorn api:app --reload --port 8000
Then: curl -s localhost:8000/ask -H 'content-type: application/json' \
        -d '{"question": "how often must fire extinguishers be inspected?"}' | jq
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from docrouter.config import CATEGORIES, Config
from docrouter.rag import DocRouterPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = DocRouterPipeline(Config.from_env())
    yield
    _state.clear()


app = FastAPI(title="DocRouter", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int | None = None


class Source(BaseModel):
    n: int
    title: str
    url: str
    category: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    in_scope: bool
    category: str
    category_label: str | None
    routing_method: str
    confidence: float
    reasoning: str
    answer: str
    answered_by: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    pipe: DocRouterPipeline = _state["pipeline"]
    return {
        "status": "ok",
        "chunks": len(pipe.corpus["records"]),
        "retrieval_mode": pipe.retrieval_mode,
        "llm": pipe.llm.available,
        "categories": pipe.corpus.get("categories", {}),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    pipe: DocRouterPipeline = _state["pipeline"]
    ans = pipe.ask(req.question, k=req.k)
    return AskResponse(
        question=ans.question,
        in_scope=ans.in_scope,
        category=ans.route.category,
        category_label=CATEGORIES.get(ans.route.category, {}).get("label"),
        routing_method=ans.route.method,
        confidence=ans.route.confidence,
        reasoning=ans.route.reasoning,
        answer=ans.answer,
        answered_by=ans.answered_by,
        sources=[Source(**s) for s in ans.sources],
    )
