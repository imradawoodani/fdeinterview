# DocRouter — Plant Documentation Assistant

A floor supervisor asks a question in plain English. **DocRouter** decides which
body of documentation it belongs to — **Safety Procedures**, **Maintenance
Manuals**, or **Quality Control Standards** — retrieves the most relevant
passages from *that* source, and returns a **grounded answer with citations**.
Questions that don't belong to any category are **politely refused** rather than
answered from the wrong manual.

```
"What's the procedure to lock out a machine before servicing?"
   → Routed to: Safety Procedures (confidence 0.95)
   → "Equipment must be stopped, isolated from all energy sources, and locked
      out before servicing… [1]. Only the authorized employee who applied a
      lock may remove it [2]."
   → Sources: 29 CFR 1910.147 (lockout/tagout) [1][2]
```

---

## Why this matters (the business case)

On a plant floor, the cost of a *wrong* answer is not a bad search result — it's
a safety incident, unplanned downtime, or a failed quality audit. The documents
that hold the answers are fragmented across three very different worlds (OSHA
regulations, equipment service manuals, QC/SPC standards), each with its own
vocabulary. Supervisors don't know — and shouldn't need to know — which binder
the answer lives in.

DocRouter is built around three principles that matter to an operator:

1. **Route to the right source.** A maintenance question answered from the safety
   manual is worse than a slow answer. Routing is the highest-stakes decision, so
   it gets a dedicated, testable component.
2. **Never make things up.** Answers are constrained to retrieved passages and
   carry citations back to the exact regulation/section. If the documentation
   doesn't cover it, the system says so.
3. **Know what it doesn't know.** Off-topic questions are rejected by an
   empirically-tuned relevance gate — not forced into a plausible-sounding but
   ungrounded answer.

It runs as a working prototype today and is structured so each layer (sources,
chunking, retrieval, routing, generation) can be upgraded independently.

---

## Data sources

All content is pulled **live from authoritative, free, no-auth public sources**
and cached locally — no paywalled or scraped proprietary manuals.

| Category | Source | Examples |
| --- | --- | --- |
| **Safety Procedures** | OSHA **29 CFR 1910** via the [eCFR API](https://www.ecfr.gov) (clean regulatory XML) | lockout/tagout (1910.147), machine guarding (1910.212), PPE (1910.132), confined spaces (1910.146), noise (1910.95), hazard communication (1910.1200), fire extinguishers (1910.157), forklifts (1910.178), electrical (1910.303), ladders (1910.23) |
| **Maintenance Manuals** | Engineering & equipment reference via the [Wikipedia API](https://en.wikipedia.org/w/api.php) | preventive/predictive/reliability-centered maintenance, centrifugal pumps, electric motors, rolling-element bearings, lubrication, vibration/condition monitoring, drive belts, gears |
| **Quality Control Standards** | Wikipedia (QC concepts) + **FDA 21 CFR 820** Quality System Regulation via eCFR | SPC, control charts, process capability (Cp/Cpk), acceptance sampling, Six Sigma; design controls (820.30), production controls (820.70), CAPA (820.100), statistical techniques (820.250) |

Why these: OSHA/FDA regulations are the real safety/quality "source of truth" and
are keyword-heavy (section numbers, defined terms); the Wikipedia engineering
corpus is a faithful stand-in for vendor maintenance manuals for a prototype. The
source list is a simple, declarative config (`docrouter/sources.py`) — swapping in
a customer's own PDFs/manuals is a localized change to ingestion only.

Current corpus: **~926 chunks** across 30 source documents (10 per category).

---

## How it works

```
                      question
                          │
                          ▼
              ┌───────────────────────┐
              │   1. SCOPE GATE        │  reject off-topic questions
              │   semantic cosine gate │  (trivia, small talk, cooking…)
              │   + strong-lexical     │
              │   override             │
              └───────────┬───────────┘
                          │ in scope
                          ▼
              ┌───────────────────────┐
              │   2. ROUTER            │  few-shot LLM classifier (Portkey)
              │   safety │ maintenance │  informed by BM25/embedding evidence;
              │   │ quality            │  can also abstain → out_of_scope
              └───────────┬───────────┘
                          │ category
                          ▼
              ┌───────────────────────┐
              │   3. RETRIEVER         │  HYBRID: BM25 (lexical) + dense
              │   top-k within the     │  (embeddings) fused via Reciprocal
              │   chosen category      │  Rank Fusion (RRF)
              └───────────┬───────────┘
                          │ passages
                          ▼
              ┌───────────────────────┐
              │   4. ANSWERER          │  LLM synthesizes an answer grounded
              │   grounded + cited     │  ONLY in retrieved passages, with
              │   [1][2]…              │  inline citations and source links
              └───────────────────────┘
```

**Two independent scope gates** protect against bad answers: (a) a semantic
relevance threshold on the corpus, and (b) the router's own LLM judgment. Either
can reject a question. The router is kept separate from retrieval so its accuracy
can be measured and improved on its own.

**Graceful degradation.** Every LLM-dependent step has a key-free fallback:
routing falls back to embedding/BM25 category scores, and answering falls back to
returning the top passage extractively. The system is fully functional without
any API key (using a local embedding model), and "lights up" with synthesized,
cited answers when a model is connected.

---

## Data chunking strategy

Implemented in `docrouter/textutil.py` and `docrouter/ingest.py`:

- **Clean first.** eCFR XML and Wikipedia HTML are stripped to plain text with a
  stdlib parser; Wikipedia boilerplate (*See also / References / External links*)
  and leaked LaTeX (`{\displaystyle …}`) are removed so chunks contain only real
  content.
- **Sentence-aware, fixed-size windows.** Text is split on sentence boundaries
  and greedily packed into **~180-word** chunks with a **~40-word overlap**
  (configurable). Overlap preserves context across boundaries so a procedure step
  isn't cut in half; the size is a balance between retrieval precision (smaller =
  more targeted) and answer context (larger = more complete).
- **Rich metadata per chunk.** Each chunk stores its `category`, `source_title`,
  public `url`, and position, so every retrieved passage can be cited and linked
  back to the exact regulation/article.

The corpus is a single `data/corpus.json` plus a FAISS index; chunks are tagged
with their category rather than stored in three physically separate indexes —
routing happens first, then retrieval is filtered to the chosen category, which
gives the same "no cross-contamination" guarantee with simpler bookkeeping.

---

## Retrieval strategy

**Hybrid lexical + dense, fused with Reciprocal Rank Fusion** (`docrouter/hybrid.py`):

- **Dense (semantic)** — embeddings in a FAISS `IndexFlatIP` (inner product on
  L2-normalized vectors = cosine similarity). Great for *paraphrased* intent:
  *"what do I do if someone is exposed to fumes"* → hazard-communication content,
  even with no shared keywords.
- **Lexical (BM25)** — hand-rolled BM25 over tokenized chunks. Great for *exact
  terminology* supervisors actually use: part numbers, section codes like
  `1910.147`, chemical names, equipment IDs.
- **Fusion (RRF)** — each retriever produces a ranked list; we combine them with
  `score = Σ 1/(k + rank)` (`k=60`). RRF needs no score calibration between the
  two very different scales and reliably surfaces results that *either* method
  ranks highly. This is the standard, robust hybrid approach.

**Pluggable embeddings** (`docrouter/embedders.py`):

| Backend | Model | Notes |
| --- | --- | --- |
| `local` | `all-MiniLM-L6-v2` (sentence-transformers, 384-d) | no API key, runs offline, ~80 MB download |
| `portkey` | e.g. `text-embedding-3-large` (3072-d) | hosted via Portkey gateway; higher quality |

The index records which embedder built it, so a query is never run against a
mismatched index. Switching models is just an env change + `ingest --embed`.

**Routing signal.** The router is shown normalized per-category retrieval scores
(a blend of BM25 + dense) as *evidence* alongside the question, so the LLM's
decision is grounded in what the corpus actually contains — a light-weight hybrid
of the classic "LLM classifier" and "retrieval-as-router" approaches.

---

## Out-of-scope rejection (and how the threshold is tuned)

A relevance threshold "makes or breaks" rejection behavior, so it is **tuned
empirically, not guessed**:

- The **primary gate** rejects a question when its top embedding cosine
  similarity is below `OOS_COSINE_MIN`.
- A **strong-lexical override** keeps exact-terminology queries that the dense
  model might under-rank (e.g. a bare part number), requiring a high term overlap
  so off-topic keyword coincidences (*"world cup"*) don't slip through.
- The **LLM router** is a second, independent gate that can return
  `out_of_scope`.

`eval/run_eval.py --sweep` prints the cosine distributions for in-scope vs.
out-of-scope questions and the F1-optimal cutoff. For the current model
(`text-embedding-3-large`) the separation is wide and clean:

```
in-scope  top-cosine: min 0.396   median 0.626
out-scope top-cosine: max 0.232   median 0.119
perfect-F1 plateau:   0.25 – 0.38   →  OOS_COSINE_MIN = 0.32
```

**The threshold is embedding-model-specific** — re-run the sweep whenever you
change the embedding model (the local MiniLM model tunes to ~0.31), and ideally
re-tune on a larger set of real supervisor questions before production.

---

## Interfaces / UI choices

Three entry points share one pipeline (`docrouter/rag.py`) so behavior is
identical everywhere:

- **Streamlit web app** (`streamlit_app.py`) — the demo-facing UI. Chosen for
  speed of iteration and because it renders the *reasoning*, not just the answer:
  the routed category is shown as a colored badge with confidence, per-category
  routing scores as metrics, the cited answer, and expandable source passages
  with links. Out-of-scope questions render a clear refusal. One-click example
  questions (including a deliberate off-topic one) make the behavior easy to
  demonstrate. A sidebar shows live status (retrieval mode, model, corpus stats).
- **REST API** (`api.py`, FastAPI) — `POST /ask` returns structured JSON
  (category, confidence, reasoning, answer, sources, `in_scope`) and `GET /health`
  reports config/corpus state. This is the integration surface for embedding
  DocRouter into an existing operator portal or chatbot.
- **CLI** (`cli.py`) — single-shot or interactive; handy for scripting and quick
  manual testing.

---

## Setup & run

The repo ships with a `venv/` that already has the dependencies. Otherwise:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Configure (optional but recommended)

```bash
cp .env.example .env     # then edit
```

- **Embeddings:** `EMBED_BACKEND=local` (default, no key) or `portkey` with
  `EMBED_MODEL` (e.g. `text-embedding-3-large`).
- **LLM (routing + answers):** `PORTKEY_API_KEY`, `PORTKEY_BASE_URL` (ending in
  `/v1`), `LLM_MODEL`. Many gateways encode the provider in the model slug
  (e.g. `@aws-bedrock-use2/us.anthropic.claude-sonnet-4-5-...`), so the key + base
  URL + model is enough — no virtual/provider keys needed.

### 2. Build the corpus + embedding index (fetches from the internet)

```bash
./venv/bin/python -m docrouter.ingest --embed    # fetch + chunk + build FAISS index
./venv/bin/python -m docrouter.ingest --fresh     # bypass HTTP cache and re-fetch
./venv/bin/python -m docrouter.ingest --no-embed  # corpus only, BM25-only retrieval
```

### 3. Ask questions

```bash
# Web UI
./venv/bin/streamlit run streamlit_app.py

# REST API
./venv/bin/uvicorn api:app --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how often must fire extinguishers be inspected?"}'

# CLI
./venv/bin/python cli.py "What PPE is required for grinding metal?"
```

### 4. Evaluate + tune

```bash
./venv/bin/python -m eval.run_eval           # routing accuracy + OOS precision/recall
./venv/bin/python -m eval.run_eval --sweep   # tune OOS_COSINE_MIN for your model
```

---

## Evaluation

`eval/questions.jsonl` holds 48 labeled questions (12 per in-scope category + 12
out-of-scope). The harness reports routing accuracy, a confusion matrix, and
out-of-scope precision/recall/F1.

**Current results** (live `claude-sonnet-4-5` router + `text-embedding-3-large`
embeddings, hybrid retrieval):

| Metric | Result |
| --- | --- |
| Routing accuracy (in-scope) | **36 / 36 = 100%** |
| Out-of-scope detection | **precision 1.00, recall 1.00, F1 1.00** |

These numbers are on a small hand-written set — treat them as a **smoke test**,
not a production guarantee. The first real task before deployment is to grow this
set with actual supervisor questions.

---

## Configuration reference (`.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBED_BACKEND` | `local` | `local` (sentence-transformers) or `portkey` |
| `LOCAL_EMBED_MODEL` | `all-MiniLM-L6-v2` | local embedding model |
| `EMBED_MODEL` | `text-embedding-3-small` | Portkey embedding model (when backend=portkey) |
| `PORTKEY_API_KEY` | — | enables LLM routing + synthesized answers |
| `PORTKEY_BASE_URL` | `https://api.portkey.ai/v1` | gateway base URL |
| `LLM_MODEL` | `gpt-4o-mini` | chat model slug |
| `HYBRID` | `1` | hybrid BM25+dense (RRF); `0` = dense-only |
| `OOS_COSINE_MIN` | `0.32` | out-of-scope cosine threshold (model-specific) |

---

## Project layout

```
docrouter/
  config.py      # env loading + category definitions + thresholds
  sources.py     # curated source URLs per category
  ingest.py      # fetch → clean → chunk → corpus.json (+ build embedding index)
  textutil.py    # HTML/XML→text, chunking, tokenization (stdlib)
  retriever.py   # BM25 index + per-category scoring + lexical scope check
  embedders.py   # pluggable embedders: local sentence-transformers | Portkey
  embeddings.py  # FAISS cosine retriever + semantic scope check
  hybrid.py      # BM25 + dense fusion (RRF) + hybrid scope gate
  llm.py         # Portkey gateway client (chat + embeddings, httpx)
  router.py      # few-shot LLM classifier (+ lexical fallback, can abstain)
  rag.py         # orchestration: scope gate → route → retrieve → answer
api.py           # FastAPI service (/health, /ask)
cli.py           # command-line interface
streamlit_app.py # web UI
eval/            # labeled questions + accuracy/OOS harness + threshold sweep
data/            # generated: corpus.json, FAISS index, raw HTTP cache
```

---

## Future improvements

- **Reranking (highest-value next step).** Add a cross-encoder / LLM reranker
  over the top ~20 hybrid candidates before answering. Bi-encoder retrieval is
  fast but coarse; a reranker (e.g. a cross-encoder MiniLM, Cohere Rerank, or a
  cheap LLM scoring pass) materially improves precision@k, which directly
  improves answer quality and citation accuracy. Slots in cleanly between
  retrieval and the answerer.
- **Multi-category routing.** Some questions legitimately span categories
  (a *lockout during servicing* is both safety and maintenance). Let the router
  return a ranked set, retrieve from each, and merge with RRF.
- **Confidence-aware behavior.** When routing confidence or top similarity is in
  a gray zone, ask a clarifying question or query all stores rather than commit.
- **Bigger, real eval set + CI.** Expand `eval/questions.jsonl` with real
  supervisor questions, add answer-faithfulness scoring (does every claim trace to
  a citation?), and gate changes on it. Re-tune `OOS_COSINE_MIN` on that set.
- **Production retrieval.** Move to a managed vector DB (Weaviate/OpenSearch/
  Pinecone) with native hybrid search; switch FAISS `IndexFlatIP` → IVF/HNSW once
  the corpus is large.
- **Ingestion for real manuals.** PDF/scanned-doc parsing (with OCR + table
  extraction), revision-date metadata, and incremental re-indexing so updated
  procedures propagate.
- **Trust & UX.** Highlight the exact sentences a citation came from, capture
  thumbs-up/down for an evaluation flywheel, and log routing decisions for audit.
- **Safety guardrails.** For high-risk topics, surface the full source text
  prominently and add a "verify against the official document" disclaimer.

---

## Limitations

- Maintenance content uses Wikipedia as a stand-in for vendor manuals; production
  use needs the customer's actual equipment documentation.
- The eval set is small and hand-written; metrics are indicative, not definitive.
- Answers are only as current as the last `ingest` run (eCFR content is dated via
  `ecfr_date` in config).
- `IndexFlatIP` is exact but linear — fine for thousands of chunks, not millions.
