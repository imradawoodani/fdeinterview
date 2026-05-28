# DocRouter — Plant Documentation Assistant (prototype)

A floor supervisor asks a question in plain English. DocRouter **intelligently
routes** it to the right body of documentation — **Safety Procedures**,
**Maintenance Manuals**, or **Quality Control Standards** — retrieves the most
relevant passages, and returns a **grounded answer with citations**. Questions
that don't belong to any category are **rejected** instead of force-answered.

Data is pulled **live from authoritative public sources** (no scraping of
paywalled content):

| Category | Source |
| --- | --- |
| **Safety Procedures** | OSHA **29 CFR 1910** standards via the [eCFR API](https://www.ecfr.gov) (lockout/tagout, machine guarding, PPE, confined spaces, HazCom, noise, forklifts, electrical, ladders, fire extinguishers) |
| **Maintenance Manuals** | Engineering & equipment reference (pumps, motors, bearings, lubrication, belts, gears, preventive/predictive maintenance, vibration) via the Wikipedia API |
| **Quality Control Standards** | SPC, control charts, Cp/Cpk, acceptance sampling, Six Sigma (Wikipedia) + **FDA 21 CFR 820** Quality System Regulation (eCFR) |

## How it works

```
question
   │
   ▼
[ Scope gate ]  reject off-topic questions (trivia, small talk, etc.)
   │             • embedding mode: top cosine similarity < threshold → reject
   │             • lexical mode:   query-term coverage of top chunk too low → reject
   │             • LLM router can also return "out_of_scope"
   ▼
[ Router ]  LLM classifier (via Portkey) + retrieval-overlap evidence
   │         └─ falls back to pure-lexical routing when no API key is set
   ▼
[ Retriever ]  top-k chunks *within the chosen category*
   │             • embedding mode: FAISS cosine over Portkey embeddings
   │             • lexical mode:   hand-rolled BM25
   ▼
[ Answerer ]  LLM synthesizes a cited answer from retrieved passages
              └─ falls back to showing the top passage when no API key is set
```

Design choices for a robust prototype:
- **Runs with zero keys.** Routing, retrieval, and out-of-scope rejection are
  fully functional offline using BM25. Adding a Portkey key upgrades routing to
  an LLM classifier, switches retrieval to **FAISS embeddings**, and enables
  synthesized multi-source answers.
- **Refuses unrelated questions.** Two independent gates (a relevance threshold
  on the corpus + the router's own judgment) reject anything that isn't about
  safety, maintenance, or quality — so the system doesn't fabricate answers.
- **Grounded + cited.** The answerer is constrained to the retrieved passages
  and cites them as `[1]`, `[2]`, … with links back to the source.
- **Graceful degradation.** If embeddings are configured but a call fails at
  query time, retrieval transparently falls back to BM25.

### Retrieval: BM25 vs. embeddings

| | Lexical (default, no key) | Embeddings (with Portkey key) |
| --- | --- | --- |
| Index | hand-rolled BM25 | FAISS `IndexFlatIP` (cosine) over Portkey embeddings |
| Routing signal | BM25 per-category scores | per-category cosine similarity |
| Out-of-scope gate | top-chunk term coverage | top cosine similarity threshold (`OOS_COSINE_MIN`, 0.30) |
| Strengths | zero-dependency, deterministic, strong on keyword-heavy regulatory text | semantic recall + reliable off-topic detection |

## Setup

The repo ships with a `venv/` that already has the dependencies. Otherwise:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Build the corpus (fetches from the internet)

```bash
./venv/bin/python -m docrouter.ingest          # ~30–60s; caches raw responses
./venv/bin/python -m docrouter.ingest --fresh  # bypass cache and re-fetch
```

### 2. (Optional) Connect a model via Portkey

Copy `.env.example` to `.env` and fill in your Portkey details:

```bash
cp .env.example .env
```

- `PORTKEY_API_KEY` — your Portkey API key (required for LLM/embedding mode)
- `PORTKEY_BASE_URL` — gateway base URL, ending in `/v1`
- `LLM_MODEL` — chat model, e.g. `gpt-4o-mini`, `claude-3-5-sonnet-latest`
- `EMBED_MODEL` — embeddings model, e.g. `text-embedding-3-small`
- *(optional)* `PORTKEY_VIRTUAL_KEY`, or `PORTKEY_PROVIDER` +
  `PORTKEY_PROVIDER_API_KEY` — only if your key isn't already bound to a provider

Then build the embedding/FAISS index (requires the key above):

```bash
./venv/bin/python -m docrouter.ingest --embed      # embed + build FAISS index
./venv/bin/python -m docrouter.ingest --no-embed   # corpus only, skip embeddings
```

If `EMBED_MODEL` isn't set or embedding fails, the app falls back to BM25 and
LLM routing/answering still work.

### 3. Ask questions

CLI:

```bash
./venv/bin/python cli.py "What's the procedure to lock out a machine before servicing?"
./venv/bin/python cli.py            # interactive
```

Web UI:

```bash
./venv/bin/streamlit run streamlit_app.py
```

## Project layout

```
docrouter/
  config.py      # env loading + category definitions + thresholds
  sources.py     # curated source URLs per category
  ingest.py      # fetch → clean → chunk → data/corpus.json (+ optional embeddings)
  textutil.py    # HTML/XML→text, chunking, tokenization (stdlib)
  retriever.py   # BM25 index + per-category scoring + lexical scope check
  embeddings.py  # Portkey embeddings → FAISS cosine retriever + scope check
  llm.py         # Portkey gateway client (chat + embeddings, httpx)
  router.py      # intelligent routing (LLM + lexical fallback, can abstain)
  rag.py         # orchestration: scope gate → route → retrieve → answer
cli.py
streamlit_app.py
```

## Notes / next steps

- Make routing return multiple categories (and merge retrieval) when a question
  legitimately spans, e.g., maintenance + safety (lockout during servicing).
- Add an evaluation set of supervisor questions with expected category + source
  to measure routing accuracy, answer faithfulness, and the out-of-scope
  precision/recall (tune `OOS_COSINE_MIN`).
- Persist a richer FAISS index (e.g., IVF/HNSW) once the corpus grows beyond a
  few thousand chunks; `IndexFlatIP` is exact and fine at this size.
