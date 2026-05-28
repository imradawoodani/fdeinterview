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
   │             • semantic: top embedding cosine < OOS_COSINE_MIN → reject
   │             • strong-lexical override keeps exact-terminology queries
   │             • LLM router can also return "out_of_scope"
   ▼
[ Router ]  LLM classifier (via Portkey, few-shot) + retrieval-overlap evidence
   │         └─ falls back to lexical/embedding routing when no API key is set
   ▼
[ Retriever ]  top-k chunks *within the chosen category*
   │             • hybrid: BM25 (lexical) + dense (embeddings) fused via RRF
   ▼
[ Answerer ]  LLM synthesizes a cited answer from retrieved passages
              └─ falls back to showing the top passage when no API key is set
```

Design choices for a robust prototype:
- **Open embeddings by default.** Retrieval uses a local
  `sentence-transformers` model (`all-MiniLM-L6-v2`) — **no API key required** —
  stored in a FAISS cosine index. Switch to hosted Portkey embeddings with
  `EMBED_BACKEND=portkey`.
- **Hybrid retrieval.** BM25 and dense results are fused with Reciprocal Rank
  Fusion, so exact terminology (part numbers, `1910.147`, chemical names) and
  paraphrased intent both retrieve well.
- **Refuses unrelated questions.** Two independent gates — a semantic relevance
  threshold + the LLM router's own judgment — reject anything that isn't about
  safety, maintenance, or quality, so the system doesn't fabricate answers.
- **Empirically tuned rejection.** `OOS_COSINE_MIN` is set from a sweep over a
  labeled eval set (`eval/`), not guessed. It is model-specific — re-tune it
  whenever you change the embedding model.
- **Runs with zero LLM keys.** Routing, hybrid retrieval, and rejection work
  offline; a Portkey key adds the LLM classifier + synthesized cited answers.
- **Grounded + cited.** The answerer is constrained to retrieved passages and
  cites them as `[1]`, `[2]`, … with links back to the source.

### Retrieval modes

| | Default (no key) | + Portkey key |
| --- | --- | --- |
| Embeddings | local `all-MiniLM-L6-v2` (sentence-transformers) | local, or hosted (`EMBED_BACKEND=portkey`) |
| Retrieval | hybrid BM25 + dense (RRF) | same |
| Routing | embedding/BM25 category scores | **few-shot LLM classifier** (+ evidence) |
| Out-of-scope gate | semantic cosine threshold + strong-lexical override | same, plus LLM can abstain |
| Answers | top passage shown extractively | **LLM-synthesized, cited** |

### Measured quality (eval/questions.jsonl, 48 questions)

With the live LLM router + local embeddings: **routing accuracy 36/36 (100%)**,
**out-of-scope detection F1 = 1.00** (precision 1.00, recall 1.00). Numbers are
on a small hand-written set — treat as a smoke test, not a guarantee.

## Setup

The repo ships with a `venv/` that already has the dependencies. Otherwise:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Build the corpus + embedding index (fetches from the internet)

```bash
./venv/bin/python -m docrouter.ingest --embed   # fetch + chunk + build FAISS index
./venv/bin/python -m docrouter.ingest --fresh    # bypass HTTP cache and re-fetch
./venv/bin/python -m docrouter.ingest --no-embed # corpus only, BM25-only retrieval
```

By default this uses the **local** `all-MiniLM-L6-v2` model (no key needed). The
first run downloads the model (~80 MB).

### 2. (Optional) Connect an LLM via Portkey

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

- `PORTKEY_API_KEY`, `PORTKEY_BASE_URL` (ending in `/v1`), `LLM_MODEL`
- With many gateways the provider is encoded in the model slug
  (e.g. `@aws-bedrock-use2/us.anthropic.claude-sonnet-4-5-...`), so the API key +
  base URL + model is enough — no virtual/provider keys needed.
- To use hosted embeddings instead of local: set `EMBED_BACKEND=portkey` and
  `EMBED_MODEL`, then re-run ingest with `--embed`.

### 3. Ask questions

```bash
# CLI
./venv/bin/python cli.py "What's the procedure to lock out a machine before servicing?"
./venv/bin/python cli.py            # interactive

# Web UI
./venv/bin/streamlit run streamlit_app.py

# REST API
./venv/bin/uvicorn api:app --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how often must fire extinguishers be inspected?"}'
```

### 4. Evaluate + tune the out-of-scope threshold

```bash
./venv/bin/python -m eval.run_eval           # routing accuracy + OOS precision/recall
./venv/bin/python -m eval.run_eval --sweep   # sweep OOS_COSINE_MIN, recommend a value
```

`OOS_COSINE_MIN` is **embedding-model-specific**: the sweep prints the cosine
distributions for in-scope vs. out-of-scope questions and the F1-optimal cutoff.
Re-run it (ideally with real supervisor questions) whenever you change the model.

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
cli.py
streamlit_app.py
eval/            # labeled questions + accuracy/OOS harness + threshold sweep
```

## Notes / next steps

- Grow `eval/questions.jsonl` with real supervisor questions and re-tune
  `OOS_COSINE_MIN`; the current value is fit on a small set.
- Make routing return multiple categories (and merge retrieval) when a question
  legitimately spans, e.g., maintenance + safety (lockout during servicing).
- Persist a richer FAISS index (IVF/HNSW) once the corpus grows beyond a few
  thousand chunks; `IndexFlatIP` is exact and fine at this size.
