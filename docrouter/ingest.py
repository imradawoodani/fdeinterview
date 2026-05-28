"""Fetch documentation from online sources, clean, chunk, and persist a corpus.

Run:  python -m docrouter.ingest          (uses cache where present)
      python -m docrouter.ingest --fresh  (ignore cache, re-fetch everything)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import requests

from .config import CORPUS_PATH, DATA_DIR, RAW_DIR, CATEGORIES, Config
from . import sources as S
from .textutil import chunk_text, html_to_text, clean_text

USER_AGENT = "DocRouter-Prototype/0.1 (FDE interview demo; contact: demo@example.com)"


def _cache_path(key: str):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return RAW_DIR / f"{h}.txt"


def _fetch(url: str, *, fresh: bool, is_json: bool) -> str:
    cache = _cache_path(url)
    if cache.exists() and not fresh:
        return cache.read_text()
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    body = resp.text
    cache.write_text(body)
    return body


def _extract_wikipedia(raw_json: str) -> str:
    data = json.loads(raw_json)
    pages = data.get("query", {}).get("pages", {})
    parts = []
    for page in pages.values():
        extract = page.get("extract", "")
        if extract:
            parts.append(extract)
    text = "\n".join(parts)
    # Drop Wikipedia "See also / References / External links" tail sections.
    for marker in ("\nSee also", "\nReferences", "\nExternal links", "\nNotes", "\nFurther reading"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return clean_text(text)


def fetch_source(spec: dict, cfg: Config, *, fresh: bool) -> tuple[str, str, str]:
    """Return (full_text, public_url, display_title) for a single source spec."""
    if spec["type"] == "ecfr":
        url = S.ecfr_xml_url(spec["title"], spec["part"], spec["section"], cfg.ecfr_date)
        raw = _fetch(url, fresh=fresh, is_json=False)
        text = html_to_text(raw)
        if spec["title"] == 29:
            public = S.osha_public_url(spec["part"], spec["section"])
        else:
            public = S.ecfr_public_url(spec["title"], spec["section"])
        title = f"{spec['name']} ({spec['title']} CFR {spec['section']})"
        return text, public, title

    if spec["type"] == "wikipedia":
        url = S.wikipedia_extract_url(spec["page"])
        raw = _fetch(url, fresh=fresh, is_json=True)
        text = _extract_wikipedia(raw)
        public = S.wikipedia_public_url(spec["page"])
        return text, public, spec["name"]

    raise ValueError(f"Unknown source type: {spec['type']}")


def build_corpus(fresh: bool = False, embed: bool | None = None) -> dict:
    cfg = Config.from_env()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    stats: dict[str, dict] = {}
    chunk_id = 0

    for category, specs in S.SOURCES.items():
        cat_docs = 0
        cat_chunks = 0
        for spec in specs:
            try:
                text, url, title = fetch_source(spec, cfg, fresh=fresh)
            except Exception as exc:  # keep going; a flaky source shouldn't kill ingest
                print(f"  [WARN] failed {spec}: {exc}", file=sys.stderr)
                continue
            chunks = chunk_text(text, cfg.chunk_target_words, cfg.chunk_overlap_words)
            if not chunks:
                print(f"  [WARN] no text extracted for {title}", file=sys.stderr)
                continue
            cat_docs += 1
            for i, chunk in enumerate(chunks):
                records.append({
                    "id": chunk_id,
                    "category": category,
                    "source_title": title,
                    "url": url,
                    "section": f"chunk {i + 1}/{len(chunks)}",
                    "text": chunk,
                })
                chunk_id += 1
                cat_chunks += 1
            print(f"  [{category}] {title}: {len(chunks)} chunks")
            time.sleep(0.2)  # be polite to the public APIs
        stats[category] = {"documents": cat_docs, "chunks": cat_chunks}

    corpus = {
        "categories": {k: v["label"] for k, v in CATEGORIES.items()},
        "stats": stats,
        "records": records,
    }
    CORPUS_PATH.write_text(json.dumps(corpus, indent=2))

    # Embedding index (optional). Default: build it when a Portkey key is present.
    should_embed = cfg.embed_enabled if embed is None else embed
    if should_embed:
        from .embeddings import build_index
        from .llm import LLMError, PortkeyClient

        if not cfg.embed_enabled:
            print("  [WARN] --embed requested but PORTKEY_API_KEY/EMBED_MODEL not set; "
                  "skipping embeddings.", file=sys.stderr)
        else:
            try:
                build_index(records, PortkeyClient(cfg))
            except LLMError as exc:
                print(f"  [WARN] embedding failed ({exc}). Falling back to BM25 only.",
                      file=sys.stderr)
    return corpus


def load_corpus() -> dict:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No corpus at {CORPUS_PATH}. Run `python -m docrouter.ingest` first."
        )
    return json.loads(CORPUS_PATH.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documentation sources into a corpus.")
    ap.add_argument("--fresh", action="store_true", help="ignore cache and re-fetch")
    ap.add_argument("--embed", dest="embed", action="store_true", default=None,
                    help="force building the embedding/FAISS index")
    ap.add_argument("--no-embed", dest="embed", action="store_false",
                    help="skip embeddings even if a key is configured")
    args = ap.parse_args()
    print("Building corpus from online sources...")
    corpus = build_corpus(fresh=args.fresh, embed=args.embed)
    total = len(corpus["records"])
    print(f"\nDone. {total} chunks across {len(corpus['stats'])} categories.")
    for cat, st in corpus["stats"].items():
        print(f"  - {cat}: {st['documents']} docs, {st['chunks']} chunks")
    print(f"Saved to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
