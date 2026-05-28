"""Pluggable embedding backends.

- LocalEmbedder: open-source sentence-transformers model (default, no API key).
- PortkeyEmbedder: hosted embeddings via the Portkey gateway.

Both return an (n, dim) float32 matrix of UN-normalized vectors; the FAISS layer
L2-normalizes so inner product == cosine similarity.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .llm import PortkeyClient


class EmbedderError(RuntimeError):
    pass


class LocalEmbedder:
    """Sentence-Transformers embedder (e.g. all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbedderError(
                "sentence-transformers not installed. `pip install sentence-transformers` "
                "or set EMBED_BACKEND=portkey."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return f"local:{self.model_name}"

    def embed(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=len(texts) > 256, normalize_embeddings=False,
        )
        return np.asarray(vecs, dtype="float32")


class PortkeyEmbedder:
    """Hosted embeddings through Portkey (OpenAI-compatible /embeddings)."""

    def __init__(self, client: PortkeyClient):
        self.client = client

    @property
    def name(self) -> str:
        return f"portkey:{self.client.cfg.embed_model}"

    def embed(self, texts: list[str], *, batch_size: int = 96) -> np.ndarray:
        return self.client.embed(texts, batch_size=batch_size)


def make_embedder(cfg: Config, client: PortkeyClient | None = None):
    """Construct the configured embedder, or raise EmbedderError if unavailable."""
    if cfg.embed_backend == "local":
        return LocalEmbedder(cfg.local_embed_model)
    if cfg.embed_backend == "portkey":
        if not (client and client.embed_available):
            raise EmbedderError(
                "Portkey embeddings need PORTKEY_API_KEY + EMBED_MODEL."
            )
        return PortkeyEmbedder(client)
    raise EmbedderError(f"Unknown EMBED_BACKEND: {cfg.embed_backend!r}")
