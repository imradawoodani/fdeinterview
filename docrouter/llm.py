"""Thin Portkey gateway client (OpenAI-compatible /chat/completions).

Portkey is a unified gateway: you authenticate with your Portkey API key and then
either reference a "virtual key" (recommended) or pass a provider + provider key.
We call it directly over HTTP so the only dependency is httpx.
Docs: https://portkey.ai/docs
"""
from __future__ import annotations

import json

import httpx
import numpy as np

from .config import Config


class LLMError(RuntimeError):
    pass


class PortkeyClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return self.cfg.llm_available

    @property
    def embed_available(self) -> bool:
        return self.cfg.embed_enabled

    def _base(self) -> str:
        base = self.cfg.portkey_base_url.rstrip("/")
        # Tolerate users pasting a full endpoint URL instead of the base.
        for suffix in ("/chat/completions", "/embeddings"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        return base

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-portkey-api-key": self.cfg.portkey_api_key,
            "Content-Type": "application/json",
        }
        if self.cfg.portkey_virtual_key:
            headers["x-portkey-virtual-key"] = self.cfg.portkey_virtual_key
        else:
            # Provider mode: tell Portkey which provider, and pass that provider's
            # raw key in the Authorization header.
            if self.cfg.portkey_provider:
                headers["x-portkey-provider"] = self.cfg.portkey_provider
            headers["Authorization"] = f"Bearer {self.cfg.portkey_provider_api_key}"
        return headers

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
        response_json: bool = False,
    ) -> str:
        if not self.available:
            raise LLMError("LLM not configured (set PORTKEY_API_KEY + a key/virtual key).")

        payload: dict = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        url = self._base() + "/chat/completions"
        try:
            resp = httpx.post(url, headers=self._headers(), json=payload, timeout=60)
        except httpx.HTTPError as exc:
            raise LLMError(f"Request to Portkey failed: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"Portkey error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Portkey response: {json.dumps(data)[:500]}") from exc

    def embed(self, texts: list[str], *, batch_size: int = 96) -> np.ndarray:
        """Embed a list of texts. Returns an (n, dim) float32 array (un-normalized)."""
        if not self.embed_available:
            raise LLMError("Embeddings not configured (set PORTKEY_API_KEY + EMBED_MODEL).")

        url = self._base() + "/embeddings"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": self.cfg.embed_model, "input": batch}
            try:
                resp = httpx.post(url, headers=self._headers(), json=payload, timeout=120)
            except httpx.HTTPError as exc:
                raise LLMError(f"Embedding request to Portkey failed: {exc}") from exc
            if resp.status_code >= 400:
                raise LLMError(f"Portkey embeddings error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            try:
                # OpenAI-compatible: data sorted by "index"
                items = sorted(data["data"], key=lambda d: d["index"])
                vectors.extend(item["embedding"] for item in items)
            except (KeyError, TypeError) as exc:
                raise LLMError(f"Unexpected embeddings response: {json.dumps(data)[:500]}") from exc
        return np.asarray(vectors, dtype="float32")
