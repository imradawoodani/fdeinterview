"""Configuration + tiny .env loader (no external deps)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_PATH = DATA_DIR / "corpus.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_PATH = DATA_DIR / "index.faiss"

# Human-readable definitions of each documentation category. These are used both
# in the UI and in the LLM router prompt, so keep them crisp and distinct.
CATEGORIES = {
    "safety": {
        "label": "Safety Procedures",
        "description": (
            "Worker safety, hazard control, and OSHA compliance: lockout/tagout, "
            "machine guarding, personal protective equipment (PPE), confined spaces, "
            "hazard communication, noise exposure, fire extinguishers, forklifts, "
            "electrical safety, and ladders/walking surfaces."
        ),
    },
    "maintenance": {
        "label": "Maintenance Manuals",
        "description": (
            "Keeping equipment running: how to service, inspect, lubricate, align, "
            "and troubleshoot pumps, electric motors, bearings, belts, and gearboxes; "
            "preventive vs. predictive maintenance, condition monitoring, and vibration analysis."
        ),
    },
    "quality": {
        "label": "Quality Control Standards",
        "description": (
            "Product quality and process control: statistical process control (SPC), "
            "control charts, process capability, acceptance sampling, Six Sigma, "
            "and FDA Quality System Regulation requirements (design controls, "
            "production controls, CAPA, statistical techniques)."
        ),
    },
}


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, no quoting magic, comments with #."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Don't clobber values already set in the real environment.
        os.environ.setdefault(key, value)


@dataclass
class Config:
    portkey_api_key: str = ""
    portkey_virtual_key: str = ""
    portkey_provider: str = ""
    portkey_provider_api_key: str = ""
    portkey_base_url: str = "https://api.portkey.ai/v1"
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    retrieval_k: int = 5
    chunk_target_words: int = 180
    chunk_overlap_words: int = 40
    ecfr_date: str = "2025-01-01"
    # Out-of-scope gating: refuse questions the corpus can't support.
    oos_cosine_min: float = 0.30   # min top cosine similarity (embedding mode)
    oos_min_terms: int = 3         # lexical mode: only gate queries this "wordy"
    oos_min_coverage: int = 2      # lexical mode: distinct query terms in top chunk

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            portkey_api_key=os.environ.get("PORTKEY_API_KEY", ""),
            portkey_virtual_key=os.environ.get("PORTKEY_VIRTUAL_KEY", ""),
            portkey_provider=os.environ.get("PORTKEY_PROVIDER", ""),
            portkey_provider_api_key=os.environ.get("PORTKEY_PROVIDER_API_KEY", ""),
            portkey_base_url=os.environ.get("PORTKEY_BASE_URL", "https://api.portkey.ai/v1"),
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            embed_model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        )

    @property
    def llm_available(self) -> bool:
        """True if we have enough config to attempt a call through Portkey.

        A Portkey API key alone is sufficient when the provider/credentials are
        attached server-side (config or default virtual key). A virtual key or
        provider key can also be supplied explicitly.
        """
        return bool(self.portkey_api_key)

    @property
    def embed_enabled(self) -> bool:
        """Whether we should attempt embedding-based retrieval."""
        return bool(self.portkey_api_key and self.embed_model)
