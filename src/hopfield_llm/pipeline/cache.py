"""Cache manager for pipeline artifacts.

Artifacts are stored with parameter-encoded filenames so that recomputation
is skipped when matching results already exist on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.cache")


def _param_hash(*args: Any, length: int = 8) -> str:
    """Deterministic short hash from arbitrary parameters."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


class CacheManager:
    """Manage cached artifacts on disk.

    Naming convention:
        {model_id}__{bank_type}__{extra_params}__{hash}.{ext}

    Supports .pt (torch), .json, and .parquet files.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key building
    # ------------------------------------------------------------------

    def bank_key(self, model_id: str, bank: str, normalize: bool) -> str:
        h = _param_hash(model_id, bank, normalize)
        slug = model_id.replace("/", "_")
        return f"{slug}__{bank}__norm{int(normalize)}__{h}.pt"

    def hidden_key(
        self,
        model_id: str,
        dataset_id: str,
        pooling: str,
        layers: list[int] | None,
    ) -> str:
        h = _param_hash(model_id, dataset_id, pooling, layers)
        slug = model_id.replace("/", "_")
        return f"{slug}__{dataset_id}__{pooling}__{h}.pt"

    def scores_key(
        self,
        model_id: str,
        dataset_id: str,
        bank: str,
        beta: float,
        metric: str,
    ) -> str:
        h = _param_hash(model_id, dataset_id, bank, beta, metric)
        slug = model_id.replace("/", "_")
        return f"{slug}__{dataset_id}__{bank}__beta{beta}__{metric}__{h}.pt"

    def analysis_key(self, experiment_id: str) -> str:
        return f"{experiment_id}__analysis.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def path(self, key: str) -> Path:
        return self.base_dir / key

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def load(self, key: str) -> Any:
        p = self.path(key)
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {p}")
        if p.suffix == ".pt":
            return torch.load(p, map_location="cpu", weights_only=False)
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        if p.suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(p)
        raise ValueError(f"Unknown cache file type: {p.suffix}")

    def save(self, key: str, payload: Any) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".pt":
            torch.save(payload, p)
        elif p.suffix == ".json":
            p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        elif p.suffix == ".parquet":
            payload.to_parquet(p)
        else:
            raise ValueError(f"Unknown cache file type: {p.suffix}")
        log.info("Cached: %s", p)
        return p

    def invalidate(self, key: str) -> bool:
        p = self.path(key)
        if p.exists():
            p.unlink()
            log.info("Invalidated: %s", p)
            return True
        return False
