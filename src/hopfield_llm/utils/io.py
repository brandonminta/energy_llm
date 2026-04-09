"""Save / load utilities for torch and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_torch_artifact(payload: Any, path: str | Path) -> Path:
    path = _ensure_parent(path)
    torch.save(payload, path)
    return path


def load_torch_artifact(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_json_artifact(payload: Any, path: str | Path) -> Path:
    path = _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_json_artifact(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
