"""Artifact save / load utilities for torch and JSON payloads."""

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
    """Save an arbitrary payload with torch.save."""
    path = _ensure_parent(path)
    torch.save(payload, path)
    return path


def load_torch_artifact(path: str | Path) -> Any:
    """Load a torch.save artifact.

    weights_only=False is required because artifacts are dicts containing
    mixed Python types (str, int, dict) alongside tensors.  These files are
    written by this codebase only — do not load untrusted .pt files.
    """
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_json_artifact(payload: Any, path: str | Path) -> Path:
    """Serialise payload to indented JSON."""
    path = _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_json_artifact(path: str | Path) -> Any:
    """Load a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
