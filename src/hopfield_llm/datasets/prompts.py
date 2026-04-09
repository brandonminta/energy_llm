"""Standalone JSON prompt file loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_SINGLE_KEYS = {"question", "answer"}
REQUIRED_PAIRED_KEYS = {"question", "answer_factual", "answer_hallucinated"}


def load_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and normalise prompt records from a JSON file.

    Each record is a dict with at least the keys for single mode
    (question, answer) or paired mode (question, answer_factual,
    answer_hallucinated).  Missing optional fields are filled with defaults.

    Returns:
        List of normalised prompt dicts with 'id', 'mode', 'category', 'label'.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Prompt JSON must be a list of records")

    records: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Record {i} is not a dict")
        keys = set(entry.keys())
        if REQUIRED_PAIRED_KEYS.issubset(keys):
            mode = "paired"
        elif REQUIRED_SINGLE_KEYS.issubset(keys):
            mode = "single"
        else:
            raise ValueError(
                f"Record {i} missing required keys. "
                f"Need {REQUIRED_SINGLE_KEYS} or {REQUIRED_PAIRED_KEYS}, got {keys}"
            )
        entry.setdefault("id", i)
        entry.setdefault("category", "unknown")
        entry.setdefault("label", None)
        entry["mode"] = mode
        records.append(entry)

    return records
