"""Atomic file writes, JSONL logs, hashing and input reading.

Every artefact is written to a ``.tmp`` sibling, fsynced, and renamed into
place, then the directory is fsynced, so a SIGKILL at any point leaves either
the previous file or the complete new one — never a partial.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def atomic_path(final: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to ``final``; on clean exit fsync it and
    rename it over ``final``. On error the temporary file is removed."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        _fsync_dir(final.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_bytes_atomic(path: str | Path, data: bytes) -> None:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    write_bytes_atomic(path, text.encode("utf-8"))


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record and fsync. A torn trailing line (kill mid-append) is
    skipped by :func:`read_jsonl`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_key(item_id: str) -> str:
    """Filesystem-safe, collision-resistant name for a per-id file."""
    return hashlib.sha1(item_id.encode("utf-8")).hexdigest()


def normalise_question(question: str) -> str:
    """Canonical form used only for the warm-up/evaluation disjointness check."""
    return re.sub(r"\s+", " ", question).strip().lower()


# ----------------------------------------------------------------------
# Input: JSONL with {id, question}; every other field is ignored.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    id: str
    question: str


def read_items(path: str | Path) -> list[Item]:
    """Read ``{id, question}`` records in file order. Other fields are
    dropped here and never reach any output. Ids must be unique, non-empty
    and free of ``/`` (they become HDF5 group names)."""
    items: list[Item] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from None
            if not isinstance(rec, dict) or "id" not in rec or "question" not in rec:
                raise ValueError(f"{path}:{lineno}: record needs 'id' and 'question'")
            item_id = str(rec["id"])
            question = rec["question"]
            if not item_id or "/" in item_id or item_id in (".", ".."):
                raise ValueError(f"{path}:{lineno}: invalid id {item_id!r}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{lineno}: empty or non-string question")
            if item_id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {item_id!r}")
            seen.add(item_id)
            # Stripped because chat templates (Llama) trim message content;
            # the role mask locates the question verbatim in the prompt.
            items.append(Item(item_id, question.strip()))
    if not items:
        raise ValueError(f"{path}: no records")
    return items
