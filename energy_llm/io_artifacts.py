"""Atomic artifact writes, manifests, and resume logic.

Every stage's per-sample artifact is its checkpoint unit: all writes go to a
``.tmp`` sibling, are fsynced, and are renamed into place, so a SIGKILL leaves
no corrupt partials. A sample is *complete* when its metadata JSON exists and
parses — the JSON is always written last, so it is the commit marker for the
sample's whole artifact group.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _fsync_replace(tmp: Path, final: Path) -> None:
    with open(tmp, "rb+") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    # fsync the directory so the rename itself survives a crash
    dir_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def save_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_npz_atomic(path: str | Path, compressed: bool = True, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    buf = io.BytesIO()
    if compressed:
        np.savez_compressed(buf, **arrays)
    else:
        np.savez(buf, **arrays)
    tmp.write_bytes(buf.getvalue())
    _fsync_replace(tmp, path)


def save_torch_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    _fsync_replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record to a JSONL log (failures.jsonl). Appends are not
    atomic, but a torn trailing line is ignored by readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    path = Path(path)
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn trailing line from a kill mid-append
    return records


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    """Order-independent hash of a set of sample ids (test-split lock)."""
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


class SampleManifest:
    """Resume logic over a per-sample artifact directory.

    A sample ``sid`` is complete iff ``{sid}{suffix}`` exists and parses as
    JSON. Stages write the JSON last, after every other file of the sample.
    """

    def __init__(self, directory: str | Path, suffix: str = ".json"):
        self.directory = Path(directory)
        self.suffix = suffix

    def is_complete(self, sample_id: str) -> bool:
        path = self.directory / f"{sample_id}{self.suffix}"
        if not path.exists():
            return False
        try:
            load_json(path)
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def complete_ids(self) -> set[str]:
        if not self.directory.exists():
            return set()
        out = set()
        for path in self.directory.glob(f"*{self.suffix}"):
            sid = path.name[: -len(self.suffix)]
            if self.is_complete(sid):
                out.add(sid)
        return out

    def pending(self, sample_ids: Iterable[str]) -> list[str]:
        done = self.complete_ids()
        return [sid for sid in sample_ids if sid not in done]
