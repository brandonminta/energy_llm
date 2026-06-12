"""Atomic writes, manifests, and resume primitives."""

import json

import numpy as np
import pytest

from energy_llm.io_artifacts import (
    SampleManifest,
    append_jsonl,
    load_json,
    read_jsonl,
    save_json_atomic,
    save_npz_atomic,
    sha256_ids,
)


class TestAtomicWrites:
    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "x.json"
        save_json_atomic(path, {"a": 1, "b": [1, 2]})
        assert load_json(path) == {"a": 1, "b": [1, 2]}
        assert not path.with_name(path.name + ".tmp").exists()

    def test_npz_roundtrip(self, tmp_path):
        path = tmp_path / "x.npz"
        save_npz_atomic(path, a=np.arange(5), b=np.eye(3))
        with np.load(path) as npz:
            np.testing.assert_array_equal(npz["a"], np.arange(5))
        assert not path.with_name(path.name + ".tmp").exists()

    def test_overwrite_is_atomic_replace(self, tmp_path):
        path = tmp_path / "x.json"
        save_json_atomic(path, {"v": 1})
        save_json_atomic(path, {"v": 2})
        assert load_json(path)["v"] == 2


class TestManifest:
    def test_complete_requires_parseable_json(self, tmp_path):
        m = SampleManifest(tmp_path)
        (tmp_path / "s1.json").write_text(json.dumps({"sample_id": "s1"}))
        (tmp_path / "s2.json").write_text('{"truncated by a kill')  # torn write
        assert m.is_complete("s1")
        assert not m.is_complete("s2")
        assert not m.is_complete("s3")
        assert m.complete_ids() == {"s1"}

    def test_pending_preserves_order(self, tmp_path):
        m = SampleManifest(tmp_path)
        (tmp_path / "b.json").write_text("{}")
        assert m.pending(["a", "b", "c"]) == ["a", "c"]

    def test_missing_directory_is_empty(self, tmp_path):
        m = SampleManifest(tmp_path / "nope")
        assert m.complete_ids() == set()


class TestJsonl:
    def test_append_and_read(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        append_jsonl(path, {"sample_id": "a", "error": "boom"})
        append_jsonl(path, {"sample_id": "b", "error": "bang"})
        records = read_jsonl(path)
        assert [r["sample_id"] for r in records] == ["a", "b"]

    def test_torn_trailing_line_ignored(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        append_jsonl(path, {"sample_id": "a"})
        with open(path, "a") as fh:
            fh.write('{"sample_id": "torn')
        assert [r["sample_id"] for r in read_jsonl(path)] == ["a"]


class TestHashing:
    def test_sha256_ids_order_independent(self):
        assert sha256_ids(["b", "a"]) == sha256_ids(["a", "b"])
        assert sha256_ids(["a"]) != sha256_ids(["a", "b"])
