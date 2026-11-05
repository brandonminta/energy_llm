import json

import numpy as np
import pytest

from energy_llm.storage import (
    append_jsonl, atomic_path, item_key, read_items, read_jsonl, write_json_atomic,
)


def test_atomic_write_leaves_no_temporary(tmp_path):
    write_json_atomic(tmp_path / "a.json", {"x": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_failed_atomic_write_keeps_previous_file(tmp_path):
    write_json_atomic(tmp_path / "a.json", {"x": 1})
    with pytest.raises(RuntimeError):
        with atomic_path(tmp_path / "a.json") as tmp:
            tmp.write_text("partial")
            raise RuntimeError("killed")
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert len(list(tmp_path.iterdir())) == 1


def test_read_jsonl_skips_torn_trailing_line(tmp_path):
    p = tmp_path / "log.jsonl"
    append_jsonl(p, {"id": "a"})
    with open(p, "a") as fh:
        fh.write('{"id": "b"')
    assert read_jsonl(p) == [{"id": "a"}]


def test_read_items_keeps_only_id_and_question(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"id": 7, "question": "  what is it ", "answer": "gold", "category": "x"}\n')
    (item,) = read_items(p)
    assert item.id == "7" and item.question == "what is it"
    assert set(vars(item)) == {"id", "question"}


@pytest.mark.parametrize("lines, msg", [
    (['{"id": "a", "question": "q"}', '{"id": "a", "question": "r"}'], "duplicate"),
    (['{"id": "a/b", "question": "q"}'], "invalid id"),
    (['{"id": "a"}'], "needs"),
    (['{"id": "a", "question": "  "}'], "empty"),
])
def test_read_items_rejects_bad_records(tmp_path, lines, msg):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=msg):
        read_items(p)


def test_item_key_is_filesystem_safe():
    k = item_key("weird id: ü?*")
    assert len(k) == 40 and k.isalnum()


def test_item_file_round_trip(tmp_path):
    from energy_llm.core.capture import REFERENCES, ItemResult
    from energy_llm.pipeline.extract import item_path, read_item, write_item

    rng = np.random.default_rng(0)
    L, T, nb, k = 3, 7, 4, 5
    traces = {n: rng.standard_normal((L, T, nb)).astype(np.float32)
              for n in ("energy", "norm_entropy", "js/question_last", "dkl/response_first")}
    traces["margin"] = rng.random((L, T)).astype(np.float32)
    traces["topk_ids"] = rng.integers(0, 40, (L, T, k)).astype(np.int32)
    hidden = {n: rng.standard_normal((L, 6)).astype(np.float16) for n in ("prompt_last", "response_last")}
    res = ItemResult(item_id="a/b", answer={"id": "a/b", "x": [1, 2]}, baselines={"id": "a/b"},
                     betas=[1.0, 10.0, 100.0, 37.5], topk_mass_k=3, topk_ids_k=k,
                     reference_positions={"question_last": 2, "prompt_last": 4, "response_first": None},
                     traces=traces, role_mask=np.arange(T, dtype=np.int8) % 4, hidden=hidden)
    write_item(tmp_path, res)
    back = read_item(item_path(tmp_path, "a/b"))
    assert back["id"] == "a/b" and back["answer"] == res.answer
    assert back["betas"].tolist() == res.betas
    assert (back["topk_mass_k"], back["topk_ids_k"]) == (3, k)
    assert back["reference_positions"] == res.reference_positions
    assert list(REFERENCES) == list(res.reference_positions)
    for n, arr in traces.items():
        want = arr if arr.dtype == np.int32 else arr.astype(np.float16)
        assert back["traces"][n].dtype == want.dtype
        np.testing.assert_array_equal(back["traces"][n], want)
    for n, arr in hidden.items():
        np.testing.assert_array_equal(back["hidden"][n], arr)
    np.testing.assert_array_equal(back["role_mask"], res.role_mask)
    import h5py
    with h5py.File(item_path(tmp_path, "a/b")) as f:
        d = f["traces/energy"]
        assert d.chunks == (1, T, nb) and d.compression == "gzip" and d.compression_opts == 4
        assert f["traces/topk_ids"].chunks == (1, T, k)
