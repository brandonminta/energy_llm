import json

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
