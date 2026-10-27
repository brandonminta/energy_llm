"""Interrupted + resumed runs produce outputs identical to an uninterrupted
run; failures are never marked complete and are retried up to max_attempts."""

import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from energy_llm.capture import compute_item
from energy_llm.cli import main as cli_main
from energy_llm.extract import plan_shards, run_extract, shard_path, shard_state
from energy_llm.merge import run_merge, status_report
from energy_llm.storage import item_key, read_items, read_jsonl

PARQUETS = ("answers.parquet", "features.parquet", "baselines.parquet")


def _run_all(cfg, bundle, ev, name, **kw):
    items = read_items(ev)
    reports = [run_extract(cfg, bundle, items, ev, name, k, **kw)
               for k in range(len(plan_shards(items, cfg.extraction.samples_per_shard)))]
    return items, reports


def _h5_dict(path):
    out = {}
    with h5py.File(path) as f:
        f.visititems(lambda k, v: out.__setitem__(k, v[()]) if isinstance(v, h5py.Dataset) else None)
    return out


def _assert_same_outputs(a, b):
    for name in PARQUETS:
        pd.testing.assert_frame_equal(pd.read_parquet(a / name), pd.read_parquet(b / name))
    for name in ("traces.h5", "hidden.h5", "bank.h5"):
        da, db = _h5_dict(a / name), _h5_dict(b / name)
        assert da.keys() == db.keys()
        for k in da:
            np.testing.assert_array_equal(da[k], db[k])


def test_resume_after_interruption_is_identical_to_full_run(calibrated):
    cfg, bundle, ev, _ = calibrated
    items, reports = _run_all(cfg, bundle, ev, "full")
    assert all(r["state"] == "done" for r in reports)
    run_merge(cfg, items, ev, "full")

    # interrupted run: shard 0 stops after one id, plus crash debris
    r = run_extract(cfg, bundle, items, ev, "resumed", 0, stop_after=1)
    assert r["state"] == "pending" and r["n_completed"] == 1
    sdir = shard_path(cfg.extraction_dir("resumed"), 0)
    (sdir / "items" / ".stale.h5.123.tmp").write_bytes(b"partial")
    orphan = sdir / "items" / f"{item_key(items[1].id)}.h5"     # written, not indexed
    shutil.copy(next((sdir / "items").glob("*.h5")), orphan)
    assert items[1].id not in shard_state(sdir, 0, [items[1].id], 3).completed
    with pytest.raises(RuntimeError, match="not finished"):
        run_merge(cfg, items, ev, "resumed")
    _, reports = _run_all(cfg, bundle, ev, "resumed")
    assert all(r["state"] == "done" for r in reports)
    run_merge(cfg, items, ev, "resumed")

    _assert_same_outputs(cfg.extraction_dir("full"), cfg.extraction_dir("resumed"))
    # completion index has no duplicates; the rerun skipped finished ids
    done = [r["id"] for r in read_jsonl(sdir / "completed.jsonl")]
    assert len(done) == len(set(done)) == 3


def test_outputs_indexed_by_id_without_input_extras(calibrated):
    cfg, bundle, ev, rec = calibrated
    items, _ = _run_all(cfg, bundle, ev, "x")
    m = run_merge(cfg, items, ev, "x")
    d = cfg.extraction_dir("x")
    ans = pd.read_parquet(d / "answers.parquet")
    assert list(ans["id"]) == [it.id for it in items]
    assert "gold" not in ans.to_json() and "category" not in ans.columns
    feats = pd.read_parquet(d / "features.parquet")
    L = bundle.dims.n_layers
    assert len(feats) == len(items) * L * 3 * 2
    assert set(feats["variant"]) == {"mean", "first"}
    np.testing.assert_allclose(sorted(feats["beta"].unique()),
                               [rec["beta_star"] * k for k in (0.5, 1.0, 2.0)])
    with h5py.File(d / "traces.h5") as f:
        assert set(f.keys()) == {it.id for it in items}
    assert m["counts"] == {"completed": 7, "exhausted": 0, "pending": 0}
    assert m["beta_star"] == rec["beta_star"] and m["model_dtype"] == ["float32"]
    assert json.loads((d / "manifest.json").read_text())["config_sha256"] == cfg.sha256()


def _flaky(fail_ids, times):
    calls = {}

    def compute(bundle, bank, trace, item_id, *a):
        calls[item_id] = calls.get(item_id, 0) + 1
        if item_id in fail_ids and calls[item_id] <= times:
            raise RuntimeError("CUDA out of memory (simulated)")
        return compute_item(bundle, bank, trace, item_id, *a)
    return compute


def test_failures_are_retried_and_not_marked_complete(calibrated):
    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    bad = items[0].id
    compute = _flaky({bad}, times=2)
    sdir = shard_path(cfg.extraction_dir("f"), 0)
    for attempt in (1, 2):
        r = run_extract(cfg, bundle, items, ev, "f", 0, compute=compute)
        assert r["state"] == "pending"
        st = shard_state(sdir, 0, [it.id for it in items[:3]], 3)
        assert bad not in st.completed and st.attempts[bad] == attempt
    r = run_extract(cfg, bundle, items, ev, "f", 0, compute=compute)
    assert r["state"] == "done" and r["n_completed"] == 3
    recs = read_jsonl(sdir / "failed.jsonl")
    assert [x["attempt"] for x in recs] == [1, 2] and "out of memory" in recs[0]["error"]


def test_exhausted_ids_end_the_shard_and_are_reported(calibrated):
    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    bad = items[1].id
    compute = _flaky({bad}, times=99)
    states = [run_extract(cfg, bundle, items, ev, "e", 0, compute=compute)["state"]
              for _ in range(3)]
    assert states == ["pending", "pending", "done"]
    for k in (1, 2):
        run_extract(cfg, bundle, items, ev, "e", k)
    m = run_merge(cfg, items, ev, "e")
    assert list(m["exhausted_ids"]) == [bad] and m["counts"]["completed"] == 6
    ans = pd.read_parquet(cfg.extraction_dir("e") / "answers.parquet")
    assert bad not in set(ans["id"])


def test_changed_config_refuses_to_resume(calibrated):
    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    run_extract(cfg, bundle, items, ev, "c", 0, stop_after=1)
    cfg.extraction.jaccard_top_k = 4
    with pytest.raises(RuntimeError, match="differs"):
        run_extract(cfg, bundle, items, ev, "c", 0)


def test_status_reports_missing_shards(calibrated, tmp_path, capsys):
    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    run_extract(cfg, bundle, items, ev, "s", 1)
    rep = status_report(cfg, items, "s", ev)
    assert rep["num_shards"] == 3 and rep["missing_shards"] == [0, 2]
    assert rep["throughput"]["suggested_samples_per_shard"] >= 1

    import yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()))
    args = ["--config", str(cfg_path), "--input", str(ev), "--name", "s"]
    assert cli_main(["status", *args, "--missing"]) == 0
    assert capsys.readouterr().out.strip() == "0,2"
    assert cli_main(["status", *args, "--num-shards"]) == 0
    assert capsys.readouterr().out.strip() == "3"


def test_sigusr1_stops_after_current_id_and_resume_completes(calibrated):
    import os
    import signal

    from energy_llm.signals import GracefulStop

    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    stopper = GracefulStop()

    def compute(*a):
        res = compute_item(*a)
        os.kill(os.getpid(), signal.SIGUSR1)   # SLURM's pre-walltime warning
        return res

    try:
        r = run_extract(cfg, bundle, items, ev, "sig", 0, stopper=stopper, compute=compute)
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    assert r["stopped_by_signal"] and r["signal"] == "SIGUSR1"
    assert r["state"] == "pending" and r["n_completed"] == 1   # the in-flight id was finished
    r = run_extract(cfg, bundle, items, ev, "sig", 0)
    assert r["state"] == "done" and r["n_done_this_run"] == 2


def test_cli_extract_skips_finished_shard_without_loading_model(calibrated, tmp_path, capsys):
    import yaml

    cfg, bundle, ev, _ = calibrated
    run_extract(cfg, bundle, read_items(ev), ev, "k", 2)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()))
    # the tiny model id does not exist on the hub: reaching the loader would raise
    code = cli_main(["extract", "--config", str(cfg_path), "--input", str(ev),
                     "--name", "k", "--shard-index", "2"])
    assert code == 0 and '"skipped_model_load": true' in capsys.readouterr().out


def test_empty_responses_are_completed_flagged_and_not_failures(calibrated):
    from energy_llm.capture import ResidualHooks, forward_item
    from energy_llm.prompting import build_prompt

    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    hooks = ResidualHooks(bundle.model)
    first = forward_item(bundle, hooks, build_prompt(
        bundle.tokenizer, cfg.prompt.system, items[0].question, None)).gen_ids[0]
    hooks.remove()
    stops = sorted(set(bundle.stop_token_ids) | {first})     # items[0] now stops at once
    bundle.stop_token_ids = stops
    bundle.generation_config.eos_token_id = stops

    _, reports = _run_all(cfg, bundle, ev, "empty")
    assert all(r["state"] == "done" for r in reports)
    for k in range(len(reports)):
        assert read_jsonl(shard_path(cfg.extraction_dir("empty"), k) / "failed.jsonl") == []
    m = run_merge(cfg, items, ev, "empty")
    assert m["exhausted_ids"] == {} and m["counts"]["completed"] == len(items)
    assert m["checks"]["n_empty_response"] >= 1
    d = cfg.extraction_dir("empty")
    ans = pd.read_parquet(d / "answers.parquet").set_index("id")
    assert ans.loc[items[0].id, "empty_response"] and ans.loc[items[0].id, "n_response_tokens"] == 0
    feats = pd.read_parquet(d / "features.parquet")
    f0 = feats[feats["id"] == items[0].id]
    assert f0["empty_response"].all()
    assert f0["energy_resp"].isna().all() and f0["js"].isna().all()
    assert f0["energy_prompt"].notna().all() and f0["margin_prompt"].notna().all()
    others = feats[~feats["empty_response"]]
    assert others.drop(columns=["empty_response"]).notna().all().all()

