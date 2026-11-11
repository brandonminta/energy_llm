"""Interrupted + resumed runs produce outputs identical to an uninterrupted
run; failures are never marked complete and are retried up to max_attempts."""

import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from energy_llm.core.capture import compute_item
from energy_llm.cli import main as cli_main
from energy_llm.pipeline.extract import plan_shards, run_extract, shard_path, shard_state
from energy_llm.pipeline.merge import run_merge, status_report
from energy_llm.storage import item_key, read_items, read_jsonl

PARQUETS = ("answers.parquet", "baselines.parquet")


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
    with h5py.File(d / "traces.h5") as f:
        assert set(f.keys()) == {it.id for it in items}
        assert f.attrs["beta_star"] == rec["beta_star"]
        assert f.attrs["betas"].tolist() == cfg.extraction.betas(rec["beta_star"])
        g = f[items[0].id]
        assert g["energy"].shape[-1] == 10 and g["energy"].dtype == np.float16
        assert g["topk_ids"].dtype == np.int32 and g["js/question_last"].dtype == np.float16
    with h5py.File(d / "hidden.h5") as f:
        assert set(f[items[0].id]) == {"question_last", "prompt_last", "response_first", "response_last"}
    assert m["counts"] == {"completed": 7, "exhausted": 0, "pending": 0}
    assert m["beta_star"] == rec["beta_star"] and m["model_dtype"] == ["float32"]
    g = m["gamma_diagnostic"]
    assert g["k"] == 5 and len(g["fraction"]) == bundle.dims.n_layers
    assert all(0.0 <= f <= 1.0 for f in g["fraction"])
    assert json.loads((d / "manifest.json").read_text())["config_sha256"] == cfg.sha256()


def test_merge_of_two_shards(calibrated):
    from energy_llm.core.capture import REFERENCES

    cfg, bundle, ev, rec = calibrated
    items = read_items(ev)
    for k in (0, 2):
        run_extract(cfg, bundle, items, ev, "two", k)
    m = run_merge(cfg, items, ev, "two", allow_incomplete=True)
    ids = [it.id for it in items[:3] + items[6:]]
    assert m["counts"]["completed"] == 4 and m["incomplete_shards"] == [1]
    d = cfg.extraction_dir("two")
    ans = pd.read_parquet(d / "answers.parquet").set_index("id")
    base = pd.read_parquet(d / "baselines.parquet").set_index("id")
    assert list(ans.index) == ids
    with h5py.File(d / "traces.h5") as f:
        assert list(f) == sorted(ids)
        assert f.attrs["betas"].tolist() == cfg.extraction.betas(rec["beta_star"])
        assert (f.attrs["topk_mass_k"], f.attrs["topk_ids_k"]) == (4, 8)
        assert list(f.attrs["reference_names"]) == list(REFERENCES)
        worst = dict.fromkeys(REFERENCES, 0.0)
        for i in ids:
            g, a = f[i], ans.loc[i]
            pos = g.attrs["reference_positions"].tolist()
            assert pos[1] == a["n_prompt_tokens"] - 1
            assert pos[2] == (-1 if a["empty_response"] else a["n_prompt_tokens"])
            assert g["energy"].chunks == (1, g["energy"].shape[1], 10)
            for r, p in zip(REFERENCES, pos):
                if p >= 0:
                    worst[r] = max(worst[r], float(g["xhat_top1_frac"][:, p].max()))
            resp = np.asarray(a["token_logits"])[:a["n_response_tokens"]]
            if len(resp):
                assert base.loc[i, "mean_neg_logit"] == pytest.approx(-resp.mean(), rel=1e-6)
    assert m["checks"]["max_xhat_top1_frac"] == pytest.approx(worst)
    g = m["checks"]["gamma_top_fraction_max"]
    assert g["k"] == 5 and g["value"] == max(m["gamma_diagnostic"]["fraction"])


def test_merge_refuses_items_with_different_betas(calibrated):
    from energy_llm.pipeline.extract import item_path

    cfg, bundle, ev, _ = calibrated
    items, _ = _run_all(cfg, bundle, ev, "b")
    with h5py.File(item_path(shard_path(cfg.extraction_dir("b"), 1), items[3].id), "r+") as f:
        f.attrs["betas"] = f.attrs["betas"] * 2
    with pytest.raises(RuntimeError, match="betas"):
        run_merge(cfg, items, ev, "b")


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


@pytest.mark.parametrize("key, value", [
    ("topk_ids_k", 4), ("beta_grid_log10", {"min": 0.0, "max": 3.0, "num": 7})])
def test_changed_config_refuses_to_resume(calibrated, key, value):
    cfg, bundle, ev, _ = calibrated
    items = read_items(ev)
    run_extract(cfg, bundle, items, ev, "c", 0, stop_after=1)
    setattr(cfg.extraction, key, value)
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

    from energy_llm.pipeline.signals import GracefulStop

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
    from energy_llm.core.capture import ResidualHooks, forward_item
    from energy_llm.core.prompting import build_prompt

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
    base = pd.read_parquet(d / "baselines.parquet").set_index("id")
    assert np.isnan(base.loc[items[0].id, "mean_logprob"])
    with h5py.File(d / "traces.h5") as f, h5py.File(d / "hidden.h5") as h:
        assert np.isfinite(f[items[0].id]["energy"][()]).all()
        assert np.isnan(h[items[0].id]["response_last"][()]).all()
        assert np.isnan(f[items[0].id]["dkl/response_first"][()]).all()
        assert np.isfinite(f[items[0].id]["dkl/prompt_last"][()]).all()


def test_smoke_summary(calibrated, capsys):
    from energy_llm.pipeline.smoke import format_summary, run_smoke

    cfg, bundle, ev, rec = calibrated
    s = run_smoke(cfg, bundle, read_items(ev), n=5, name="smoke-test")
    assert s["n_questions"] == 5 and s["n_completed"] == 5 and s["n_failed"] == 0
    assert s["model_dtype"] == "float32" and s["beta_star"] == rec["beta_star"]
    assert s["tf_token_match_pct"] == 100.0 and s["tf_exact_answers_pct"] == 100.0
    assert s["sec_per_question"] > 0 and s["peak_gpu_mem_gb"] is None   # CPU
    assert len(s["mean_norm_entropy_per_layer"]["prompt"]) == bundle.dims.n_layers
    assert {"n_empty_response", "n_truncated", "xhat_norm_max_rel_dev"} <= set(s)
    assert (cfg.extraction_dir("smoke-test") / "smoke_report.json").exists()
    text = format_summary(s)
    assert "TF == greedy" in text and "layer   0" in text
    assert s["tf_disagreements"]["n"] == 0 and s["tf_disagreements"]["all_below_delta"] is True
    assert "TF/greedy disagreements 0" in text
    assert f"report: {cfg.extraction_dir('smoke-test')}/smoke_report.json" in text
    assert "ablation top_gamma_zeroed" in text and "ablation dstar_zeroed" in text
    assert "without layer 0" in text and "beta_l" in text and "re-matched beta" in text
    L = bundle.dims.n_layers
    assert len(s["per_layer"]) == L
    for r in s["per_layer"]:
        assert 0 <= r["dstar"] < bundle.dims.d_model and 0 < r["dstar_share"] <= 1
        assert set(r["delta_norm_entropy"]) == {"intact", "top_gamma_zeroed", "dstar_zeroed"}
        assert set(r["criteria"]) == {"criterion3_xhat_prompt_last_median_gt_0.5",
                                      "criterion4_gamma_fraction_gt_0.5_or_sign_flip",
                                      "gamma_ablation_sign_flip"}
    first = s["bands_excluding_layer0"][0]          # None when the first third is layer 0 alone
    assert first is None if s["bands"][0]["layers"] == [0, 0] else first["layers"][0] == 1
    assert s["bands_excluding_layer0"][1:] == s["bands"][1:]
    # re-matched beta: the ablated bank's mean H_bar(prompt_last) returns to the intact value
    ab = s["ablation_betas"]
    assert ab["beta_matched"]["intact"] == rec["beta_star"]
    for k in ("top_gamma_zeroed", "dstar_zeroed"):
        bm = ab["beta_matched"][k]
        assert bm is None or bm > 0
        blk = s["bands"][0]["ablation"][k]
        assert (blk["beta_matched"] is None) == (bm is None)
    # intact at the re-matched beta is intact at beta*
    assert s["bands"][0]["ablation"]["intact"]["beta_matched"] == s["bands"][0]["ablation"]["intact"]["beta_star"]
    bl = s["beta_per_layer"]
    assert len(bl["per_layer"]) == L and bl["target"] == rec["target_norm_entropy"]
    assert bl["min"] is None or bl["min"] <= bl["max"]
    pl = s["prompt_last_entropy"]
    assert pl["target_band"] == pytest.approx([rec["target_norm_entropy"] - 0.05, rec["target_norm_entropy"] + 0.05])
    lo, hi = pl["calibration_bracket"]
    assert lo <= hi and pl["in_calibration_bracket"] == (lo <= pl["mean_h_prompt_last_beta_star"] <= hi)
    L = bundle.dims.n_layers
    assert len(s["bands"]) == 3 and s["bands"][0]["layers"][0] == 0 and s["bands"][-1]["layers"][1] == L - 1
    # the intact branch of the ablation equals the production traces (float16 storage)
    d = cfg.extraction_dir("smoke-test")
    ans = pd.read_parquet(d / "answers.parquet")
    deltas = []
    with h5py.File(d / "traces.h5") as f:
        for i, n_p in zip(ans["id"], ans["n_prompt_tokens"]):
            h, resp = f[i]["norm_entropy"][..., -1].astype(np.float64), f[i]["role_mask"][()] == 3
            if resp.any():
                deltas.append(h[:, resp].mean(1) - h[:, n_p - 1])
    for b in s["bands"]:
        lo, hi = b["layers"]
        want = np.mean(np.asarray(deltas)[:, lo:hi + 1])
        got = b["delta_norm_entropy_response_minus_prompt_last"]
        assert got["intact"] == pytest.approx(want, abs=2e-3)
        assert np.isfinite(got["top_gamma_zeroed"]) and np.isfinite(got["dstar_zeroed"])
        assert 0.0 <= b["gamma_top_fraction"] <= 1.0
        for r in ("question_last", "prompt_last", "response_first"):
            x = b["xhat_top1_frac"][r]
            assert 0.0 < x["median"] <= x["max"] <= 1.0


def test_tf_disagreements_delta_rule():
    import pandas as pd

    from energy_llm.pipeline.smoke import bf16_ulp, tf_disagreements

    assert bf16_ulp(1.0) == 2.0 ** -7 and bf16_ulp(-20.0) == 2.0 ** -3 and bf16_ulp(0.0) == 0.0
    ans = pd.DataFrame([
        {"id": "a", "generated_token_ids": [5, 6, 7], "tf_mismatch_positions": [1],
         "tf_mismatch_tf_tokens": [9], "tf_mismatch_margin_tf": [0.01], "tf_mismatch_margin_gen": [0.02],
         "tf_mismatch_top_logit": [20.0], "tf_logit_diff": [0.05, 0.9, 0.05], "tf_logit_diff_step0": 0.05},
        {"id": "b", "generated_token_ids": [1, 2], "tf_mismatch_positions": [1],
         "tf_mismatch_tf_tokens": [3], "tf_mismatch_margin_tf": [0.5], "tf_mismatch_margin_gen": [0.4],
         "tf_mismatch_top_logit": [1.0], "tf_logit_diff": [0.1, 0.6], "tf_logit_diff_step0": 0.1},
    ])
    d = tf_disagreements(ans)
    # non-divergent steps: 0.05, 0.05, 0.1 -> p99 (linear) 0.099; the 0.9 / 0.6 steps are excluded
    assert d["p99_logit_diff_non_divergent"] == pytest.approx(0.099)
    assert d["max_logit_diff_step0"] == 0.1
    a, b = d["list"]
    assert a["delta"] == 3 * 2.0 ** -3 and a["below_delta"]                # 3 ulp at |20| = 0.375
    assert b["token_gen"] == 2 and b["token_tf"] == 3 and not b["below_delta"]
    assert d["n"] == 2 and d["all_below_delta"] is False
