import json

import numpy as np
import pytest

from energy_llm.pipeline.calibrate import CalibrationError, interpolate_log_beta, run_calibrate
from energy_llm.pipeline.extract import run_extract
from energy_llm.storage import read_items
from tests.conftest import QUESTION_WORDS, make_bundle, make_cfg, make_questions, write_jsonl


def test_interpolation_in_log_beta():
    grid = [1.0, 10.0, 100.0]
    curve = np.array([0.9, 0.6, 0.2])
    assert interpolate_log_beta(grid, curve, 0.6) == pytest.approx(10.0)
    assert interpolate_log_beta(grid, curve, 0.4) == pytest.approx(10 ** 1.5)


def test_unbracketed_target_fails_instead_of_clamping():
    with pytest.raises(CalibrationError, match="not bracketed"):
        interpolate_log_beta([1.0, 10.0], np.array([0.9, 0.8]), 0.55)


def test_calibration_hits_target_for_each_variant(calibrated):
    cfg, _, _, rec = calibrated
    assert rec["primary"] == "cos_gamma_query"
    assert set(rec["variants"]) == {"cos_gamma_query", "cos_gamma_key"}
    grid = np.log(rec["beta_grid"])
    for v, r in rec["variants"].items():
        curve = np.asarray(r["mean_norm_entropy_curve"])
        assert np.all(np.diff(curve) <= 1e-6), v          # decreasing in beta
        h_at = np.interp(np.log(r["beta_star"]), grid, curve)
        assert h_at == pytest.approx(0.55, abs=1e-6), v
        p = r["grid_position"]
        assert p["decades_from_min"] + p["decades_from_max"] == pytest.approx(5.0)   # grid 10^-1 .. 10^4
        i, j = p["bracket_indices"]
        assert rec["beta_grid"][i] <= r["beta_star"] <= rec["beta_grid"][j]
    assert rec["variants"]["cos_gamma_query"]["beta_star"] != rec["variants"]["cos_gamma_key"]["beta_star"]
    assert rec["n_questions"] == 4 and rec["failed_ids"] == []
    assert cfg.bank_h5_path.exists()


def test_grid_position_and_edge_warning():
    from energy_llm.pipeline.calibrate import EDGE_DECADES, format_beta_stars, grid_position

    grid = [10 ** (k / 4) for k in range(17)]          # 10^0 .. 10^4
    mid, low = grid_position(grid, 100.0), grid_position(grid, 10 ** 0.3)
    assert mid["decades_from_min"] == pytest.approx(2.0) and not mid["near_edge"]
    assert low["decades_from_min"] == pytest.approx(0.3) and low["near_edge"]
    assert EDGE_DECADES == 0.5
    assert grid_position(grid, 10 ** 3.6)["near_edge"]
    rec = {"primary": "a", "beta_grid": grid, "target_norm_entropy": 0.55,
           "variants": {"a": {"beta_star": 100.0, "grid_position": mid},
                        "b": {"beta_star": 10 ** 0.3, "grid_position": low}}}
    text = format_beta_stars(rec).splitlines()
    assert "beta*[a] (primary) = 100" in text[1] and "WARNING" not in text[1]
    assert "beta*[b]" in text[2] and "WARNING" in text[2]


def test_extract_refuses_a_beta_star_record_missing_a_variant(calibrated):
    from energy_llm.storage import load_json, write_json_atomic

    cfg, bundle, ev, rec = calibrated
    old = {k: v for k, v in rec.items() if k != "variants"} | {
        "variants": {"cos_gamma_key": rec["variants"]["cos_gamma_key"]}}
    write_json_atomic(cfg.beta_star_path, old)
    with pytest.raises(RuntimeError, match="no beta\\* for \\['cos_gamma_query'\\]"):
        run_extract(cfg, bundle, read_items(ev), ev, "t", 0)
    legacy = {k: v for k, v in rec.items() if k != "variants"} | {"beta_star": 1.0}
    write_json_atomic(cfg.beta_star_path, legacy)
    with pytest.raises(RuntimeError, match="re-run `energy-llm calibrate`"):
        run_extract(cfg, bundle, read_items(ev), ev, "t", 0)
    assert load_json(cfg.beta_star_path)["beta_star"] == 1.0


def test_calibrate_is_idempotent(calibrated, tmp_path):
    cfg, bundle, _, rec = calibrated
    again = run_calibrate(cfg, bundle, [], tmp_path / "unused.jsonl")
    assert again == rec


def test_extract_refuses_without_beta_star(tmp_path, qwen_pair):
    bundle = make_bundle(*qwen_pair)
    cfg = make_cfg(tmp_path / "run")
    ev = write_jsonl(tmp_path / "eval.jsonl", make_questions(3, QUESTION_WORDS, 2), "q")
    with pytest.raises(RuntimeError, match="calibrate"):
        run_extract(cfg, bundle, read_items(ev), ev, "t", 0)


def test_extract_refuses_overlap_with_warmup(calibrated, tmp_path):
    cfg, bundle, _, rec = calibrated
    warm_q = json.loads(open(tmp_path / "warmup.jsonl").readline())["question"]
    ev = tmp_path / "overlap.jsonl"
    ev.write_text(json.dumps({"id": "new", "question": warm_q.upper()}) + "\n")
    with pytest.raises(RuntimeError, match="overlap"):
        run_extract(cfg, bundle, read_items(ev), ev, "t", 0)


def test_extract_refuses_beta_from_other_context(calibrated, tmp_path):
    cfg, bundle, ev, _ = calibrated
    cfg.prompt.system = "be verbose"
    with pytest.raises(RuntimeError, match="different"):
        run_extract(cfg, bundle, read_items(ev), ev, "t", 0)
