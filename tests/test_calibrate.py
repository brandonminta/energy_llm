import json

import numpy as np
import pytest

from energy_llm.calibrate import CalibrationError, interpolate_log_beta, run_calibrate
from energy_llm.extract import run_extract
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


def test_calibration_hits_target(calibrated):
    cfg, _, _, rec = calibrated
    curve = np.asarray(rec["mean_norm_entropy_curve"])
    assert np.all(np.diff(curve) <= 1e-6)          # decreasing in beta
    grid = np.log(rec["beta_grid"])
    h_at = np.interp(np.log(rec["beta_star"]), grid, curve)
    assert h_at == pytest.approx(0.55, abs=1e-6)
    assert rec["n_questions"] == 4 and rec["failed_ids"] == []
    assert cfg.bank_h5_path.exists()


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
