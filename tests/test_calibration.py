"""Beta calibration tests (subsec:beta_calibration): monotone curve, exact
log-linear interpolation, and the clamping path."""

import math

import numpy as np
import pytest

from energy_llm.calibration import interpolate_beta_star, mean_norm_entropy_curve

GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]


class TestCurve:
    def test_monotone_decreasing(self):
        rng = np.random.default_rng(0)
        scores = [rng.normal(size=(4, 256)).astype(np.float32) for _ in range(5)]
        curve = mean_norm_entropy_curve(scores, GRID)
        assert np.all(np.diff(curve) <= 1e-12)
        assert curve[0] <= 1.0 and curve[-1] >= 0.0

    def test_beta_zero_limit_near_uniform(self):
        rng = np.random.default_rng(1)
        scores = [rng.normal(size=(2, 128)).astype(np.float32) * 0.001]
        curve = mean_norm_entropy_curve(scores, [0.001])
        assert curve[0] == pytest.approx(1.0, abs=1e-6)


class TestInterpolation:
    def test_exact_on_log_linear_curve(self):
        """If the curve is exactly linear in log(beta), interpolation must
        recover the exact crossing point."""
        log_b = np.log(GRID)
        slope, intercept = -0.2, 1.0
        curve = intercept + slope * log_b
        target = 0.55
        expected_log_b = (target - intercept) / slope
        beta_star, clamped = interpolate_beta_star(GRID, curve, target)
        assert not clamped
        assert math.log(beta_star) == pytest.approx(expected_log_b, rel=1e-12)

    def test_target_on_grid_point(self):
        curve = np.linspace(0.9, 0.1, len(GRID))
        target = float(curve[3])
        beta_star, clamped = interpolate_beta_star(GRID, curve, target)
        assert not clamped
        assert beta_star == pytest.approx(GRID[3], rel=1e-9)

    def test_clamp_low_beta_endpoint(self):
        """Target above the whole curve -> clamp to the endpoint whose value
        is closest (here beta=1)."""
        curve = np.linspace(0.5, 0.1, len(GRID))
        beta_star, clamped = interpolate_beta_star(GRID, curve, 0.9)
        assert clamped
        assert beta_star == GRID[0]

    def test_clamp_high_beta_endpoint(self):
        curve = np.linspace(0.9, 0.6, len(GRID))
        beta_star, clamped = interpolate_beta_star(GRID, curve, 0.1)
        assert clamped
        assert beta_star == GRID[-1]


class TestEndToEndCalibration:
    def test_calibrate_writes_snapshot_and_record(
        self, tiny_model, tiny_tokenizer, tiny_banks, tiny_samples, tmp_path
    ):
        from energy_llm.calibration import calibrate_beta
        from energy_llm.config import ExperimentConfig, load_config

        cfg = ExperimentConfig(name="test", output_dir=str(tmp_path / "run"))
        cfg.calibration.num_samples = 4
        record = calibrate_beta(cfg, tiny_model, tiny_tokenizer, tiny_banks, tiny_samples)

        assert record["beta_star"] > 0
        assert len(record["calibration_sample_ids"]) == 4
        assert len(record["mean_norm_entropy_curve"]) == len(record["beta_grid"])
        assert (cfg.calibration_dir / "calibration.json").exists()
        assert (cfg.calibration_dir / "calibration_scores.npz").exists()

        # the snapshot now carries beta_star and wins on reload
        yaml_path = tmp_path / "e_test.yaml"
        yaml_path.write_text(f"name: test\noutput_dir: {cfg.output_dir}\n")
        reloaded = load_config(yaml_path)
        assert reloaded.calibration.beta_star == pytest.approx(record["beta_star"])
