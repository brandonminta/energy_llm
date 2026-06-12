"""beta* calibration (subsec:beta_calibration, subsec:impl_beta).

Warm-up: capture prefill reference retrieval scores for N_c training-split
questions. For each beta on the fixed grid, evaluate the mean normalised
entropy of the induced softmax across all layers and warm-up samples (the
softmax is recomputed from cached score vectors — no further forward
passes). beta* is found by linear interpolation in log(beta) between the two
grid points bracketing the target H_bar*; if the target is not bracketed,
beta* is clamped to the nearest endpoint and the clamping is logged.

The calibrated value is written into the experiment config snapshot so every
subsequent stage reads the same beta*. Calibrated once per model (E2
recalibrates). The calibration sample ids are recorded and later forced into
the probe's train split, which keeps test-set information out of beta*.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from energy_llm import metrics
from energy_llm.config import ExperimentConfig
from energy_llm.data import Sample, calibration_sample_ids
from energy_llm.io_artifacts import save_json_atomic, save_npz_atomic

log = logging.getLogger("energy_llm.calibration")


def mean_norm_entropy_curve(
    score_list: list[np.ndarray], beta_grid: list[float]
) -> np.ndarray:
    """Mean normalised entropy of softmax(beta*s) across all captured layers
    and warm-up samples, for each beta on the grid. Monotone decreasing."""
    curve = np.empty(len(beta_grid))
    for i, beta in enumerate(beta_grid):
        values = [
            metrics.norm_entropy(metrics.retrieval_distribution(s, beta)).mean()
            for s in score_list  # s: [L, K] per sample
        ]
        curve[i] = float(np.mean(values))
    return curve


def interpolate_beta_star(
    beta_grid: list[float], curve: np.ndarray, target: float
) -> tuple[float, bool]:
    """Linear interpolation in log(beta) between the bracketing grid points.

    Returns (beta_star, clamped). The curve decreases in beta, so the target
    is bracketed at i where curve[i] >= target >= curve[i+1]. If never
    bracketed, clamp to the grid endpoint whose value is closest to target.
    """
    for i in range(len(beta_grid) - 1):
        hi, lo = curve[i], curve[i + 1]
        if hi >= target >= lo:
            if math.isclose(hi, lo):
                return float(beta_grid[i]), False
            frac = (hi - target) / (hi - lo)
            log_b = math.log(beta_grid[i]) + frac * (
                math.log(beta_grid[i + 1]) - math.log(beta_grid[i])
            )
            return float(math.exp(log_b)), False
    nearest = int(np.argmin(np.abs(curve - target)))
    return float(beta_grid[nearest]), True


def calibrate_beta(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    banks: dict[str, Any],
    samples: list[Sample],
) -> dict[str, Any]:
    """Run the full warm-up + interpolation and freeze beta* into the
    config snapshot. Returns the calibration record."""
    from energy_llm.trajectory import ProbePointHooks, capture_prefill_reference_scores

    cal_ids = calibration_sample_ids(samples, cfg.calibration.num_samples, seed=cfg.seed)
    by_id = {s.sample_id: s for s in samples}
    cal_dir = cfg.calibration_dir
    cal_dir.mkdir(parents=True, exist_ok=True)

    hooks = ProbePointHooks(model, banks, score_dtype=cfg.trajectory.score_dtype)
    score_list: list[np.ndarray] = []
    try:
        for sid in cal_ids:
            s = capture_prefill_reference_scores(model, tokenizer, hooks, by_id[sid].question)
            score_list.append(s.astype(np.float32))
    finally:
        hooks.remove()

    # Cache the warm-up scores for audit/recompute (no forward passes later).
    save_npz_atomic(
        cal_dir / "calibration_scores.npz",
        **{sid: s for sid, s in zip(cal_ids, score_list)},
    )

    grid = list(cfg.calibration.beta_grid)
    curve = mean_norm_entropy_curve(score_list, grid)
    target = cfg.calibration.target_norm_entropy
    beta_star, clamped = interpolate_beta_star(grid, curve, target)
    if clamped:
        log.warning(
            "beta* target H_bar*=%.3f not bracketed by the grid — clamped to "
            "beta=%.4g (curve range [%.4f, %.4f])",
            target, beta_star, curve.min(), curve.max(),
        )

    record = {
        "beta_star": beta_star,
        "clamped": clamped,
        "target_norm_entropy": target,
        "beta_grid": grid,
        "mean_norm_entropy_curve": curve.tolist(),
        "calibration_sample_ids": cal_ids,
        "n_calibration_samples": len(cal_ids),
        "model_id": banks["model_id"],
        "bank_id": banks["bank_id"],
    }
    save_json_atomic(cal_dir / "calibration.json", record)

    # Freeze beta* into the snapshot read by all later stages.
    cfg.calibration.beta_star = beta_star
    cfg.save_snapshot()
    log.info("beta* = %.4f (clamped=%s) written to %s", beta_star, clamped, cfg.snapshot_path)
    return record
