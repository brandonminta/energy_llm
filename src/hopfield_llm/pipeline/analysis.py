"""Analysis stage: aggregate trajectory artifacts into an analysis JSON.

CPU-only — no model or GPU required.  Reads the per-sample .npz and .json
files produced by ``run_trajectory`` and computes:

  1. Per-layer delta metrics (generation_mean[l] − prefill[l])
  2. Interpretable scalar summaries per sample (energy_shift_l1, etc.)
  3. Scalar hallucination scores per sample
  4. Dataset-level aggregation (mean/std per layer, score summary)

Per-layer KL/JS/Hellinger divergences require distributions saved during
trajectory collection (save_distributions=True in capture_prefill /
capture_generation).  The standard .npz files do not contain distributions,
so those fields are NaN in the analysis output.  Run capture with
save_distributions=True and pass them to compute_sample_divergences directly
to get the full M2 divergence picture.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hopfield_llm.metrics.divergences import SampleDivergenceResult
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.analysis")


def _load_sample(traj_dir: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta   = json.loads((traj_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(traj_dir / f"{sample_id}.npz"))
    return meta, arrays


def _compute_sample_divergence(arrays: dict[str, np.ndarray]) -> SampleDivergenceResult:
    """Compute divergence metrics for one (prefill, generation) pair from .npz arrays.

    Per-layer deltas: generation_mean[l] − prefill[l].
    Per-layer KL/JS/Hellinger: NaN (distributions not stored in .npz).
    Scalar summaries: computed from delta_energy.
    """
    # Mean-pool generation metrics over tokens → [L]
    gen_energy_mean       = np.nanmean(arrays["gen_energy"],       axis=1)
    gen_entropy_mean      = np.nanmean(arrays["gen_entropy"],      axis=1)
    gen_norm_entropy_mean = np.nanmean(arrays["gen_norm_entropy"], axis=1)
    gen_top_act_mean      = np.nanmean(arrays["gen_top_act"],      axis=1)
    gen_lse_mean          = np.nanmean(arrays["gen_lse"],          axis=1)

    delta_energy          = gen_energy_mean       - arrays["prefill_energy"]
    delta_entropy         = gen_entropy_mean      - arrays["prefill_entropy"]
    delta_norm_entropy    = gen_norm_entropy_mean - arrays["prefill_norm_entropy"]
    delta_top_activation  = gen_top_act_mean      - arrays["prefill_top_act"]
    delta_mean_activation = gen_lse_mean          - arrays["prefill_lse"]

    # Scalar energy summaries
    de_abs           = np.abs(np.where(np.isnan(delta_energy), 0.0, delta_energy))
    energy_shift_l1  = float(np.sum(de_abs))
    energy_shift_l2  = float(np.sqrt(np.sum(de_abs ** 2)))
    peak_delta_layer = int(np.argmax(de_abs))

    # KL/JS/Hellinger are NaN — distributions not in .npz
    L     = len(arrays["prefill_energy"])
    T_gen = arrays["gen_energy"].shape[1]
    nan_LT = np.full((L, T_gen), np.nan, dtype=np.float32)

    return SampleDivergenceResult(
        delta_energy=delta_energy.astype(np.float64),
        delta_entropy=delta_entropy.astype(np.float64),
        delta_norm_entropy=delta_norm_entropy.astype(np.float64),
        delta_top_activation=delta_top_activation.astype(np.float64),
        delta_mean_activation=delta_mean_activation.astype(np.float64),
        kl_gen_to_prompt_per_layer=nan_LT,
        js_gen_to_prompt_per_layer=nan_LT.copy(),
        hellinger_gen_to_prompt_per_layer=nan_LT.copy(),
        energy_shift_l1=energy_shift_l1,
        energy_shift_l2=energy_shift_l2,
        peak_delta_layer=peak_delta_layer,
        gen_drift_mean=float(np.nan),
        gen_drift_max=float(np.nan),
    )


def analyze_trajectories(
    traj_dir: str | Path,
    output: str | Path,
    score_metric: str = "delta_energy",
    score_aggregation: str = "mean",
    signal_zone: tuple[int, int] | None = None,
) -> Path:
    """Aggregate trajectory .npz/.json files into an analysis JSON.

    Writes a single ``analysis.json`` compatible with ``visualize_analysis``.

    Args:
        traj_dir:          Directory containing .npz/.json trajectory pairs.
        output:            Path for the output analysis JSON file.
        score_metric:      Metric for the scalar hallucination score.
        score_aggregation: Aggregation method (mean / max / weighted).
        signal_zone:       Optional (start, end) layer slice for scoring.

    Returns:
        Path to the written analysis JSON file.
    """
    traj_dir = Path(traj_dir)
    output   = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = [
        jf.stem
        for jf in sorted(traj_dir.glob("*.json"))
        if not jf.stem.endswith("_hpre")
        and (traj_dir / f"{jf.stem}.npz").exists()
    ]

    if not sample_ids:
        raise ValueError(f"No trajectory samples found in {traj_dir}")

    log.info("Analyzing %d samples from %s", len(sample_ids), traj_dir)

    all_divs:   list[SampleDivergenceResult] = []
    all_scores: list[float]                  = []
    all_metas:  list[dict[str, Any]]         = []
    category_scores: dict[str, list[float]]  = defaultdict(list)
    n_layers = None

    for sample_id in sample_ids:
        try:
            meta, arrays = _load_sample(traj_dir, sample_id)
            all_metas.append(meta)
            if n_layers is None:
                n_layers = arrays["prefill_energy"].shape[0]

            div = _compute_sample_divergence(arrays)
            all_divs.append(div)

            score = hallucination_score(
                div,
                signal_zone=signal_zone,
                metric=score_metric,
                aggregation=score_aggregation,
            )
            all_scores.append(score)
            category_scores[meta.get("source", "unknown")].append(score)

        except Exception as exc:
            log.warning("Skipping %s: %s", sample_id, exc)

    if not all_divs:
        raise ValueError("No samples could be analyzed successfully")

    n_scored    = len(all_scores)
    scores_arr  = np.array(all_scores)

    # -- Per-layer delta metrics: stack [N, L] → mean/std --
    LAYER_METRIC_NAMES = [
        "delta_energy", "delta_entropy", "delta_norm_entropy",
        "delta_top_activation", "delta_mean_activation",
    ]
    layer_metrics: dict[str, dict[str, Any]] = {}
    for mn in LAYER_METRIC_NAMES:
        stacked = np.stack([getattr(d, mn) for d in all_divs], axis=0)  # [N, L]
        layer_metrics[mn] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "std":  np.nanstd(stacked,  axis=0).tolist(),
            "n_samples": n_scored,
        }

    # -- Interpretable scalar summaries: one value per sample → distribution --
    SCALAR_METRIC_NAMES = [
        "energy_shift_l1", "energy_shift_l2", "peak_delta_layer",
        "gen_drift_mean", "gen_drift_max",
    ]
    global_divergences: dict[str, dict[str, Any]] = {}
    for mn in SCALAR_METRIC_NAMES:
        vals = np.array([getattr(d, mn) for d in all_divs], dtype=np.float64)
        global_divergences[mn] = {
            "mean": float(np.nanmean(vals)),
            "std":  float(np.nanstd(vals)),
            "min":  float(np.nanmin(vals)),
            "max":  float(np.nanmax(vals)),
            "n_samples": n_scored,
        }

    score_summary: dict[str, Any] = {
        "count": n_scored,
        "mean":  float(np.mean(scores_arr)),
        "std":   float(np.std(scores_arr)),
        "min":   float(np.min(scores_arr)),
        "max":   float(np.max(scores_arr)),
    }

    score_by_category: dict[str, dict[str, Any]] = {}
    for cat, cat_scores in sorted(category_scores.items()):
        arr = np.array(cat_scores)
        score_by_category[cat] = {
            "count": len(cat_scores),
            "mean":  float(np.mean(arr)),
            "std":   float(np.std(arr)),
            "min":   float(np.min(arr)),
            "max":   float(np.max(arr)),
        }

    analysis: dict[str, Any] = {
        "artifact_type":      "analysis",
        "n_samples":          len(sample_ids),
        "n_scored":           n_scored,
        "n_layers":           n_layers,
        "score_metric":       score_metric,
        "score_aggregation":  score_aggregation,
        "score_summary":      score_summary,
        "score_by_category":  score_by_category,
        "layer_metrics":      layer_metrics,
        "global_divergences": global_divergences,
    }

    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    log.info(
        "Analysis complete: %d samples, mean score=%.6f → %s",
        n_scored, score_summary["mean"], output,
    )
    return output
