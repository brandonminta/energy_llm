"""Analysis stage: aggregate trajectory artifacts into an analysis JSON.

CPU-only — no model or GPU required.  Reads the per-sample .npz and .json
files produced by ``run_trajectory`` and computes:

  1. Per-layer delta metrics (generation_mean[l] − prefill[l])
  2. Global distribution divergences (KL, JS, Hellinger) per sample,
     treating the layer axis as the probability space
  3. Scalar hallucination scores per sample
  4. Dataset-level aggregation (mean/std per layer, score summary)

Design note on divergences
--------------------------
The global divergences (KL, JS, Hellinger) compare softmax(prefill_energy)
and softmax(gen_energy_mean) where the probability space is the L-layer axis.
They produce ONE scalar per sample, not per-layer values.  They are reported
in the output JSON under ``global_divergences``.

The genuine per-layer signals are the delta_* arrays and the raw energy/entropy
trajectories.  These are in ``layer_metrics``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hopfield_llm.metrics.divergences import (
    SampleDivergenceResult,
    hellinger,
    js_divergence,
    kl_divergence,
)
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.analysis")


def _load_sample(traj_dir: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta = json.loads((traj_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(traj_dir / f"{sample_id}.npz"))
    return meta, arrays


def _compute_sample_divergence(arrays: dict[str, np.ndarray]) -> SampleDivergenceResult:
    """Compute divergence metrics for one (prefill, generation) pair.

    Per-layer deltas are straightforward: gen_mean[l] − prefill[l].

    Global divergences use the layer axis as the probability space:
        p = softmax(−prefill_energy)   [L]
        q = softmax(−gen_energy_mean)  [L]
    This is a single scalar per sample, not a per-layer value.
    """
    # Mean-pool generation over tokens → [L]
    gen_energy_mean      = np.nanmean(arrays["gen_energy"],      axis=1)
    gen_entropy_mean     = np.nanmean(arrays["gen_entropy"],     axis=1)
    gen_norm_entropy_mean = np.nanmean(arrays["gen_norm_entropy"], axis=1)
    gen_top_act_mean     = np.nanmean(arrays["gen_top_act"],     axis=1)
    gen_lse_mean         = np.nanmean(arrays["gen_lse"],         axis=1)

    # Per-layer delta arrays [L]
    delta_energy         = gen_energy_mean      - arrays["prefill_energy"]
    delta_entropy        = gen_entropy_mean     - arrays["prefill_entropy"]
    delta_norm_entropy   = gen_norm_entropy_mean - arrays["prefill_norm_entropy"]
    delta_top_activation = gen_top_act_mean     - arrays["prefill_top_act"]
    delta_mean_activation = gen_lse_mean        - arrays["prefill_lse"]

    # Global divergences: softmax of negated energies across the layer axis
    prefill_e = torch.from_numpy(
        np.nan_to_num(-arrays["prefill_energy"].astype(np.float64), nan=0.0)
    )
    gen_e = torch.from_numpy(
        np.nan_to_num(-gen_energy_mean.astype(np.float64), nan=0.0)
    )
    p = torch.softmax(prefill_e, dim=0)
    q = torch.softmax(gen_e, dim=0)

    kl_fwd, kl_rev = kl_divergence(p, q, direction="both")
    js              = js_divergence(p, q)
    hell            = hellinger(p, q)

    return SampleDivergenceResult(
        kl_fwd=float(kl_fwd),
        kl_rev=float(kl_rev),
        js=float(js),
        hellinger=float(hell),
        delta_energy=delta_energy.astype(np.float64),
        delta_entropy=delta_entropy.astype(np.float64),
        delta_norm_entropy=delta_norm_entropy.astype(np.float64),
        delta_top_activation=delta_top_activation.astype(np.float64),
        delta_mean_activation=delta_mean_activation.astype(np.float64),
    )


def analyze_trajectories(
    traj_dir: str | Path,
    output: str | Path,
    score_metric: str = "js",
    score_aggregation: str = "mean",
    signal_zone: tuple[int, int] | None = None,
) -> Path:
    """Aggregate trajectory .npz/.json files into an analysis JSON.

    Writes a single ``analysis.json`` compatible with ``visualize_analysis``.

    Args:
        traj_dir:         Directory containing .npz/.json trajectory pairs.
        output:           Path for the output analysis JSON file.
        score_metric:     Metric for the scalar hallucination score.
        score_aggregation: Aggregation method (mean / max / weighted).
        signal_zone:      Optional (start, end) layer slice for scoring.

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

    n_scored = len(all_scores)
    scores_arr = np.array(all_scores)

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

    # -- Global divergences: one scalar per sample → distribution over N --
    GLOBAL_METRIC_NAMES = ["kl_fwd", "kl_rev", "js", "hellinger"]
    global_divergences: dict[str, dict[str, Any]] = {}
    for mn in GLOBAL_METRIC_NAMES:
        vals = np.array([getattr(d, mn) for d in all_divs])
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
        "artifact_type":     "analysis",
        "n_samples":         len(sample_ids),
        "n_scored":          n_scored,
        "n_layers":          n_layers,
        "score_metric":      score_metric,
        "score_aggregation": score_aggregation,
        "score_summary":     score_summary,
        "score_by_category": score_by_category,
        "layer_metrics":     layer_metrics,
        "global_divergences": global_divergences,
    }

    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    log.info(
        "Analysis complete: %d samples, mean score=%.6f → %s",
        n_scored, score_summary["mean"], output,
    )
    return output
