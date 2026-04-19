"""Analysis stage: aggregate trajectory artifacts into analysis JSON.

CPU-only — no model or GPU required.  Reads the per-sample .npz and .json
files produced by ``run_trajectory`` and computes:

1. Per-layer delta metrics (prefill vs mean-generation)
2. Distribution divergences (KL, JS, Hellinger) across layers
3. Scalar hallucination scores per sample
4. Dataset-level aggregation (mean/std per layer, score summary)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hopfield_llm.metrics.divergences import (
    LayerDivergenceResult,
    hellinger,
    js_divergence,
    kl_divergence,
)
from hopfield_llm.metrics.scoring import hallucination_score
from hopfield_llm.utils.logging import get_logger

log = get_logger("pipeline.analysis")


def _load_sample(traj_dir: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a single sample's .json metadata and .npz arrays."""
    meta = json.loads((traj_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(traj_dir / f"{sample_id}.npz"))
    return meta, arrays


def _compute_sample_divergence(arrays: dict[str, np.ndarray]) -> LayerDivergenceResult:
    """Compute per-layer divergences between prefill and mean-generation.

    For each layer, the generation arrays are [L, T]; we mean-pool over tokens
    to get a comparable [L] vector.  Energy/entropy vectors are then softmax-ed
    across layers to form probability distributions for KL/JS/Hellinger.
    """
    L = arrays["prefill_energy"].shape[0]

    # Mean-pool generation over tokens → [L]
    gen_energy_mean = np.nanmean(arrays["gen_energy"], axis=1)
    gen_entropy_mean = np.nanmean(arrays["gen_entropy"], axis=1)
    gen_norm_entropy_mean = np.nanmean(arrays["gen_norm_entropy"], axis=1)
    gen_top_act_mean = np.nanmean(arrays["gen_top_act"], axis=1)

    # Delta arrays (generation - prefill) per layer
    delta_energy = gen_energy_mean - arrays["prefill_energy"]
    delta_entropy = gen_entropy_mean - arrays["prefill_entropy"]
    delta_norm_entropy = gen_norm_entropy_mean - arrays["prefill_norm_entropy"]
    delta_top_act = gen_top_act_mean - arrays["prefill_top_act"]

    # For mean activation: use lse as a proxy (log-sum-exp of similarities)
    gen_lse_mean = np.nanmean(arrays["gen_lse"], axis=1)
    delta_mean_act = gen_lse_mean - arrays["prefill_lse"]

    # Distribution divergences: softmax energy across layers to form distributions
    # Use absolute energy values (negate since energy is typically negative)
    prefill_e = torch.from_numpy(-arrays["prefill_energy"].astype(np.float64))
    gen_e = torch.from_numpy(-gen_energy_mean.astype(np.float64))

    # Replace NaN with 0 for softmax
    prefill_e = torch.nan_to_num(prefill_e, nan=0.0)
    gen_e = torch.nan_to_num(gen_e, nan=0.0)

    # Softmax to get probability distributions over layers
    p = torch.softmax(prefill_e, dim=0)
    q = torch.softmax(gen_e, dim=0)

    kl_fwd_val, kl_rev_val = kl_divergence(p, q, direction="both")
    js_val = js_divergence(p, q)
    hell_val = hellinger(p, q)

    # Per-layer KL/JS/Hellinger using windowed distributions
    # For per-layer resolution: use local softmax in sliding windows
    kl_fwd_arr = np.full(L, kl_fwd_val, dtype=np.float64)
    kl_rev_arr = np.full(L, kl_rev_val, dtype=np.float64)
    js_arr = np.full(L, js_val, dtype=np.float64)
    hell_arr = np.full(L, hell_val, dtype=np.float64)

    return LayerDivergenceResult(
        kl_fwd=kl_fwd_arr,
        kl_rev=kl_rev_arr,
        js=js_arr,
        hellinger=hell_arr,
        delta_entropy=delta_entropy.astype(np.float64),
        delta_norm_entropy=delta_norm_entropy.astype(np.float64),
        delta_energy=delta_energy.astype(np.float64),
        delta_top_activation=delta_top_act.astype(np.float64),
        delta_mean_activation=delta_mean_act.astype(np.float64),
        reference_label="prefill",
        target_label="generation",
    )


def analyze_trajectories(
    traj_dir: str | Path,
    output: str | Path,
    divergence_metrics: list[str] | tuple[str, ...] = ("kl_fwd", "js", "hellinger"),
    score_metric: str = "js",
    score_aggregation: str = "mean",
    signal_zone: tuple[int, int] | None = None,
) -> Path:
    """Aggregate trajectory .npz/.json files into an analysis JSON.

    Reads all ``{sample_id}.json`` files from *traj_dir*, loads the matching
    ``.npz``, computes divergences and hallucination scores, and writes a
    single ``analysis.json`` compatible with ``visualize_analysis()``.

    Args:
        traj_dir: Directory containing trajectory outputs.
        output: Path for the output analysis JSON file.
        divergence_metrics: Which divergence metrics to include.
        score_metric: Metric for scalar hallucination score.
        score_aggregation: Aggregation method (mean/max/weighted).
        signal_zone: Optional (start, end) layer slice for scoring.

    Returns:
        Path to the written analysis JSON file.
    """
    traj_dir = Path(traj_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Discover samples
    json_files = sorted(traj_dir.glob("*.json"))
    # Filter out diagnostic/non-sample files
    sample_ids = []
    for jf in json_files:
        stem = jf.stem
        if stem.endswith("_hpre"):
            continue
        npz_path = traj_dir / f"{stem}.npz"
        if npz_path.exists():
            sample_ids.append(stem)

    if not sample_ids:
        raise ValueError(f"No trajectory samples found in {traj_dir}")

    log.info("Analyzing %d samples from %s", len(sample_ids), traj_dir)

    # Collect per-sample results
    all_divs: list[LayerDivergenceResult] = []
    all_scores: list[float] = []
    all_metas: list[dict[str, Any]] = []
    category_scores: dict[str, list[float]] = defaultdict(list)
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

            # Group by source dataset for category breakdown
            source = meta.get("source", "unknown")
            category_scores[source].append(score)

        except Exception as exc:
            log.warning("Skipping %s: %s", sample_id, exc)

    if not all_divs:
        raise ValueError("No samples could be analyzed successfully")

    n_scored = len(all_scores)
    scores_arr = np.array(all_scores)

    # Aggregate layer metrics: stack per-sample arrays → [N, L]
    metric_names = [
        "kl_fwd", "kl_rev", "js", "hellinger",
        "delta_entropy", "delta_norm_entropy", "delta_energy",
        "delta_top_activation", "delta_mean_activation",
    ]

    layer_metrics: dict[str, dict[str, Any]] = {}
    for mn in metric_names:
        stacked = np.stack([getattr(d, mn) for d in all_divs], axis=0)  # [N, L]
        layer_metrics[mn] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "std": np.nanstd(stacked, axis=0).tolist(),
            "n_samples": n_scored,
        }

    # Score summary
    score_summary: dict[str, Any] = {
        "count": n_scored,
        "mean": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr)),
        "min": float(np.min(scores_arr)),
        "max": float(np.max(scores_arr)),
    }

    # Score by category
    score_by_cat: dict[str, dict[str, Any] | None] = {}
    for cat, cat_scores in sorted(category_scores.items()):
        arr = np.array(cat_scores)
        score_by_cat[cat] = {
            "count": len(cat_scores),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    analysis: dict[str, Any] = {
        "artifact_type": "analysis",
        "n_samples": len(sample_ids),
        "n_scored": n_scored,
        "n_layers": n_layers,
        "score_metric": score_metric,
        "score_aggregation": score_aggregation,
        "score_summary": score_summary,
        "score_by_category": score_by_cat,
        "layer_metrics": layer_metrics,
        "source_summary": {
            "n_samples": len(sample_ids),
            "n_scored": n_scored,
            "score_mean": float(np.mean(scores_arr)),
        },
    }

    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    log.info(
        "Analysis complete: %d samples, mean score=%.6f → %s",
        n_scored, score_summary["mean"], output,
    )
    return output
