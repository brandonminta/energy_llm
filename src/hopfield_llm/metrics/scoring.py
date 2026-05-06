"""Hallucination score aggregation from per-sample divergence results."""

from __future__ import annotations

from typing import Literal

import numpy as np

from hopfield_llm.metrics.divergences import SampleDivergenceResult


# Scalar metrics — already reduced to one number per sample
SCALAR_METRICS = (
    "energy_shift_l1",
    "energy_shift_l2",
    "peak_delta_layer",
)

# Per-layer array metrics — shape [L] per sample
PER_LAYER_METRICS = (
    "delta_energy",
    "delta_norm_entropy",
    "delta_entropy",
    "delta_top_activation",
    "delta_lse",
    # Temporal metrics over the generation token axis
    "energy_slope",
    "energy_var",
    "energy_spread",
    # Retrieval-confidence metric (requires saved scores)
    "top_act_gap_mean",
    # Logit-lens metrics (require enable_logit_lens=True at capture)
    "delta_logit_entropy",
    "gen_logit_entropy_mean",
    "gen_logit_top_prob_min",
    # Shuffled-bank null control (require enable_null_control=True at capture)
    "delta_energy_null",
    # Inline JS and Hellinger divergences (gen vs prefill reference)
    "js_mean_per_layer",
    "js_max_per_layer",
    "hellinger_mean_per_layer",
    "hellinger_max_per_layer",
)

LAYER_METRICS = PER_LAYER_METRICS

ALL_METRICS = SCALAR_METRICS + LAYER_METRICS


def hallucination_score(
    div: SampleDivergenceResult,
    signal_zone: tuple[int, int] | None = None,
    metric: str = "delta_energy",
    aggregation: Literal["mean", "max"] = "mean",
    use_absolute: bool = False,
) -> float:
    """Aggregate divergence metrics into a scalar hallucination score.

    Metric types and how they are reduced to a scalar:

    SCALAR_METRICS (energy_shift_l1, peak_delta_layer, …):
        Returned directly — signal_zone and aggregation are ignored.

    PER_LAYER_METRICS (delta_energy, delta_entropy, js_mean_per_layer, …):
        signal_zone slices the [L] array; aggregation is applied over layers.

    Args:
        div:          SampleDivergenceResult for one sample.
        signal_zone:  Optional (start, end) layer slice (PER_LAYER_* only).
        metric:       Which metric to use (see ALL_METRICS).
        aggregation:  Aggregation for array metrics: mean / max.
        use_absolute: Apply abs() before aggregating (array metrics only).

    Returns:
        Scalar hallucination score.
    """
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric: '{metric}'. Valid: {ALL_METRICS}")

    # Scalar metrics: return directly
    if metric in SCALAR_METRICS:
        return float(getattr(div, metric))

    data = getattr(div, metric)              # [L] or None (optional metric)
    if data is None:
        return float("nan")

    if signal_zone is not None:
        start, end = signal_zone
        data = data[start:end]
    if len(data) == 0:
        raise ValueError(f"Empty data after applying signal_zone to '{metric}'")
    if use_absolute:
        data = np.abs(data)

    if aggregation == "mean":
        return float(np.nanmean(data))
    if aggregation == "max":
        return float(np.nanmax(data))

    raise ValueError(f"Unknown aggregation: '{aggregation}'. Use 'mean' or 'max'.")
