"""Hallucination score aggregation from per-sample divergence results."""

from __future__ import annotations

from typing import Literal

import numpy as np

from hopfield_llm.metrics.divergences import SampleDivergenceResult


# Per-layer metrics that can be aggregated across the layer axis
LAYER_METRICS = (
    "delta_energy",
    "delta_norm_entropy",
    "delta_entropy",
    "delta_top_activation",
    "delta_mean_activation",
)

# Global scalar metrics (one number per sample, not per layer)
GLOBAL_METRICS = ("kl_fwd", "kl_rev", "js", "hellinger")

ALL_METRICS = LAYER_METRICS + GLOBAL_METRICS


def hallucination_score(
    div: SampleDivergenceResult,
    signal_zone: tuple[int, int] | None = None,
    metric: str = "js",
    aggregation: Literal["mean", "max", "weighted"] = "mean",
    use_absolute: bool = False,
) -> float:
    """Aggregate divergence metrics into a scalar hallucination score.

    For per-layer metrics (delta_*), aggregation is applied across the layer
    axis (optionally restricted to signal_zone).  For global metrics
    (kl_fwd, kl_rev, js, hellinger) the value is returned directly —
    signal_zone and aggregation are ignored since they are already scalars.

    Args:
        div:          SampleDivergenceResult for one sample.
        signal_zone:  Optional (start, end) layer slice for per-layer metrics.
        metric:       Which metric to use (see ALL_METRICS).
        aggregation:  Aggregation for per-layer metrics: mean / max / weighted.
        use_absolute: Apply abs() before aggregating (per-layer only).

    Returns:
        Scalar hallucination score.
    """
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric: '{metric}'. Valid: {ALL_METRICS}")

    if metric in GLOBAL_METRICS:
        return float(getattr(div, metric))

    data: np.ndarray = getattr(div, metric)
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
    if aggregation == "weighted":
        mask = ~np.isnan(data)
        if not mask.any():
            raise ValueError("All values are NaN in hallucination_score (weighted)")
        weights = np.linspace(0.5, 1.0, len(data))
        return float(np.average(data[mask], weights=weights[mask]))

    raise ValueError(f"Unknown aggregation: '{aggregation}'")
