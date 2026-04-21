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
    "gen_drift_mean",
    "gen_drift_max",
)

# Per-layer array metrics — shape [L] per sample
PER_LAYER_METRICS = (
    "delta_energy",
    "delta_norm_entropy",
    "delta_entropy",
    "delta_top_activation",
    "delta_mean_activation",
)

# Per-layer per-token array metrics — shape [L, T_gen] per sample
PER_LAYER_PER_TOKEN_METRICS = (
    "kl_gen_to_prompt_per_layer",
    "js_gen_to_prompt_per_layer",
    "hellinger_gen_to_prompt_per_layer",
)

# Kept for external callers that iterate over "layer metrics"
LAYER_METRICS = PER_LAYER_METRICS + PER_LAYER_PER_TOKEN_METRICS

ALL_METRICS = SCALAR_METRICS + LAYER_METRICS


def hallucination_score(
    div: SampleDivergenceResult,
    signal_zone: tuple[int, int] | None = None,
    metric: str = "delta_energy",
    aggregation: Literal["mean", "max", "weighted"] = "mean",
    use_absolute: bool = False,
) -> float:
    """Aggregate divergence metrics into a scalar hallucination score.

    Metric types and how they are reduced to a scalar:

    SCALAR_METRICS (energy_shift_l1, gen_drift_mean, …):
        Returned directly — signal_zone and aggregation are ignored.

    PER_LAYER_METRICS (delta_energy, delta_entropy, …):
        signal_zone slices the [L] array; aggregation is applied over layers.

    PER_LAYER_PER_TOKEN_METRICS (kl_gen_to_prompt_per_layer, …):
        The [L, T_gen] array is mean-pooled over the token axis first,
        yielding [L]; then signal_zone + aggregation are applied.
        Returns NaN if all values are NaN (e.g. distributions not saved).

    Args:
        div:          SampleDivergenceResult for one sample.
        signal_zone:  Optional (start, end) layer slice (PER_LAYER_* only).
        metric:       Which metric to use (see ALL_METRICS).
        aggregation:  Aggregation for array metrics: mean / max / weighted.
        use_absolute: Apply abs() before aggregating (array metrics only).

    Returns:
        Scalar hallucination score.
    """
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric: '{metric}'. Valid: {ALL_METRICS}")

    # Scalar metrics: return directly
    if metric in SCALAR_METRICS:
        return float(getattr(div, metric))

    # Per-layer per-token: reduce over token axis first → [L]
    if metric in PER_LAYER_PER_TOKEN_METRICS:
        raw: np.ndarray = getattr(div, metric)   # [L, T_gen]
        data = np.nanmean(raw, axis=1)           # [L]
    else:
        data = getattr(div, metric)              # [L]

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
