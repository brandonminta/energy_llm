"""Hallucination score aggregation over layer-wise divergence metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np

from hopfield_llm.metrics.divergences import LayerDivergenceResult


METRIC_NAMES = (
    "kl_fwd", "kl_rev", "js", "hellinger",
    "delta_entropy", "delta_norm_entropy", "delta_energy",
    "delta_top_activation", "delta_mean_activation",
)


def hallucination_score(
    div: LayerDivergenceResult,
    signal_zone: tuple[int, int] | None = None,
    metric: str = "js",
    aggregation: Literal["mean", "max", "weighted"] = "mean",
    use_absolute: bool = False,
) -> float:
    """Aggregate a per-layer metric array into a scalar hallucination score.

    Args:
        div: LayerDivergenceResult with per-layer arrays.
        signal_zone: Optional (start, end) layer slice.
        metric: Which metric to aggregate (see METRIC_NAMES).
        aggregation: Aggregation method.
        use_absolute: Take absolute value before aggregating.

    Returns:
        Scalar hallucination score.
    """
    lookup = {
        "kl_fwd": div.kl_fwd,
        "kl_rev": div.kl_rev,
        "js": div.js,
        "hellinger": div.hellinger,
        "delta_entropy": div.delta_entropy,
        "delta_norm_entropy": div.delta_norm_entropy,
        "delta_energy": div.delta_energy,
        "delta_top_activation": div.delta_top_activation,
        "delta_mean_activation": div.delta_mean_activation,
    }
    if metric not in lookup:
        raise ValueError(f"Unknown metric: '{metric}'. Use one of {METRIC_NAMES}")

    data = lookup[metric]
    if signal_zone is not None:
        start, end = signal_zone
        data = data[start:end]
    if len(data) == 0:
        raise ValueError("No data to aggregate in hallucination_score")
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
