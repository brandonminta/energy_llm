from __future__ import annotations

from typing import Literal

import numpy as np

from energy_llm.metrics.divergences import LayerDivergenceResult


def hallucination_score(
    div: LayerDivergenceResult,
    signal_zone: tuple[int, int] | None = None,
    metric: Literal[
        "kl_fwd", "kl_rev", "js", "hellinger",
        "delta_entropy", "delta_norm_entropy", "delta_energy",
        "delta_top_activation", "delta_mean_activation",
    ] = "js",
    aggregation: Literal["mean", "max", "weighted"] = "mean",
    use_absolute: bool = False,
) -> float:
    """
    Agrega LayerDivergenceResult -> escalar.

    Args:
        use_absolute: si True, aplica abs() antes de agregar.
                      util para metricas delta cuando importa
                      magnitud del cambio, no direccion.
    """
    data = {
        "kl_fwd": div.kl_fwd,
        "kl_rev": div.kl_rev,
        "js": div.js,
        "hellinger": div.hellinger,
        "delta_entropy": div.delta_entropy,
        "delta_norm_entropy": div.delta_norm_entropy,
        "delta_energy": div.delta_energy,
        "delta_top_activation": div.delta_top_activation,
        "delta_mean_activation": div.delta_mean_activation,
    }[metric]

    if signal_zone is not None:
        start, end = signal_zone
        data = data[start:end]

    if len(data) == 0:
        raise ValueError("No hay datos para agregar en hallucination_score")

    if use_absolute:
        data = np.abs(data)

    if aggregation == "mean":
        return float(np.nanmean(data))

    if aggregation == "max":
        return float(np.nanmax(data))

    if aggregation == "weighted":
        mask = ~np.isnan(data)
        if not mask.any():
            raise ValueError("Todos los valores son NaN en hallucination_score (weighted)")
        weights = np.linspace(0.5, 1.0, len(data))
        return float(np.average(data[mask], weights=weights[mask]))

    raise ValueError(f"aggregation desconocida: '{aggregation}'")
