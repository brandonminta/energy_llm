"""Hopfield energy, divergence, and scoring metrics."""

from hopfield_llm.metrics.divergences import (
    LayerDivergenceResult,
    compute_layer_divergences,
    compute_layer_energy_profile,
    hellinger,
    js_divergence,
    kl_divergence,
)
from hopfield_llm.metrics.hopfield import LayerEnergyResult, hopfield_energy_mlp
from hopfield_llm.metrics.scoring import hallucination_score

__all__ = [
    "LayerDivergenceResult",
    "LayerEnergyResult",
    "compute_layer_divergences",
    "compute_layer_energy_profile",
    "hallucination_score",
    "hellinger",
    "hopfield_energy_mlp",
    "js_divergence",
    "kl_divergence",
]
