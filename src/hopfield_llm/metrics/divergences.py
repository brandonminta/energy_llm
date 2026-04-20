"""Divergence primitives for comparing retrieval distributions.

KL, JS, and Hellinger divergences between 1-D probability distributions.

Design note on SampleDivergenceResult
--------------------------------------
The *global* divergences (kl_fwd, kl_rev, js, hellinger) compare the
layer-energy distributions of prefill vs. generation as a whole.  They are
scalars per sample — not per-layer arrays — because their probability space IS
the layer axis (softmax across L layers).  Treating them as per-layer values
would be numerically incorrect.

The *per-layer* delta arrays (delta_energy, delta_entropy, …) are genuine
layer-resolved signals: generation_mean[l] − prefill[l].  These are what
should be plotted per-layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


# ------------------------------------------------------------------
# Divergence primitives
# ------------------------------------------------------------------


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10,
    direction: Literal["forward", "reverse", "both"] = "forward",
) -> float | tuple[float, float]:
    """KL divergence between two 1-D probability distributions."""
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
    if p.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got {p.ndim}D")

    p = p.clamp(min=eps); p = p / p.sum()
    q = q.clamp(min=eps); q = q / q.sum()

    kl_fwd = float((p * (p / q).log()).sum())
    if direction == "forward":
        return kl_fwd
    kl_rev = float((q * (q / p).log()).sum())
    if direction == "reverse":
        return kl_rev
    return kl_fwd, kl_rev


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two 1-D distributions."""
    p = p.clamp(min=eps); p = p / p.sum()
    q = q.clamp(min=eps); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())


def hellinger(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Hellinger distance between two 1-D distributions."""
    p = p.clamp(min=eps); p = p / p.sum()
    q = q.clamp(min=eps); q = q / q.sum()
    return float((0.5 * (p.sqrt() - q.sqrt()).pow(2).sum()).sqrt())


# ------------------------------------------------------------------
# Per-sample result container
# ------------------------------------------------------------------


@dataclass
class SampleDivergenceResult:
    """Divergence metrics for a single (prefill, generation) pair.

    Global scalars
    --------------
    kl_fwd, kl_rev, js, hellinger
        Compare the layer-energy distributions of prefill vs. generation.
        Probability space = layer axis (softmax across L layers).
        One scalar per sample; not meaningful as per-layer values.

    Per-layer arrays  [L]
    ---------------------
    delta_energy, delta_entropy, delta_norm_entropy,
    delta_top_activation, delta_mean_activation
        generation_mean[l] − prefill[l].  These are the genuine per-layer
        signals to plot and aggregate across the dataset.
    """

    # Global (scalar) divergences
    kl_fwd:   float
    kl_rev:   float
    js:       float
    hellinger: float

    # Per-layer deltas [L]
    delta_energy:          np.ndarray
    delta_entropy:         np.ndarray
    delta_norm_entropy:    np.ndarray
    delta_top_activation:  np.ndarray
    delta_mean_activation: np.ndarray

    reference_label: str = "prefill"
    target_label:    str = "generation"
