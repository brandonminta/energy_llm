"""Divergence primitives for comparing retrieval distributions.

KL, JS, and Hellinger divergences between 1-D probability distributions.
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

    p = p.clamp(min=eps)
    p = p / p.sum()
    q = q.clamp(min=eps)
    q = q / q.sum()

    kl_fwd = float((p * (p / q).log()).sum())
    if direction == "forward":
        return kl_fwd
    kl_rev = float((q * (q / p).log()).sum())
    if direction == "reverse":
        return kl_rev
    return kl_fwd, kl_rev


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two 1-D distributions."""
    p = p.clamp(min=eps)
    p = p / p.sum()
    q = q.clamp(min=eps)
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def hellinger(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Hellinger distance between two 1-D distributions."""
    p = p.clamp(min=eps)
    p = p / p.sum()
    q = q.clamp(min=eps)
    q = q / q.sum()
    return float((0.5 * (p.sqrt() - q.sqrt()).pow(2).sum()).sqrt())


# ------------------------------------------------------------------
# Per-layer result container (retained for downstream analysis)
# ------------------------------------------------------------------


@dataclass
class LayerDivergenceResult:
    kl_fwd: np.ndarray                # [L]
    kl_rev: np.ndarray                # [L]
    js: np.ndarray                    # [L]
    hellinger: np.ndarray             # [L]
    delta_entropy: np.ndarray         # [L]
    delta_norm_entropy: np.ndarray    # [L]
    delta_energy: np.ndarray          # [L]
    delta_top_activation: np.ndarray  # [L]
    delta_mean_activation: np.ndarray # [L]
    reference_label: str = "ref"
    target_label: str = "cmp"
