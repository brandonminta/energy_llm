"""Divergence primitives and layer-wise divergence computation.

Merges the low-level divergence functions (KL, JS, Hellinger) with the
layer-level scoring logic that compares two hidden states against the same
memory bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from hopfield_llm.metrics.hopfield import hopfield_energy_mlp


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
# Layer-level divergence
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


@torch.no_grad()
def compute_layer_divergences(
    h_ref: torch.Tensor,
    h_cmp: torch.Tensor,
    mlp_keys: dict[int, torch.Tensor],
    beta: float = 15.0,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    active_threshold: float = 0.1,
    reference_label: str = "ref",
    target_label: str = "cmp",
) -> LayerDivergenceResult:
    """Compare two hidden-state vectors layer by layer against the same bank.

    Args:
        h_ref: Reference hidden states [L, D] (e.g. factual answer).
        h_cmp: Comparison hidden states [L, D] (e.g. hallucinated answer).
        mlp_keys: Position-indexed memory banks (use ``align_banks_to_layers``).
        beta: Inverse temperature for Hopfield energy.
        energy_mode: 'cosine' or 'dot'.
        active_threshold: Activation threshold for Hopfield.
        reference_label: Label for the reference states.
        target_label: Label for the comparison states.

    Returns:
        LayerDivergenceResult with per-layer metric arrays of length L.
    """
    if h_ref.ndim != 2 or h_cmp.ndim != 2:
        raise ValueError(
            f"h_ref and h_cmp must be [L, D], got shapes: {h_ref.shape}, {h_cmp.shape}"
        )
    if h_ref.shape != h_cmp.shape:
        raise ValueError(
            f"h_ref and h_cmp must have the same shape: {h_ref.shape} vs {h_cmp.shape}"
        )

    L = h_ref.shape[0]
    kl_fwd_arr = np.full(L, np.nan, dtype=np.float32)
    kl_rev_arr = np.full(L, np.nan, dtype=np.float32)
    js_arr = np.full(L, np.nan, dtype=np.float32)
    hell_arr = np.full(L, np.nan, dtype=np.float32)
    delta_entropy = np.full(L, np.nan, dtype=np.float32)
    delta_norm_entropy = np.full(L, np.nan, dtype=np.float32)
    delta_energy = np.full(L, np.nan, dtype=np.float32)
    delta_top_act = np.full(L, np.nan, dtype=np.float32)
    delta_mean_act = np.full(L, np.nan, dtype=np.float32)

    for i in range(L):
        if i not in mlp_keys:
            continue
        W = mlp_keys[i].float()
        r_ref = hopfield_energy_mlp(
            h_ref[i].float(), W,
            beta=beta, active_threshold=active_threshold,
            energy_mode=energy_mode, return_similarities=False,
        )
        r_cmp = hopfield_energy_mlp(
            h_cmp[i].float(), W,
            beta=beta, active_threshold=active_threshold,
            energy_mode=energy_mode, return_similarities=False,
        )

        p_ref, p_cmp = r_ref.softmax_w, r_cmp.softmax_w
        kl_fwd_arr[i], kl_rev_arr[i] = kl_divergence(p_ref, p_cmp, direction="both")
        js_arr[i] = js_divergence(p_ref, p_cmp)
        hell_arr[i] = hellinger(p_ref, p_cmp)
        delta_entropy[i] = r_cmp.retrieval_entropy - r_ref.retrieval_entropy
        delta_norm_entropy[i] = r_cmp.normalized_entropy - r_ref.normalized_entropy
        delta_energy[i] = r_cmp.energy - r_ref.energy
        delta_top_act[i] = r_cmp.top_activation - r_ref.top_activation
        delta_mean_act[i] = r_cmp.mean_activation - r_ref.mean_activation

    return LayerDivergenceResult(
        kl_fwd=kl_fwd_arr,
        kl_rev=kl_rev_arr,
        js=js_arr,
        hellinger=hell_arr,
        delta_entropy=delta_entropy,
        delta_norm_entropy=delta_norm_entropy,
        delta_energy=delta_energy,
        delta_top_activation=delta_top_act,
        delta_mean_activation=delta_mean_act,
        reference_label=reference_label,
        target_label=target_label,
    )


@torch.no_grad()
def compute_layer_energy_profile(
    h_state: torch.Tensor,
    mlp_keys: dict[int, torch.Tensor],
    beta: float = 15.0,
    energy_mode: Literal["cosine", "dot"] = "cosine",
    active_threshold: float = 0.1,
) -> dict[str, np.ndarray]:
    """Compute per-layer energy profile for a single hidden-state vector.

    Returns:
        Dict of metric name -> ndarray of shape [L].
    """
    if h_state.ndim != 2:
        raise ValueError(f"h_state must be [L, D], got {h_state.shape}")

    L = h_state.shape[0]
    metrics = {
        "energy": np.full(L, np.nan, dtype=np.float32),
        "retrieval_entropy": np.full(L, np.nan, dtype=np.float32),
        "normalized_entropy": np.full(L, np.nan, dtype=np.float32),
        "top_activation": np.full(L, np.nan, dtype=np.float32),
        "mean_activation": np.full(L, np.nan, dtype=np.float32),
        "n_active_memories": np.full(L, np.nan, dtype=np.float32),
        "lse": np.full(L, np.nan, dtype=np.float32),
    }

    for i in range(L):
        if i not in mlp_keys:
            continue
        r = hopfield_energy_mlp(
            h_state[i].float(), mlp_keys[i].float(),
            beta=beta, active_threshold=active_threshold,
            energy_mode=energy_mode, return_similarities=False,
        )
        metrics["energy"][i] = r.energy
        metrics["retrieval_entropy"][i] = r.retrieval_entropy
        metrics["normalized_entropy"][i] = r.normalized_entropy
        metrics["top_activation"][i] = r.top_activation
        metrics["mean_activation"][i] = r.mean_activation
        metrics["n_active_memories"][i] = r.n_active_memories
        metrics["lse"][i] = r.lse

    return metrics
