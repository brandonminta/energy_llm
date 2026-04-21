"""Divergence primitives and per-sample retrieval-distribution comparison.

The core insight behind M2: divergences should be computed on the actual
retrieval distributions p_l^gen(t) = softmax(beta * scores_l^gen(t)) and
p_l^prompt_ref, which ARE proper probability distributions over the memory
index set {1, ..., K}.  The previous approach (softmax of the L-shaped energy
vector) was not a meaningful probability space.

KL, JS, Hellinger primitives accept either torch.Tensor (for compatibility
with existing callers) or are used internally by compute_sample_divergences
via numpy-based vectorised helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


# ------------------------------------------------------------------
# Divergence primitives (torch, 1-D vectors)
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
# Vectorised numpy helpers (internal, used by compute_sample_divergences)
# ------------------------------------------------------------------


def _kl_numpy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """KL(p||q) broadcasted over all axes except the last (distribution) axis."""
    p = np.clip(p.astype(np.float32), eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q.astype(np.float32), eps, None)
    q = q / q.sum(axis=-1, keepdims=True)
    return (p * np.log(p / q)).sum(axis=-1)


def _js_numpy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """JS divergence broadcasted over all axes except the last."""
    p = np.clip(p.astype(np.float32), eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q.astype(np.float32), eps, None)
    q = q / q.sum(axis=-1, keepdims=True)
    m = np.clip(0.5 * (p + q), eps, None)
    return (0.5 * (p * np.log(p / m)).sum(axis=-1) +
            0.5 * (q * np.log(q / m)).sum(axis=-1))


def _hellinger_numpy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Hellinger distance broadcasted over all axes except the last."""
    p = np.clip(p.astype(np.float32), eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q.astype(np.float32), eps, None)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=-1))


# ------------------------------------------------------------------
# Per-sample result container
# ------------------------------------------------------------------


@dataclass
class SampleDivergenceResult:
    """Divergence metrics for a single (prefill, generation) pair.

    Per-layer delta arrays  [L]
    ---------------------------
    delta_energy, delta_entropy, delta_norm_entropy,
    delta_top_activation, delta_mean_activation
        generation_mean[l] − prefill[l].  Genuine per-layer signals.

    Per-layer per-token divergence arrays  [L, T_gen]
    --------------------------------------------------
    kl_gen_to_prompt_per_layer
        KL(p_l^gen(t) || p_l^prompt_ref) per layer per generation token.
        NaN when save_distributions=False was used during trajectory capture.
    js_gen_to_prompt_per_layer, hellinger_gen_to_prompt_per_layer
        Same structure for JS and Hellinger.

    Interpretable scalar summaries
    ------------------------------
    energy_shift_l1   sum_l |delta_energy[l]|
    energy_shift_l2   sqrt(sum_l delta_energy[l]^2)
    peak_delta_layer  argmax_l |delta_energy[l]|
    gen_drift_mean    mean(kl_gen_to_prompt_per_layer)  — NaN if no distributions
    gen_drift_max     max(kl_gen_to_prompt_per_layer)   — NaN if no distributions
    """

    # Per-layer delta arrays [L]
    delta_energy:          np.ndarray
    delta_entropy:         np.ndarray
    delta_norm_entropy:    np.ndarray
    delta_top_activation:  np.ndarray
    delta_mean_activation: np.ndarray

    # Per-layer per-token divergences [L, T_gen] — NaN when distributions unavailable
    kl_gen_to_prompt_per_layer:         np.ndarray
    js_gen_to_prompt_per_layer:         np.ndarray
    hellinger_gen_to_prompt_per_layer:  np.ndarray

    # Interpretable scalars (always populated)
    energy_shift_l1:  float
    energy_shift_l2:  float
    peak_delta_layer: int
    gen_drift_mean:   float   # NaN when kl array is all-NaN
    gen_drift_max:    float   # NaN when kl array is all-NaN

    reference_label: str = "prefill"
    target_label:    str = "generation"


# ------------------------------------------------------------------
# compute_sample_divergences
# ------------------------------------------------------------------


def compute_sample_divergences(
    prompt_distributions: np.ndarray | None,
    prompt_energy: np.ndarray,
    gen_distributions: np.ndarray | None,
    gen_energy: np.ndarray,
    prompt_reduction: Literal["mean", "last"] = "mean",
    skip_first_gen_token: bool = True,
) -> SampleDivergenceResult:
    """Compute divergence metrics for one (prefill, generation) pair.

    The per-layer KL/JS/Hellinger arrays compare the *retrieval distribution*
    (softmax over memory index K) at generation time against a prompt reference —
    a semantically meaningful divergence, unlike the old softmax-energy approach.

    Args:
        prompt_distributions:  float16 or float32 array [L, T_prompt, K], or None.
                               Produced by capture_prefill(save_distributions=True).
        prompt_energy:         float32 array [L] — per-layer prefill energy.
        gen_distributions:     float16 or float32 array [L, T_gen, K], or None.
                               Produced by capture_generation(save_distributions=True).
        gen_energy:            float32 array [L, T_gen] — per-layer per-token energies.
        prompt_reduction:      How to build the per-layer prompt reference distribution.
                               'mean' — mean over T_prompt tokens then renormalise.
                               'last' — use the last prompt token's distribution.
        skip_first_gen_token:  Drop t=0 of the generation side before computing
                               divergences (§1.5 fix: the prefill last-token and
                               gen t=0 are the same MLP activation).

    Returns:
        SampleDivergenceResult.  Per-layer KL/JS/Hellinger arrays are NaN when
        distributions are not provided; delta_energy scalars are always finite.
    """
    L = len(prompt_energy)
    T_gen = gen_energy.shape[1] if gen_energy.ndim == 2 else 1

    # ── Per-layer delta metrics from energy arrays ──────────────────────────
    gen_energy_mean = np.nanmean(gen_energy, axis=1)  # [L]
    delta_energy    = gen_energy_mean - prompt_energy

    # ── Energy scalar summaries ─────────────────────────────────────────────
    de_abs = np.abs(np.where(np.isnan(delta_energy), 0.0, delta_energy))
    energy_shift_l1  = float(np.sum(de_abs))
    energy_shift_l2  = float(np.sqrt(np.sum(de_abs ** 2)))
    peak_delta_layer = int(np.argmax(de_abs))

    # ── Placeholder NaN fields for non-energy deltas ────────────────────────
    nan_L = np.full(L, np.nan, dtype=np.float64)

    # ── Per-layer per-token KL/JS/Hellinger ─────────────────────────────────
    if prompt_distributions is None or gen_distributions is None:
        nan_LT = np.full((L, T_gen), np.nan, dtype=np.float32)
        return SampleDivergenceResult(
            delta_energy=delta_energy.astype(np.float64),
            delta_entropy=nan_L.copy(),
            delta_norm_entropy=nan_L.copy(),
            delta_top_activation=nan_L.copy(),
            delta_mean_activation=nan_L.copy(),
            kl_gen_to_prompt_per_layer=nan_LT,
            js_gen_to_prompt_per_layer=nan_LT.copy(),
            hellinger_gen_to_prompt_per_layer=nan_LT.copy(),
            energy_shift_l1=energy_shift_l1,
            energy_shift_l2=energy_shift_l2,
            peak_delta_layer=peak_delta_layer,
            gen_drift_mean=float(np.nan),
            gen_drift_max=float(np.nan),
        )

    # Build prompt reference distribution per layer: [L, K]
    p_dists = prompt_distributions.astype(np.float32)   # [L, T_prompt, K]
    if prompt_reduction == "mean":
        p_ref = p_dists.mean(axis=1)                    # [L, K]
    else:  # "last"
        p_ref = p_dists[:, -1, :]                       # [L, K]
    eps = 1e-10
    p_ref = np.clip(p_ref, eps, None)
    p_ref = p_ref / p_ref.sum(axis=-1, keepdims=True)   # renormalise

    # Generation distributions: optionally skip first token
    g_dists = gen_distributions.astype(np.float32)      # [L, T_gen, K]
    if skip_first_gen_token and g_dists.shape[1] > 1:
        g_dists = g_dists[:, 1:, :]                     # [L, T_eff, K]

    T_eff = g_dists.shape[1]
    p_ref_exp = p_ref[:, None, :]                        # [L, 1, K] → broadcasts to [L, T_eff, K]

    # KL(gen(t) || prompt_ref) per layer per token
    kl_arr   = _kl_numpy(g_dists, p_ref_exp)            # [L, T_eff]
    js_arr   = _js_numpy(g_dists, p_ref_exp)            # [L, T_eff]
    hell_arr = _hellinger_numpy(g_dists, p_ref_exp)     # [L, T_eff]

    gen_drift_mean = float(np.nanmean(kl_arr))
    gen_drift_max  = float(np.nanmax(kl_arr))

    return SampleDivergenceResult(
        delta_energy=delta_energy.astype(np.float64),
        delta_entropy=nan_L.copy(),
        delta_norm_entropy=nan_L.copy(),
        delta_top_activation=nan_L.copy(),
        delta_mean_activation=nan_L.copy(),
        kl_gen_to_prompt_per_layer=kl_arr.astype(np.float32),
        js_gen_to_prompt_per_layer=js_arr.astype(np.float32),
        hellinger_gen_to_prompt_per_layer=hell_arr.astype(np.float32),
        energy_shift_l1=energy_shift_l1,
        energy_shift_l2=energy_shift_l2,
        peak_delta_layer=peak_delta_layer,
        gen_drift_mean=gen_drift_mean,
        gen_drift_max=gen_drift_max,
    )
