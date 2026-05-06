"""Divergence primitives and per-sample retrieval-distribution comparison.

The core insight behind M2: divergences should be computed on the actual
retrieval distributions p_l^gen(t) = softmax(beta * scores_l^gen(t)) and
p_l^prompt_ref, which ARE proper probability distributions over the memory
index set {1, ..., K}.  The previous approach (softmax of the L-shaped energy
vector) was not a meaningful probability space.

KL, JS, Hellinger torch primitives (1-D) are kept for the inline capture
path in hooks/capture.py.  The numpy-vectorised helpers that backed the old
save_distributions path have been removed along with the legacy per-token
distribution fields.
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
# Per-sample result container
# ------------------------------------------------------------------


@dataclass
class SampleDivergenceResult:
    """Divergence metrics for a single (prefill, generation) pair.

    Per-layer delta arrays  [L]
    ---------------------------
    delta_energy, delta_entropy, delta_norm_entropy,
    delta_top_activation, delta_lse
        generation_mean[l] − prefill[l].  Genuine per-layer signals.

    Per-layer temporal arrays  [L]
    ------------------------------
    energy_slope        ordinary-least-squares slope of gen_energy[l, :] over t.
                        Positive → energy rises during generation; negative → falls.
                        Tells a *direction-of-drift* story that mean-pooling hides.
    energy_var          Variance of gen_energy[l, :] across tokens.  Captures
                        token-to-token jitter that mean-pooling washes out.
    energy_spread       max - min of gen_energy[l, :] across tokens.  More
                        robust than variance for short generations.
    top_act_gap_mean    Mean over t of (top1 − top2) of gen_scores[l, t, :].
                        NaN when scores were not saved.  Smaller gap → more
                        confused retrieval at that layer/token.

    Interpretable scalar summaries
    ------------------------------
    energy_shift_l1   sum_l |delta_energy[l]|
    energy_shift_l2   sqrt(sum_l delta_energy[l]^2)
    peak_delta_layer  argmax_l |delta_energy[l]|
    """

    # Per-layer delta arrays [L]
    delta_energy:          np.ndarray
    delta_entropy:         np.ndarray
    delta_norm_entropy:    np.ndarray
    delta_top_activation:  np.ndarray
    delta_lse: np.ndarray

    # Interpretable scalars (always populated)
    energy_shift_l1:  float
    energy_shift_l2:  float
    peak_delta_layer: int

    # Per-layer temporal metrics over the generation axis [L].  Each tells a
    # different story than the mean-pooled deltas above.  All optional; NaN
    # when the gen_energy array has fewer than 2 finite tokens.
    energy_slope:    np.ndarray | None = None     # [L]  OLS slope per layer
    energy_var:      np.ndarray | None = None     # [L]  variance over tokens
    energy_spread:   np.ndarray | None = None     # [L]  max - min over tokens
    top_act_gap_mean: np.ndarray | None = None    # [L]  NaN unless scores saved

    # Logit-lens metrics (independent of bank).  Populated when
    # enable_logit_lens=True at capture time.  All NaN otherwise.
    delta_logit_entropy:    np.ndarray | None = None   # [L]  gen_mean - prefill (entropy)
    gen_logit_entropy_mean: np.ndarray | None = None   # [L]  mean over T_gen of entropy
    gen_logit_top_prob_min: np.ndarray | None = None   # [L]  min over T_gen of top-1 prob

    # Shuffled-bank null control: same delta_energy computed against a
    # randomly-permuted bank.  If real and null deltas are similar, the bank
    # identity is irrelevant for the metric.
    delta_energy_null: np.ndarray | None = None        # [L]

    # Inline JS and Hellinger divergences (gen vs prefill reference), computed
    # during trajectory collection without saving full distributions to disk.
    js_mean_per_layer:        np.ndarray | None = None   # [L]
    js_max_per_layer:         np.ndarray | None = None   # [L]
    hellinger_mean_per_layer: np.ndarray | None = None   # [L]
    hellinger_max_per_layer:  np.ndarray | None = None   # [L]

    reference_label: str = "prefill"
    target_label:    str = "generation"


# ------------------------------------------------------------------
# Temporal energy metrics (per layer over generation tokens)
# ------------------------------------------------------------------


def _temporal_energy_metrics(
    gen_energy: np.ndarray,  # [L, T_gen]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS slope, variance, and max-min spread of gen_energy along the T axis.

    Returns three [L] float64 arrays.  Layers where fewer than 2 finite
    values exist along T are filled with NaN to avoid spurious zeros.
    """
    if gen_energy.ndim != 2:
        L = gen_energy.shape[0] if gen_energy.ndim == 1 else 0
        nan_L = np.full(L, np.nan, dtype=np.float64)
        return nan_L.copy(), nan_L.copy(), nan_L.copy()

    L, T = gen_energy.shape
    slope  = np.full(L, np.nan, dtype=np.float64)
    var    = np.full(L, np.nan, dtype=np.float64)
    spread = np.full(L, np.nan, dtype=np.float64)

    if T == 0:
        return slope, var, spread

    t_axis = np.arange(T, dtype=np.float64)
    for l in range(L):
        row = gen_energy[l].astype(np.float64)
        finite = np.isfinite(row)
        n_finite = int(finite.sum())
        if n_finite >= 2:
            x = t_axis[finite]
            y = row[finite]
            x_centered = x - x.mean()
            denom = float((x_centered ** 2).sum())
            slope[l]  = float((x_centered * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
            var[l]    = float(np.var(y))
            spread[l] = float(y.max() - y.min())
        elif n_finite == 1:
            var[l]    = 0.0
            spread[l] = 0.0
            # slope undefined with one point — leave NaN
    return slope, var, spread


# ------------------------------------------------------------------
# compute_sample_divergences
# ------------------------------------------------------------------


def compute_sample_divergences(
    prompt_energy: np.ndarray,
    gen_energy: np.ndarray,
    prompt_entropy: np.ndarray | None = None,
    prompt_norm_entropy: np.ndarray | None = None,
    prompt_top_act: np.ndarray | None = None,
    prompt_lse: np.ndarray | None = None,
    gen_entropy: np.ndarray | None = None,
    gen_norm_entropy: np.ndarray | None = None,
    gen_top_act: np.ndarray | None = None,
    gen_lse: np.ndarray | None = None,
    gen_js: np.ndarray | None = None,
    gen_hellinger: np.ndarray | None = None,
) -> SampleDivergenceResult:
    """Compute divergence metrics for one (prefill, generation) pair.

    Args:
        prompt_energy:    float32 array [L] — per-layer prefill energy.
        gen_energy:       float32 array [L, T_gen] — per-layer per-token energies.
        prompt_*, gen_*:  Optional per-layer arrays.  When supplied the
                          corresponding delta_* fields are populated; otherwise NaN.
        gen_js:           float32 [L, T_gen] inline JS divergences from capture.
        gen_hellinger:    float32 [L, T_gen] inline Hellinger divergences from capture.

    Returns:
        SampleDivergenceResult with always-finite energy scalars and optional
        per-layer fields populated when input arrays are provided.
    """
    L = len(prompt_energy)
    T_gen = gen_energy.shape[1] if gen_energy.ndim == 2 else 1

    # ── Per-layer delta metrics from energy arrays ──────────────────────────
    gen_energy_mean = np.nanmean(gen_energy, axis=1)  # [L]
    delta_energy    = gen_energy_mean - prompt_energy

    def _delta(p: np.ndarray | None, g: np.ndarray | None) -> np.ndarray:
        if p is None or g is None:
            return np.full(L, np.nan, dtype=np.float64)
        return (np.nanmean(g, axis=1) - p).astype(np.float64)

    delta_entropy         = _delta(prompt_entropy,      gen_entropy)
    delta_norm_entropy    = _delta(prompt_norm_entropy, gen_norm_entropy)
    delta_top_activation  = _delta(prompt_top_act,      gen_top_act)
    delta_lse = _delta(prompt_lse,          gen_lse)

    # ── Per-layer temporal metrics over the generation axis ────────────────
    energy_slope, energy_var, energy_spread = _temporal_energy_metrics(gen_energy)

    # ── Energy scalar summaries ─────────────────────────────────────────────
    de_abs = np.abs(np.where(np.isnan(delta_energy), 0.0, delta_energy))
    energy_shift_l1  = float(np.sum(de_abs))
    energy_shift_l2  = float(np.sqrt(np.sum(de_abs ** 2)))
    peak_delta_layer = int(np.argmax(de_abs))

    # ── Inline JS / Hellinger from gen trajectory arrays ────────────────────
    js_mean = js_max = hellinger_mean = hellinger_max = None
    if gen_js is not None:
        js_mean = np.nanmean(gen_js.astype(np.float64), axis=1)   # [L]
        js_max  = np.nanmax(gen_js.astype(np.float64), axis=1)    # [L]
    if gen_hellinger is not None:
        hellinger_mean = np.nanmean(gen_hellinger.astype(np.float64), axis=1)  # [L]
        hellinger_max  = np.nanmax(gen_hellinger.astype(np.float64), axis=1)   # [L]

    return SampleDivergenceResult(
        delta_energy=delta_energy.astype(np.float64),
        delta_entropy=delta_entropy,
        delta_norm_entropy=delta_norm_entropy,
        delta_top_activation=delta_top_activation,
        delta_lse=delta_lse,
        energy_shift_l1=energy_shift_l1,
        energy_shift_l2=energy_shift_l2,
        peak_delta_layer=peak_delta_layer,
        energy_slope=energy_slope,
        energy_var=energy_var,
        energy_spread=energy_spread,
        top_act_gap_mean=None,
        js_mean_per_layer=js_mean,
        js_max_per_layer=js_max,
        hellinger_mean_per_layer=hellinger_mean,
        hellinger_max_per_layer=hellinger_max,
    )
