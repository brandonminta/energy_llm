"""Post-hoc beta sweep from cached raw dot-product scores.

beta_sweep_auroc  — recompute energy features from cached [L, T, K] score
                    tensors at multiple beta values and return best-layer AUROC
                    per beta.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.beta_sweep")


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable logsumexp along the given axis."""
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - xmax), axis=axis)) + xmax.squeeze(axis=axis)


def _energy_lse_only_from_scores(
    scores: np.ndarray,  # [..., K]
    beta: float,
) -> np.ndarray:
    """Compute -logsumexp(beta*s)/beta for each position. Shape [...]. """
    return -_logsumexp(beta * scores, axis=-1) / beta


def _entropy_from_scores(
    scores: np.ndarray,  # [..., K]
    beta: float,
) -> np.ndarray:
    """Shannon entropy of softmax(beta*s). Shape [...]."""
    scaled = beta * scores  # [..., K]
    lse    = _logsumexp(scaled, axis=-1)  # [...]
    log_p  = scaled - np.expand_dims(lse, axis=-1)  # [..., K]
    p      = np.exp(log_p)
    # -sum(p * log_p), clamp log_p to avoid -inf * 0
    log_p_safe = np.where(p > 0, log_p, 0.0)
    return -np.sum(p * log_p_safe, axis=-1)


def beta_sweep_auroc(
    samples_with_scores: list[dict],
    betas: list[float],
) -> pd.DataFrame:
    """Recompute energy/entropy at each beta from cached scores; report AUROC.

    Requires that each sample dict contains a 'gen_scores' key (float16 or
    float32 array of shape [L, T, K]) populated by load_sample_features when
    the trajectory was captured with save_scores=True.

    Args:
        samples_with_scores: Feature dicts including 'gen_scores'.
        betas:               List of beta values to sweep.

    Returns:
        DataFrame indexed by beta with columns:
            best_layer_energy,  best_auroc_energy,
            best_layer_entropy, best_auroc_entropy,
            per_layer_auroc_energy, per_layer_auroc_entropy  (list columns).

    Raises:
        ValueError: If 'gen_scores' is absent or None in any sample (with a
                    message mentioning 'save_scores').
    """
    if not samples_with_scores:
        raise ValueError("Empty sample list passed to beta_sweep_auroc.")

    # Validate scores presence
    if "gen_scores" not in samples_with_scores[0] or samples_with_scores[0]["gen_scores"] is None:
        raise ValueError(
            "Samples do not contain 'gen_scores'. "
            "Re-run trajectory capture with save_scores=True to enable the beta sweep. "
            "Then reload samples via load_sample_features."
        )

    labels = np.array([int(s["is_hallucination"]) for s in samples_with_scores], dtype=float)

    rows: list[dict] = []
    for beta in betas:
        energy_feats:  list[np.ndarray] = []
        entropy_feats: list[np.ndarray] = []

        for s in samples_with_scores:
            gen_scores = s["gen_scores"]
            if gen_scores is None:
                L = s["gen_energy_answer"].shape[0]
                energy_feats.append(np.full(L, np.nan, np.float32))
                entropy_feats.append(np.full(L, np.nan, np.float32))
                continue

            sc = gen_scores.astype(np.float32)  # [L, T, K]
            L, T, K = sc.shape

            # Per-layer per-token features, then mean over tokens (answer proxy)
            energy_lt  = _energy_lse_only_from_scores(sc, beta)   # [L, T]
            entropy_lt = _entropy_from_scores(sc, beta)            # [L, T]

            energy_feats.append(np.nanmean(energy_lt,  axis=1))    # [L]
            entropy_feats.append(np.nanmean(entropy_lt, axis=1))   # [L]

        L = energy_feats[0].shape[0]

        def _layer_aurocs(feat_list: list[np.ndarray]) -> list[float]:
            X = np.stack(feat_list, axis=0)  # [N, L]
            aurocs = []
            for l in range(L):
                col = X[:, l].copy()
                nan_m = np.isnan(col)
                if nan_m.any():
                    col[nan_m] = float(np.nanmedian(col))
                try:
                    aurocs.append(float(roc_auc_score(labels, col)))
                except ValueError:
                    aurocs.append(float("nan"))
            return aurocs

        energy_aurocs  = _layer_aurocs(energy_feats)
        entropy_aurocs = _layer_aurocs(entropy_feats)

        best_energy_l  = int(np.nanargmax(energy_aurocs))
        best_entropy_l = int(np.nanargmax(entropy_aurocs))

        rows.append({
            "beta":                    beta,
            "best_layer_energy":       best_energy_l,
            "best_auroc_energy":       energy_aurocs[best_energy_l],
            "best_layer_entropy":      best_entropy_l,
            "best_auroc_entropy":      entropy_aurocs[best_entropy_l],
            "per_layer_auroc_energy":  energy_aurocs,
            "per_layer_auroc_entropy": entropy_aurocs,
        })

        log.debug(
            "beta=%.1f  energy best: layer=%d AUROC=%.4f  "
            "entropy best: layer=%d AUROC=%.4f",
            beta, best_energy_l, energy_aurocs[best_energy_l],
            best_entropy_l, entropy_aurocs[best_entropy_l],
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("beta")
