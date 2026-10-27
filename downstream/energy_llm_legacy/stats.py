"""Statistical procedures (sec:analysis_method).

- Discrimination: test AUROC with a nonparametric bootstrap, B = 10000, 95%
  CI; resamples containing a single class are redrawn
  (subsec:an_discrimination).
- Contribution: paired bootstrap of delta AUROC (full - base) on identical
  resamples (subsec:an_contribution).
- Localisation: per-layer folded statistic max(AUROC, 1-AUROC) with a
  permutation null of the scan MAXIMUM over all layer-feature pairs,
  B_perm = 1000, 95th-percentile threshold (subsec:an_localisation).
- Confirmatory family: exactly {discrimination, contribution}, Bonferroni at
  family alpha = 0.05 — i.e. each claim is tested at alpha/2, so the
  Bonferroni-corrected decision uses the 97.5% CI (subsec:an_sensitivity).
- Per-category descriptive AUROC for the largest TruthfulQA categories.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (identical to sklearn.roc_auc_score, average ties)."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def folded(a: float | np.ndarray) -> float | np.ndarray:
    """Direction-agnostic per-layer statistic max(AUROC, 1-AUROC)."""
    return np.maximum(a, 1.0 - a)


# ----------------------------------------------------------------------
# Bootstrap (discrimination and contribution claims)
# ----------------------------------------------------------------------

def _resample_indices(rng: np.random.Generator, y: np.ndarray, max_redraws: int = 1000) -> np.ndarray:
    """One bootstrap resample of the test set; single-class draws are redrawn
    (the AUROC is undefined there, subsec:an_discrimination)."""
    n = len(y)
    for _ in range(max_redraws):
        idx = rng.integers(0, n, size=n)
        if 0 < y[idx].sum() < n:
            return idx
    raise RuntimeError("could not draw a two-class bootstrap resample")


def bootstrap_auroc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alphas: tuple[float, ...] = (0.05, 0.025),
) -> dict[str, Any]:
    """Test AUROC with bootstrap percentile CIs.

    ``alphas``: 0.05 gives the reported 95% CI; 0.025 gives the
    Bonferroni-corrected (2-test family) decision interval.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _resample_indices(rng, y)
        draws[b] = auroc(y[idx], scores[idx])
    out: dict[str, Any] = {
        "auroc": auroc(y, scores),
        "n_bootstrap": n_bootstrap,
        "n_test": len(y),
    }
    for alpha in alphas:
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"ci_{1 - alpha:.3f}"] = [float(lo), float(hi)]
    # discrimination claim: lower confidence bound above 0.5
    out["discriminates_95"] = out["ci_0.950"][0] > 0.5
    out["discriminates_bonferroni"] = out["ci_0.975"][0] > 0.5
    return out


def paired_bootstrap_delta_auroc(
    y: np.ndarray,
    scores_full: np.ndarray,
    scores_base: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alphas: tuple[float, ...] = (0.05, 0.025),
) -> dict[str, Any]:
    """delta AUROC = AUROC(full) - AUROC(base) on IDENTICAL resamples: both
    probes are evaluated on the same draw, so the variance of the difference
    excludes the sampling noise common to both (subsec:an_contribution)."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _resample_indices(rng, y)
        draws[b] = auroc(y[idx], scores_full[idx]) - auroc(y[idx], scores_base[idx])
    out: dict[str, Any] = {
        "delta_auroc": auroc(y, scores_full) - auroc(y, scores_base),
        "auroc_full": auroc(y, scores_full),
        "auroc_base": auroc(y, scores_base),
        "n_bootstrap": n_bootstrap,
        "n_test": len(y),
    }
    for alpha in alphas:
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"ci_{1 - alpha:.3f}"] = [float(lo), float(hi)]
    # contribution claim: lower bound of the delta CI positive
    out["contributes_95"] = out["ci_0.950"][0] > 0
    out["contributes_bonferroni"] = out["ci_0.975"][0] > 0
    return out


# ----------------------------------------------------------------------
# Depth-axis localisation (exploratory)
# ----------------------------------------------------------------------

def _impute_median(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan = np.isnan(col)
        if nan.any():
            col[nan] = np.nanmedian(col) if not nan.all() else 0.0
    return X


def per_feature_folded_auroc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Observed folded AUROC of every column of X (single features, no
    fitted model). NaNs are median-imputed, matching the probe pipeline."""
    X = _impute_median(X)
    y = np.asarray(y, dtype=int)
    return np.array([folded(auroc(y, X[:, j])) for j in range(X.shape[1])])


def permutation_scan_threshold(
    X: np.ndarray,
    y: np.ndarray,
    n_permutation: int = 1_000,
    seed: int = 42,
    percentile: float = 95.0,
) -> dict[str, Any]:
    """Permutation null of the scan maximum (subsec:an_localisation).

    Labels are shuffled; the MAXIMUM folded AUROC over ALL layer-feature
    columns is recomputed per permutation. Column ranks are label-free, so
    they are computed once and reused across permutations.
    """
    X = _impute_median(X)
    y = np.asarray(y, dtype=int)
    n, m = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    ranks = np.column_stack([rankdata(X[:, j]) for j in range(m)])  # [N, M]

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_permutation)
    y_perm = y.copy()
    for b in range(n_permutation):
        rng.shuffle(y_perm)
        pos_rank_sum = ranks[y_perm == 1].sum(axis=0)
        a = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        null_max[b] = folded(a).max()

    observed = per_feature_folded_auroc(X, y)
    threshold = float(np.percentile(null_max, percentile))
    return {
        "observed_folded_auroc": observed,
        "scan_threshold": threshold,
        "percentile": percentile,
        "n_permutation": n_permutation,
        "n_significant": int((observed > threshold).sum()),
        "null_max_distribution": null_max,
    }


# ----------------------------------------------------------------------
# Descriptive breakdowns
# ----------------------------------------------------------------------

def per_category_auroc(
    categories: list[str | None],
    y: np.ndarray,
    scores: np.ndarray,
    min_n: int = 10,
) -> list[dict[str, Any]]:
    """Descriptive AUROC within the largest categories
    (subsec:an_discrimination); categories needing both classes and at
    least ``min_n`` samples."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cats = np.asarray([c if c is not None else "(none)" for c in categories])
    out = []
    for cat in sorted(set(cats)):
        mask = cats == cat
        n = int(mask.sum())
        if n < min_n or len(np.unique(y[mask])) < 2:
            continue
        out.append({
            "category": cat,
            "n": n,
            "hallucination_rate": float(y[mask].mean()),
            "auroc": auroc(y[mask], scores[mask]),
        })
    out.sort(key=lambda r: -r["n"])
    return out
