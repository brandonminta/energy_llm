"""Logistic-regression probe for combined hallucination detection.

fit_logreg_probe  — StratifiedKFold LR with nested C-grid search, returns
                    mean/std AUROC and per-fold details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.probe")

_C_GRID = [0.01, 0.1, 1.0, 10.0]
_MIN_SAMPLES = 50


def fit_logreg_probe(
    samples: list[dict],
    feature_keys: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Fit a logistic-regression probe using per-layer features.

    Feature matrix construction: for each key in feature_keys, the [L] array
    is extracted per sample and concatenated horizontally → [N, n_keys * L].

    NaN imputation uses per-feature-column median, logged as a rate.

    C is chosen via 3-fold inner CV on each outer training fold.

    Args:
        samples:       Feature dicts from load_sample_features.
        feature_keys:  Keys with [L] per-layer arrays to concatenate.
        n_splits:      Number of outer StratifiedKFold splits.
        random_state:  Random seed for reproducibility.

    Returns:
        {
          "mean_auroc":     float,
          "std_auroc":      float,
          "per_fold":       list[float],
          "n_features":     int,
          "coef_importance": pd.Series,   # mean |coef| per feature
        }

    Raises:
        ValueError: When fewer than _MIN_SAMPLES (50) labeled samples are
                    available — estimates are too noisy to be meaningful.
    """
    n = len(samples)
    if n < _MIN_SAMPLES:
        raise ValueError(
            f"fit_logreg_probe requires ≥{_MIN_SAMPLES} labeled samples, "
            f"got {n}.  Collect more data before fitting the probe."
        )

    labels = np.array([int(s["is_hallucination"]) for s in samples], dtype=float)

    # ── Build feature matrix ────────────────────────────────────────────────
    blocks = [
        np.stack([s[k].astype(np.float64) for s in samples], axis=0)
        for k in feature_keys
    ]  # each block: [N, L]
    X = np.concatenate(blocks, axis=1).astype(np.float64)  # [N, n_keys * L]

    n_total_features = X.shape[1]

    # ── NaN imputation (global median per column) ───────────────────────────
    nan_mask = np.isnan(X)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        impute_rate = n_nan / X.size
        log.info(
            "fit_logreg_probe: imputing %d NaNs (%.2f%%) with per-column median",
            n_nan, impute_rate * 100,
        )
        for j in range(n_total_features):
            col_nan = nan_mask[:, j]
            if col_nan.any():
                median = float(np.nanmedian(X[:, j]))
                X[col_nan, j] = median

    # ── Outer CV ────────────────────────────────────────────────────────────
    outer_cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_cv   = StratifiedKFold(n_splits=3,        shuffle=True, random_state=random_state)
    fold_aurocs: list[float] = []
    fold_coefs:  list[np.ndarray] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_cv.split(X, labels)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = labels[tr_idx], labels[val_idx]

        # Inner C selection
        best_C      = _C_GRID[0]
        best_inner  = -1.0
        for C in _C_GRID:
            inner_aurocs: list[float] = []
            for in_tr, in_val in inner_cv.split(X_tr, y_tr):
                sc  = StandardScaler()
                Xii = sc.fit_transform(X_tr[in_tr])
                Xiv = sc.transform(X_tr[in_val])
                clf = LogisticRegression(
                    C=C, max_iter=1000, random_state=random_state,
                )
                clf.fit(Xii, y_tr[in_tr])
                try:
                    inner_aurocs.append(
                        float(roc_auc_score(y_tr[in_val], clf.predict_proba(Xiv)[:, 1]))
                    )
                except ValueError:
                    inner_aurocs.append(0.0)
            mean_inner = float(np.mean(inner_aurocs))
            if mean_inner > best_inner:
                best_inner = mean_inner
                best_C     = C

        # Outer fit
        scaler  = StandardScaler()
        X_trs   = scaler.fit_transform(X_tr)
        X_vals  = scaler.transform(X_val)
        clf_out = LogisticRegression(
            C=best_C, max_iter=1000, random_state=random_state,
        )
        clf_out.fit(X_trs, y_tr)
        try:
            auroc = float(roc_auc_score(y_val, clf_out.predict_proba(X_vals)[:, 1]))
        except ValueError:
            auroc = float("nan")

        fold_aurocs.append(auroc)
        fold_coefs.append(clf_out.coef_[0].copy())
        log.debug("fold %d: best_C=%.2f, val_auroc=%.4f", fold_idx, best_C, auroc)

    # ── Coefficient importance ───────────────────────────────────────────────
    mean_abs_coef = np.mean(np.abs(fold_coefs), axis=0)  # [n_total_features]

    # Build feature-name index
    feat_names: list[str] = []
    for k in feature_keys:
        L_k = samples[0][k].shape[0]
        feat_names.extend(f"{k}_l{l}" for l in range(L_k))

    coef_series = pd.Series(mean_abs_coef, index=feat_names, name="importance")

    mean_auroc = float(np.nanmean(fold_aurocs))
    std_auroc  = float(np.nanstd(fold_aurocs))
    log.info("fit_logreg_probe: mean_auroc=%.4f ± %.4f", mean_auroc, std_auroc)

    return {
        "mean_auroc":      mean_auroc,
        "std_auroc":       std_auroc,
        "per_fold":        fold_aurocs,
        "n_features":      n_total_features,
        "coef_importance": coef_series,
    }


# ------------------------------------------------------------------
# Out-of-fold predictions + paired-bootstrap nested-model test
# ------------------------------------------------------------------


def _build_matrix(samples: list[dict], feature_keys: list[str]) -> np.ndarray:
    """Stack feature_keys into [N, F], accepting both [L] arrays and scalars."""
    cols: list[np.ndarray] = []
    for k in feature_keys:
        v0 = samples[0][k]
        if np.ndim(v0) == 0:  # scalar feature → one column
            cols.append(np.array([[float(s[k]) if s[k] is not None else np.nan]
                                   for s in samples], dtype=np.float64))
        else:                 # per-layer [L] feature → L columns
            cols.append(np.stack([np.asarray(s[k], dtype=np.float64) for s in samples], axis=0))
    return np.concatenate(cols, axis=1).astype(np.float64)


def oof_probe_predictions(
    samples: list[dict],
    feature_keys: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, oof_proba)`` cross-validated out-of-fold probabilities.

    Mirrors :func:`fit_logreg_probe` (StandardScaler + nested C-grid + median
    imputation) but accepts mixed scalar/per-layer features and emits a
    per-sample prediction usable by the paired-bootstrap nested-model test.
    """
    n = len(samples)
    if n < _MIN_SAMPLES:
        raise ValueError(f"oof_probe_predictions requires ≥{_MIN_SAMPLES} samples, got {n}.")

    labels = np.array([int(s["is_hallucination"]) for s in samples], dtype=float)
    X = _build_matrix(samples, feature_keys)

    # Global-median imputation (column-wise).
    for j in range(X.shape[1]):
        col_nan = np.isnan(X[:, j])
        if col_nan.any():
            X[col_nan, j] = float(np.nanmedian(X[:, j])) if not np.isnan(X[:, j]).all() else 0.0

    oof = np.full(n, np.nan, dtype=float)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in outer_cv.split(X, labels):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = labels[tr_idx]

        best_C, best_inner = _C_GRID[0], -1.0
        for C in _C_GRID:
            inner = []
            for in_tr, in_val in inner_cv.split(X_tr, y_tr):
                sc = StandardScaler()
                clf = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
                clf.fit(sc.fit_transform(X_tr[in_tr]), y_tr[in_tr])
                try:
                    inner.append(float(roc_auc_score(
                        y_tr[in_val], clf.predict_proba(sc.transform(X_tr[in_val]))[:, 1])))
                except ValueError:
                    inner.append(0.0)
            if np.mean(inner) > best_inner:
                best_inner, best_C = float(np.mean(inner)), C

        scaler = StandardScaler()
        clf = LogisticRegression(C=best_C, max_iter=1000, random_state=random_state)
        clf.fit(scaler.fit_transform(X_tr), y_tr)
        oof[val_idx] = clf.predict_proba(scaler.transform(X_val))[:, 1]

    return labels, oof


def paired_bootstrap_delta_auroc(
    labels: np.ndarray,
    full_scores: np.ndarray,
    base_scores: np.ndarray,
    n_boot: int = 10_000,
    random_state: int = 42,
) -> dict:
    """Paired-bootstrap CI for ΔAUROC = AUROC(full) − AUROC(base).

    Resamples the test indices with replacement and recomputes ΔAUROC on each
    resample, returning the point estimate and a percentile 95% CI.  A positive
    lower bound is the condition under which the richer feature set is claimed
    to add information (methodology §paired_bootstrap).
    """
    labels = np.asarray(labels, dtype=float)
    full_scores = np.asarray(full_scores, dtype=float)
    base_scores = np.asarray(base_scores, dtype=float)

    mask = ~(np.isnan(full_scores) | np.isnan(base_scores))
    labels, full_scores, base_scores = labels[mask], full_scores[mask], base_scores[mask]
    n = len(labels)
    if n < _MIN_SAMPLES or labels.sum() == 0 or (1 - labels).sum() == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "auroc_full": float("nan"), "auroc_base": float("nan")}

    auroc_full = float(roc_auc_score(labels, full_scores))
    auroc_base = float(roc_auc_score(labels, base_scores))

    rng = np.random.default_rng(random_state)
    deltas = np.empty(n_boot, dtype=float)
    n_valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if yb.sum() == 0 or (1 - yb).sum() == 0:
            continue
        deltas[n_valid] = (roc_auc_score(yb, full_scores[idx])
                           - roc_auc_score(yb, base_scores[idx]))
        n_valid += 1
    deltas = deltas[:n_valid]
    lo, hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))) \
        if n_valid else (float("nan"), float("nan"))

    return {
        "delta": auroc_full - auroc_base,
        "ci_low": lo, "ci_high": hi,
        "auroc_full": auroc_full, "auroc_base": auroc_base,
    }
