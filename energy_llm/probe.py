"""Detection probe and evaluation protocol (subsec:probe, subsec:stage4).

The probe is a scikit-learn Pipeline chaining SimpleImputer (median),
StandardScaler, and LogisticRegression (lbfgs). The Pipeline class is used
deliberately: passed to the CV utilities, the imputer and scaler are re-fit
inside every training fold, so no held-out-fold statistics leak into the
standardisation.

Protocol: stratified 70/15/15 split at the fixed seed; the test split is
written once, hashed, and locked before any probe is fit. C is selected by
3-fold stratified CV on the training split only; the probe is re-fit on
train+validation with the selected C and evaluated ONCE on the test split.
An outer 5-fold stratified CV over train+validation with C held fixed is
reported as descriptive stability only.

The calibration sample ids are forced into the train split, which enforces
'calibration restricted to the training split' (subsec:beta_calibration) by
construction; the remainder is split stratified so the overall ratios hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_llm.io_artifacts import load_json, save_json_atomic, sha256_ids

log = logging.getLogger("energy_llm.probe")


def make_probe_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="lbfgs", C=C, max_iter=5000,
                                      random_state=seed)),
    ])


@dataclass
class Splits:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    test_hash: str
    forced_train_ids: list[str] = field(default_factory=list)

    def index_of(self, sample_ids: list[str]) -> dict[str, np.ndarray]:
        pos = {sid: i for i, sid in enumerate(sample_ids)}
        return {
            "train": np.array([pos[s] for s in self.train_ids if s in pos], dtype=int),
            "val": np.array([pos[s] for s in self.val_ids if s in pos], dtype=int),
            "test": np.array([pos[s] for s in self.test_ids if s in pos], dtype=int),
        }


def make_locked_splits(
    sample_ids: list[str],
    y: np.ndarray,
    split_path: str | Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    force_train_ids: list[str] | None = None,
) -> Splits:
    """Create the stratified 70/15/15 split once and lock it to disk.

    If the split file already exists it is LOADED, its test hash verified,
    and never recomputed — the test split is locked before any probe is fit
    and is not inspected until the final evaluation.
    """
    split_path = Path(split_path)
    if split_path.exists():
        raw = load_json(split_path)
        splits = Splits(**raw)
        if sha256_ids(splits.test_ids) != splits.test_hash:
            raise RuntimeError(f"locked test split in {split_path} fails its hash check")
        return splits

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    sample_ids = list(sample_ids)
    y = np.asarray(y, dtype=int)
    n = len(sample_ids)
    n_test = round(ratios[2] * n)
    n_val = round(ratios[1] * n)

    forced = sorted(set(force_train_ids or []) & set(sample_ids))
    pool = [i for i, sid in enumerate(sample_ids) if sid not in set(forced)]
    pool_y = y[pool]

    rest, test_idx = train_test_split(
        pool, test_size=n_test, stratify=pool_y, random_state=seed
    )
    rest_y = y[rest]
    train_pool, val_idx = train_test_split(
        rest, test_size=n_val, stratify=rest_y, random_state=seed
    )

    train_ids = sorted([sample_ids[i] for i in train_pool] + forced)
    splits = Splits(
        train_ids=train_ids,
        val_ids=sorted(sample_ids[i] for i in val_idx),
        test_ids=sorted(sample_ids[i] for i in test_idx),
        test_hash=sha256_ids(sample_ids[i] for i in test_idx),
        forced_train_ids=forced,
    )
    save_json_atomic(split_path, {
        "train_ids": splits.train_ids, "val_ids": splits.val_ids,
        "test_ids": splits.test_ids, "test_hash": splits.test_hash,
        "forced_train_ids": splits.forced_train_ids,
    })
    log.info("locked split written: %d/%d/%d (test hash %s…)",
             len(splits.train_ids), len(splits.val_ids), len(splits.test_ids),
             splits.test_hash[:12])
    return splits


def select_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: list[float],
    n_folds: int = 3,
    seed: int = 42,
) -> tuple[float, dict[float, float]]:
    """C by stratified CV on the TRAIN split only. The pipeline is cloned per
    fold by sklearn, so imputer/scaler are re-fit inside every fold."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores: dict[float, float] = {}
    for C in c_grid:
        fold_aurocs = []
        for fit_idx, eval_idx in cv.split(X_train, y_train):
            pipe = clone(make_probe_pipeline(C=C, seed=seed))
            pipe.fit(X_train[fit_idx], y_train[fit_idx])
            scores = pipe.predict_proba(X_train[eval_idx])[:, 1]
            fold_aurocs.append(roc_auc_score(y_train[eval_idx], scores))
        cv_scores[C] = float(np.mean(fold_aurocs))
    best_c = max(c_grid, key=lambda C: cv_scores[C])
    return best_c, cv_scores


def run_probe_protocol(
    X: np.ndarray,
    y: np.ndarray,
    idx: dict[str, np.ndarray],
    c_grid: list[float],
    inner_cv_folds: int = 3,
    outer_cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Full protocol of subsec:probe for one feature set.

    Returns the single test evaluation (the reported generalisation
    estimate), the selected C with its inner-CV table, the descriptive outer
    5-fold CV AUROCs over train+val, and the fitted final pipeline.
    """
    y = np.asarray(y, dtype=int)
    tr, va, te = idx["train"], idx["val"], idx["test"]

    # 1. C on train only (validation takes no part in selection)
    best_c, cv_table = select_c(X[tr], y[tr], c_grid, n_folds=inner_cv_folds, seed=seed)

    # 2. descriptive outer CV over train ∪ val with C held fixed
    trval = np.concatenate([tr, va])
    outer = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=seed)
    outer_aurocs = []
    for fit_idx, eval_idx in outer.split(X[trval], y[trval]):
        pipe = clone(make_probe_pipeline(C=best_c, seed=seed))
        pipe.fit(X[trval][fit_idx], y[trval][fit_idx])
        scores = pipe.predict_proba(X[trval][eval_idx])[:, 1]
        outer_aurocs.append(float(roc_auc_score(y[trval][eval_idx], scores)))

    # 3. refit on train ∪ val, single test evaluation
    final = make_probe_pipeline(C=best_c, seed=seed)
    final.fit(X[trval], y[trval])
    test_scores = final.predict_proba(X[te])[:, 1]
    test_auroc = float(roc_auc_score(y[te], test_scores))

    return {
        "best_c": best_c,
        "inner_cv_auroc_by_c": cv_table,
        "outer_cv_aurocs": outer_aurocs,
        "outer_cv_auroc_mean": float(np.mean(outer_aurocs)),
        "outer_cv_auroc_std": float(np.std(outer_aurocs)),
        "test_auroc": test_auroc,
        "test_scores": test_scores,
        "y_test": y[te],
        "pipeline": final,
    }


def run_base_and_full_probes(
    X_baselines: np.ndarray,
    X_phi: np.ndarray,
    y: np.ndarray,
    idx: dict[str, np.ndarray],
    c_grid: list[float],
    inner_cv_folds: int = 3,
    outer_cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Base probe (baselines only) and full probe (baselines + phi) under
    the IDENTICAL protocol (subsec:an_contribution)."""
    common = dict(idx=idx, c_grid=c_grid, inner_cv_folds=inner_cv_folds,
                  outer_cv_folds=outer_cv_folds, seed=seed)
    return {
        "base": run_probe_protocol(X_baselines, y, **common),
        "full": run_probe_protocol(np.hstack([X_baselines, X_phi]), y, **common),
    }
