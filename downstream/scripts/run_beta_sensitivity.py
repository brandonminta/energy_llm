#!/usr/bin/env python
"""Beta sensitivity analysis — single disk pass over cached retrieval-score
tensors; no model loads, no new forward passes.

Performance design
------------------
Original: build_feature_table_at_beta called once per beta → 9 sequential
passes over all *_retrieval_scores.npz files (≈ 33 GB × 9 = 297 GB of I/O).

This version:
  Phase 1 (disk I/O, parallelised with joblib):
    Read each *_retrieval_scores.npz ONCE.  For every beta in the grid,
    compute delta_norm_entropy from the cached scores inside the same worker
    before discarding the tensor.  Score tensors are never accumulated in the
    main process; peak memory from tensors is O(n_jobs × one_sample_size).

  Phase 2 (in-memory, no disk I/O):
    delta_energy / mean_js / mean_hellinger_sq are beta-independent for the
    probe sweep: they are loaded once from features.csv (values at beta_star)
    and held fixed.  Only the delta_norm_entropy columns are swapped per beta.
    The probe runs on the modified phi matrix against the locked split.

Target: < 20 min on 8 CPUs for E1 (817 samples, L=36, K=11008, T≤50).

Output: results/<exp>/beta_sensitivity.csv
  columns: beta, sample_id, mean_delta_hbar, is_hallucination, probe_auroc

Usage:
    python scripts/run_beta_sensitivity.py --config configs/e1.yaml
    python scripts/run_beta_sensitivity.py --config configs/e1.yaml --n-jobs 8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.config import load_config, require_beta_star
from energy_llm.features import baseline_matrix, infer_n_layers, phi_matrix
from energy_llm.io_artifacts import load_json
from energy_llm.labeling import load_labels
from energy_llm.probe import Splits, make_probe_pipeline, run_probe_protocol

log = logging.getLogger("energy_llm.beta_sensitivity")

BETA_GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
SMOKE_THRESHOLD = 20   # below this use in-sample AUROC, not the locked-split protocol
LOG_EVERY = 100        # log a progress line after every N samples in Phase 1


# ---------------------------------------------------------------------------
# Phase-1 worker — runs inside a joblib subprocess
# ---------------------------------------------------------------------------

def _compute_sample_dh(
    score_path: Path,
    sid: str,
    beta_grid: list[float],
) -> tuple[str, dict[float, np.ndarray]]:
    """Load one sample's retrieval-score tensors and return delta_norm_entropy
    for every beta.  The tensors are discarded when this function returns.

    Returns
    -------
    (sid, {beta: delta_norm_entropy_array [L]})
    """
    with np.load(score_path) as f:
        s_ref = f["ref_retrieval_scores"].astype(np.float64)  # [L, K]
        s_gen = f["gen_retrieval_scores"].astype(np.float64)  # [L, T_g, K]

    K = s_ref.shape[-1]
    log_K = float(np.log(K))
    dh_by_beta: dict[float, np.ndarray] = {}

    for beta in beta_grid:
        p_ref = softmax(beta * s_ref, axis=-1)   # [L, K]
        p_gen = softmax(beta * s_gen, axis=-1)   # [L, T_g, K]
        with np.errstate(divide="ignore", invalid="ignore"):
            H_ref = -np.sum(
                np.where(p_ref > 0.0, p_ref * np.log(p_ref), 0.0), axis=-1
            )  # [L]
            H_gen = -np.sum(
                np.where(p_gen > 0.0, p_gen * np.log(p_gen), 0.0), axis=-1
            )  # [L, T_g]
        # delta_norm_entropy[l] = mean_t Hbar_gen[l,t] - Hbar_ref[l]
        dh_by_beta[beta] = H_gen.mean(axis=-1) / log_K - H_ref / log_K  # [L]

    return sid, dh_by_beta


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_beta_sensitivity(cfg, n_jobs: int = 4) -> pd.DataFrame:
    """Core computation; returns the beta sensitivity DataFrame.

    Parameters
    ----------
    cfg:
        Loaded ExperimentConfig (snapshot preferred so beta_star is present).
    n_jobs:
        Number of joblib workers for the Phase-1 disk I/O pass.
    """
    require_beta_star(cfg)

    score_paths = sorted(cfg.trajectories_dir.glob("*_retrieval_scores.npz"))
    if not score_paths:
        raise RuntimeError(
            f"no cached score tensors in {cfg.trajectories_dir} — "
            "re-run Stage 2 with trajectory.cache_score_tensors: true"
        )
    score_path_by_sid = {
        p.name.replace("_retrieval_scores.npz", ""): p for p in score_paths
    }

    # --- load features.csv for beta-independent phi columns -----------------
    features_csv = cfg.features_dir / "features.csv"
    if not features_csv.exists():
        raise RuntimeError(
            f"{features_csv} not found — run scripts/build_features.py first"
        )
    df = pd.read_csv(features_csv)

    labels = load_labels(cfg.labels_dir) if cfg.labels_dir.exists() else None
    if labels is not None:
        df["is_hallucination"] = df["sample_id"].map(labels)

    n_layers = infer_n_layers(df)

    # Keep only samples present in both features.csv and score tensors.
    valid_mask = df["sample_id"].isin(score_path_by_sid)
    n_skip = int((~valid_mask).sum())
    if n_skip:
        log.warning(
            "%d samples in features.csv have no score tensor and will be skipped",
            n_skip,
        )
    df = df[valid_mask].reset_index(drop=True)
    valid_sids: list[str] = df["sample_id"].tolist()

    log.info(
        "Phase 1: %d samples × %d betas, n_jobs=%d",
        len(valid_sids), len(BETA_GRID), n_jobs,
    )

    # --- Phase 1: single disk pass (parallelised in batches of LOG_EVERY) ---
    sample_dh: dict[str, dict[float, np.ndarray]] = {}
    for batch_start in range(0, len(valid_sids), LOG_EVERY):
        batch = valid_sids[batch_start: batch_start + LOG_EVERY]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_sample_dh)(score_path_by_sid[sid], sid, BETA_GRID)
            for sid in batch
        )
        for sid, dh_per_beta in results:
            sample_dh[sid] = dh_per_beta
        n_done = min(batch_start + LOG_EVERY, len(valid_sids))
        log.info("  loaded %d/%d samples", n_done, len(valid_sids))

    # --- Phase 2: probe per beta — no disk I/O ------------------------------
    # Identify the labelled subset and pre-build the fixed parts of phi.
    lab_mask = (
        df["is_hallucination"].notna()
        if "is_hallucination" in df.columns
        else pd.Series(False, index=df.index)
    )
    df_lab = df[lab_mask].reset_index(drop=True)
    n_labelled = len(df_lab)
    both_classes = (
        n_labelled >= 2
        and df_lab["is_hallucination"].nunique() >= 2
    )

    # Pre-compute baseline + beta-independent phi once for the labelled rows.
    #
    # phi column order (eq:feature_vector):
    #   [delta_energy | delta_norm_entropy | mean_js | mean_hellinger_sq]
    #    cols 0..L-1    L..2L-1             2L..3L-1   3L..4L-1
    #
    # phi_fixed drops the delta_norm_entropy band (L..2L-1); it is re-inserted
    # per beta from sample_dh.
    X_base: np.ndarray | None = None
    phi_fixed: np.ndarray | None = None   # [N_lab, 3L]
    y_lab: np.ndarray | None = None
    lab_sids: list[str] = []
    idx: dict | None = None

    if both_classes:
        X_base = baseline_matrix(df_lab)
        phi_full = phi_matrix(df_lab, n_layers)   # [N_lab, 4L]
        phi_fixed = np.delete(
            phi_full,
            np.arange(n_layers, 2 * n_layers),    # remove delta_norm_entropy band
            axis=1,
        )   # [N_lab, 3L]  = [delta_energy | mean_js | mean_hellinger_sq]
        y_lab = df_lab["is_hallucination"].to_numpy(dtype=int)
        lab_sids = df_lab["sample_id"].tolist()

        splits_path = cfg.out / "split.json"
        if splits_path.exists() and n_labelled >= SMOKE_THRESHOLD:
            raw_split = load_json(splits_path)
            splits_obj = Splits(**raw_split)
            idx = splits_obj.index_of(lab_sids)

    log.info("Phase 2: probe sweep over %d betas (n_labelled=%d) …", len(BETA_GRID), n_labelled)

    hall_vals = (
        df["is_hallucination"].values
        if "is_hallucination" in df.columns
        else np.full(len(df), float("nan"))
    )

    beta_chunks: list[pd.DataFrame] = []
    for beta in BETA_GRID:
        # [N_valid, L] — mean over layers gives per-sample mean_delta_hbar [N_valid]
        dh_matrix = np.stack([sample_dh[sid][beta] for sid in valid_sids])
        mean_dh = dh_matrix.mean(axis=1)

        probe_auroc = float("nan")
        if both_classes:
            # [N_lab, L] — rows from the labelled subset only
            dh_lab = np.stack([sample_dh[sid][beta] for sid in lab_sids])

            # Reassemble phi with delta_norm_entropy substituted at this beta:
            # [delta_energy | dh_lab | mean_js | mean_hellinger_sq]
            X_phi_mod = np.concatenate(
                [
                    phi_fixed[:, :n_layers],   # delta_energy          [N_lab, L]
                    dh_lab,                    # delta_norm_entropy@β  [N_lab, L]
                    phi_fixed[:, n_layers:],   # mean_js + hell_sq     [N_lab, 2L]
                ],
                axis=1,
            )   # [N_lab, 4L]
            X = np.hstack([X_base, X_phi_mod])

            if idx is not None:
                result = run_probe_protocol(
                    X, y_lab, idx,
                    c_grid=cfg.probe.c_grid,
                    inner_cv_folds=cfg.probe.inner_cv_folds,
                    outer_cv_folds=cfg.probe.outer_cv_folds,
                    seed=cfg.seed,
                )
                probe_auroc = result["test_auroc"]
            else:
                pipe = make_probe_pipeline(C=1.0, seed=cfg.seed)
                pipe.fit(X, y_lab)
                scores = pipe.predict_proba(X)[:, 1]
                probe_auroc = float(roc_auc_score(y_lab, scores))

        log.info("beta=%.0f: probe_auroc=%.4f", beta, probe_auroc)

        beta_chunks.append(pd.DataFrame({
            "beta": float(beta),
            "sample_id": valid_sids,
            "mean_delta_hbar": mean_dh,
            "is_hallucination": hall_vals,
            "probe_auroc": probe_auroc,
        }))

    return (
        pd.concat(beta_chunks, ignore_index=True)
        if beta_chunks
        else pd.DataFrame(
            columns=["beta", "sample_id", "mean_delta_hbar", "is_hallucination", "probe_auroc"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help="joblib parallel workers for Phase-1 score-tensor loading (default 4)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)

    df = run_beta_sensitivity(cfg, n_jobs=args.n_jobs)
    out = cfg.out / "beta_sensitivity.csv"
    tmp = out.with_name(out.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(
        f"[beta_sensitivity] wrote {out}: {len(df)} rows, "
        f"{df['beta'].nunique()} betas, {df['sample_id'].nunique()} samples"
    )


if __name__ == "__main__":
    main()
