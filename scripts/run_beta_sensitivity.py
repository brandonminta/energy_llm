#!/usr/bin/env python
"""Beta sensitivity analysis — recomputes phi from cached retrieval-score
tensors over a beta grid; no new forward passes.

For each beta in {1, 2, 5, 10, 15, 20, 30, 50, 100}:
  - delta_norm_entropy (Delta H_bar) is recomputed from the cached tensors
    (*_retrieval_scores.npz) at that beta; mean Delta H_bar per sample = mean
    over all layers of delta_norm_entropy[l].
  - phi is rebuilt at that beta via build_feature_table_at_beta and the full
    probe is re-run on the identical locked split (cfg.out/split.json) when it
    exists and there are >= SMOKE_THRESHOLD labelled examples; otherwise an
    in-sample AUROC is reported.

Output: results/<exp>/beta_sensitivity.csv
  columns: beta, sample_id, mean_delta_hbar, is_hallucination, probe_auroc

Usage:
    python scripts/run_beta_sensitivity.py --config configs/e1.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.config import load_config, require_beta_star
from energy_llm.features import (
    baseline_matrix,
    build_feature_table_at_beta,
    infer_n_layers,
    phi_matrix,
)
from energy_llm.io_artifacts import load_json
from energy_llm.labeling import load_labels
from energy_llm.probe import Splits, make_probe_pipeline, run_probe_protocol

log = logging.getLogger("energy_llm.beta_sensitivity")

BETA_GRID = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
# Below this threshold the locked-split protocol is skipped; an in-sample
# AUROC is reported instead (too few samples for stratified train/test).
SMOKE_THRESHOLD = 20


def run_beta_sensitivity(cfg) -> pd.DataFrame:
    """Core computation; returns the beta sensitivity DataFrame.

    Reads cached score tensors from ``cfg.trajectories_dir``, recomputes the
    full phi at each beta, and reports mean Delta H_bar per sample and probe
    AUROC per beta.  The locked split (``cfg.out/split.json``) is reused
    across all betas so AUROCs are directly comparable.
    """
    require_beta_star(cfg)

    if not list(cfg.trajectories_dir.glob("*_retrieval_scores.npz")):
        raise RuntimeError(
            f"no cached score tensors in {cfg.trajectories_dir} — "
            "re-run Stage 2 with trajectory.cache_score_tensors: true"
        )

    labels = load_labels(cfg.labels_dir) if cfg.labels_dir.exists() else None

    # Load the locked split once so all betas use the same test set.
    splits_path = cfg.out / "split.json"
    idx: dict | None = None

    beta_chunks: list[pd.DataFrame] = []
    for beta in BETA_GRID:
        log.info("beta=%.0f: recomputing feature table …", beta)
        df_b = build_feature_table_at_beta(
            cfg.trajectories_dir, float(beta), labels=labels
        )
        if df_b.empty:
            log.warning("beta=%.0f: no samples with cached score tensors — skipping", beta)
            continue

        n_layers = infer_n_layers(df_b)
        dh_cols = [f"delta_norm_entropy_l{l}" for l in range(n_layers)]
        mean_dh = df_b[dh_cols].mean(axis=1).to_numpy()  # [N] mean over layers

        hall_vals = (
            df_b["is_hallucination"].values
            if "is_hallucination" in df_b.columns
            else np.full(len(df_b), float("nan"))
        )

        # Probe AUROC — use the locked split when available and large enough,
        # otherwise fit on all labelled samples and evaluate in-sample.
        probe_auroc = float("nan")
        lab_mask = ~pd.isna(hall_vals)
        n_labelled = int(lab_mask.sum())
        both_classes = (
            n_labelled >= 2
            and len(np.unique(hall_vals[lab_mask].astype(int))) >= 2
        )

        if both_classes:
            df_lab = df_b[lab_mask].reset_index(drop=True)
            y_lab = df_lab["is_hallucination"].to_numpy(dtype=int)
            X = np.hstack([baseline_matrix(df_lab), phi_matrix(df_lab, n_layers)])

            if splits_path.exists() and n_labelled >= SMOKE_THRESHOLD:
                if idx is None:
                    raw = load_json(splits_path)
                    splits = Splits(**raw)
                    # Index into the labelled subset — same order for all betas
                    # because build_feature_table_at_beta sorts by sample_id.
                    idx = splits.index_of(df_lab["sample_id"].tolist())
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

        log.info(
            "beta=%.0f: probe_auroc=%.4f  n_labelled=%d",
            beta, probe_auroc, n_labelled,
        )

        chunk = pd.DataFrame({
            "beta": float(beta),
            "sample_id": df_b["sample_id"].values,
            "mean_delta_hbar": mean_dh,
            "is_hallucination": hall_vals,
            "probe_auroc": probe_auroc,
        })
        beta_chunks.append(chunk)

    return pd.concat(beta_chunks, ignore_index=True) if beta_chunks else pd.DataFrame(
        columns=["beta", "sample_id", "mean_delta_hbar", "is_hallucination", "probe_auroc"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)

    df = run_beta_sensitivity(cfg)
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
