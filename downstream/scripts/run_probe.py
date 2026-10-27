#!/usr/bin/env python
"""Stage 4 (probe + statistics) — headless run of the full evaluation
protocol: locked split, base/full probes, bootstrap discrimination, paired
bootstrap contribution, permutation scan, per-category AUROC. Writes
results/<exp>/probe_results.json. The analysis notebook reproduces and
extends these numbers with figures.

Usage:
    python scripts/run_probe.py --config configs/e1.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.config import load_config
from energy_llm.features import baseline_matrix, infer_n_layers, phi_matrix
from energy_llm.io_artifacts import load_json, save_json_atomic
from energy_llm.probe import make_locked_splits, make_probe_pipeline, run_base_and_full_probes
from energy_llm.stats import (
    bootstrap_auroc_ci,
    paired_bootstrap_delta_auroc,
    per_category_auroc,
    permutation_scan_threshold,
)

SMOKE_THRESHOLD = 30  # below this, the full protocol is statistically void


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    df = pd.read_csv(cfg.features_dir / "features.csv")
    df = df[df["is_hallucination"].notna()].reset_index(drop=True)
    n = len(df)
    n_layers = infer_n_layers(df)
    y = df["is_hallucination"].to_numpy(dtype=int)
    X_base = baseline_matrix(df)
    X_phi = phi_matrix(df, n_layers)

    if n < SMOKE_THRESHOLD:
        # dry-run path: too few labelled samples for the stratified protocol;
        # fit one pipeline on everything just to exercise the code path.
        print(f"[probe] SMOKE MODE: only {n} labelled samples (<{SMOKE_THRESHOLD}); "
              "fitting a single probe with no protocol — numbers are meaningless")
        pipe = make_probe_pipeline(C=1.0, seed=cfg.seed)
        pipe.fit(np.hstack([X_base, X_phi]), y)
        scores = pipe.predict_proba(np.hstack([X_base, X_phi]))[:, 1]
        save_json_atomic(cfg.out / "probe_results.json",
                         {"smoke": True, "n": n, "in_sample_scores_ok": bool(np.isfinite(scores).all())})
        return

    cal_ids: list[str] = []
    cal_path = cfg.calibration_dir / "calibration.json"
    if cal_path.exists():
        cal_ids = load_json(cal_path)["calibration_sample_ids"]

    splits = make_locked_splits(
        df["sample_id"].tolist(), y, cfg.out / "split.json",
        seed=cfg.seed, ratios=tuple(cfg.probe.split), force_train_ids=cal_ids,
    )
    idx = splits.index_of(df["sample_id"].tolist())

    probes = run_base_and_full_probes(
        X_base, X_phi, y, idx, c_grid=cfg.probe.c_grid,
        inner_cv_folds=cfg.probe.inner_cv_folds,
        outer_cv_folds=cfg.probe.outer_cv_folds, seed=cfg.seed,
    )
    y_test = probes["full"]["y_test"]

    discrimination = bootstrap_auroc_ci(
        y_test, probes["full"]["test_scores"],
        n_bootstrap=cfg.stats.n_bootstrap, seed=cfg.seed,
    )
    contribution = paired_bootstrap_delta_auroc(
        y_test, probes["full"]["test_scores"], probes["base"]["test_scores"],
        n_bootstrap=cfg.stats.n_bootstrap, seed=cfg.seed,
    )
    scan = permutation_scan_threshold(
        X_phi, y, n_permutation=cfg.stats.n_permutation, seed=cfg.seed,
    )
    test_mask = np.isin(np.arange(n), idx["test"])
    categories = per_category_auroc(
        df.loc[test_mask, "category"].tolist(), y[test_mask],
        probes["full"]["test_scores"], min_n=10,
    )

    results = {
        "experiment": cfg.name,
        "n_labelled": n,
        "n_layers": n_layers,
        "beta_star": cfg.calibration.beta_star,
        "split_sizes": {k: int(len(v)) for k, v in idx.items()},
        "test_split_hash": splits.test_hash,
        "probes": {
            kind: {
                "best_c": p["best_c"],
                "inner_cv_auroc_by_c": p["inner_cv_auroc_by_c"],
                "outer_cv_aurocs": p["outer_cv_aurocs"],
                "outer_cv_auroc_mean": p["outer_cv_auroc_mean"],
                "outer_cv_auroc_std": p["outer_cv_auroc_std"],
                "test_auroc": p["test_auroc"],
            }
            for kind, p in probes.items()
        },
        "confirmatory": {
            "discrimination": {k: v for k, v in discrimination.items()},
            "contribution": {k: v for k, v in contribution.items()},
            "bonferroni_note": "family alpha=0.05 over exactly 2 tests; "
                               "corrected decisions use the 97.5% CIs",
        },
        "localisation": {
            "scan_threshold": scan["scan_threshold"],
            "n_significant": scan["n_significant"],
            "observed_folded_auroc": scan["observed_folded_auroc"].tolist(),
        },
        "per_category_test_auroc": categories,
    }
    save_json_atomic(cfg.out / "probe_results.json", results)

    print(f"[probe] full test AUROC = {probes['full']['test_auroc']:.4f}  "
          f"95% CI {discrimination['ci_0.950']}")
    print(f"[probe] base test AUROC = {probes['base']['test_auroc']:.4f}")
    print(f"[probe] delta AUROC = {contribution['delta_auroc']:.4f}  "
          f"95% CI {contribution['ci_0.950']}")
    print(f"[probe] discrimination (Bonferroni): {discrimination['discriminates_bonferroni']}, "
          f"contribution (Bonferroni): {contribution['contributes_bonferroni']}")
    print(f"[probe] permutation scan threshold = {scan['scan_threshold']:.4f}, "
          f"{scan['n_significant']}/{4 * n_layers} layer-feature pairs above")
    print(f"[probe] wrote {cfg.out / 'probe_results.json'}")


if __name__ == "__main__":
    main()
