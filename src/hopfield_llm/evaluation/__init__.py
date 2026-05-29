"""Evaluation pipeline: per-layer AUROC, logistic-regression probe, baselines,
beta sweep, and HTML report.

Usage
-----
    from hopfield_llm.evaluation.features import load_all_features
    from hopfield_llm.evaluation.per_layer_auroc import (
        per_layer_auroc_table, best_single_feature,
    )
    from hopfield_llm.evaluation.probe import fit_logreg_probe
    from hopfield_llm.evaluation.report import build_eval_report

    # C3 — Semantic Entropy (Farquhar 2024)
    from hopfield_llm.evaluation.semantic_entropy import (
        SemanticEntropyScorer, run_semantic_entropy_batch,
    )

    # C4 — HalluField (temperature perturbation)
    from hopfield_llm.evaluation.hallufield import (
        compute_hallufield_score, run_hallufield_batch,
    )

    # C5 — INTRA (Vazhentsev 2026)
    from hopfield_llm.evaluation.intra import run_intra_batch
"""

from hopfield_llm.evaluation.features import (
    load_all_features,
    load_sample_features,
    save_sample_label,
)
from hopfield_llm.evaluation.per_layer_auroc import (
    best_single_feature,
    per_layer_auroc_table,
)
from hopfield_llm.evaluation.probe import fit_logreg_probe
from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc
from hopfield_llm.evaluation.report import build_eval_report

__all__ = [
    "load_sample_features",
    "load_all_features",
    "save_sample_label",
    "per_layer_auroc_table",
    "best_single_feature",
    "fit_logreg_probe",
    "beta_sweep_auroc",
    "build_eval_report",
]
