"""HTML evaluation report builder.

build_eval_report  — loads all labeled samples, computes AUROC table, probe,
                     and baselines, then writes a self-contained HTML report
                     with embedded plots.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from hopfield_llm.evaluation.beta_sweep import beta_sweep_auroc
from hopfield_llm.evaluation.features import load_all_features
from hopfield_llm.evaluation.per_layer_auroc import best_single_feature, per_layer_auroc_table
from hopfield_llm.evaluation.probe import (
    _MIN_SAMPLES,
    fit_logreg_probe,
    oof_probe_predictions,
    paired_bootstrap_delta_auroc,
)
from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.report")

_PER_LAYER_KEYS = [
    "gen_energy_answer",
    "gen_entropy_answer",
    "delta_energy",
]

# Scalar baselines surfaced in the AUROC table. p_true / semantic_entropy /
# hallufield_score are NaN unless their sidecars were computed
# (python -m hopfield_llm.evaluation.{baselines,semantic_entropy,hallufield}).
_SCALAR_BASELINE_KEYS = [
    "seq_logprob_mean",
    "token_entropy_mean",
    "token_entropy_max",
    "p_true",
    "semantic_entropy",
    "hallufield_score",
]

_DEFAULT_BETA_SWEEP = [1.0, 5.0, 10.0, 15.0, 25.0, 50.0]


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _plot_per_layer_auroc(table: pd.DataFrame, keys: list[str]) -> str:
    data = table.drop("best_layer", errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in keys:
        if key not in data.columns:
            continue
        vals = data[key].values.astype(float)
        layers = np.arange(len(vals))
        ax.plot(layers, vals, marker="o", markersize=3, label=key, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("per-layer AUROC by feature")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_violin(samples: list[dict], feature_key: str, layer: int) -> str:
    vals_hall = [s[feature_key][layer] for s in samples if s["is_hallucination"]]
    vals_corr = [s[feature_key][layer] for s in samples if not s["is_hallucination"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot(
        [v for v in [vals_corr, vals_hall] if v],
        positions=[0, 1],
        showmedians=True,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Hallucination"])
    ax.set_ylabel(f"{feature_key} @ layer {layer}")
    ax.set_title("best_single_feature distribution")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_beta_sweep(sweep_df) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "best_auroc_energy" in sweep_df.columns:
        ax.plot(sweep_df.index, sweep_df["best_auroc_energy"],  marker="o", label="energy (best layer)")
    if "best_auroc_entropy" in sweep_df.columns:
        ax.plot(sweep_df.index, sweep_df["best_auroc_entropy"], marker="s", label="entropy (best layer)")
    ax.set_xlabel("beta")
    ax.set_ylabel("AUROC")
    ax.set_title("Beta sweep — best-layer AUROC")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ------------------------------------------------------------------
# Scalar AUROC helpers
# ------------------------------------------------------------------


def _scalar_auroc(samples: list[dict], key: str) -> tuple[float, float]:
    """Return (AUROC, PR-AUC) for a scalar feature, handling NaNs."""
    labels = np.array([int(s["is_hallucination"]) for s in samples], dtype=float)
    scores = np.array([
        s[key] if isinstance(s[key], (int, float)) and not math.isnan(s[key] if s[key] is not None else float("nan")) else float("nan")
        for s in samples
    ], dtype=float)
    mask = ~np.isnan(scores)
    if mask.sum() < 10 or labels[mask].sum() == 0 or (1 - labels[mask]).sum() == 0:
        return float("nan"), float("nan")
    try:
        auroc   = float(roc_auc_score(labels[mask], scores[mask]))
        prauc   = float(average_precision_score(labels[mask], scores[mask]))
        return auroc, prauc
    except ValueError:
        return float("nan"), float("nan")


# ------------------------------------------------------------------
# HTML templating
# ------------------------------------------------------------------


def _html_table(rows: list[dict], title: str = "") -> str:
    if not rows:
        return f"<p><em>{title}: no data</em></p>"
    cols = list(rows[0].keys())
    header = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for r in rows:
        cells = "".join(
            f"<td>{r[c]:.4f}</td>" if isinstance(r[c], float) else f"<td>{r[c]}</td>"
            for c in cols
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return (
        f"<h3>{title}</h3>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table>"
    )


def _img_tag(b64: str) -> str:
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;'/>"


# ------------------------------------------------------------------
# Main report builder
# ------------------------------------------------------------------


def build_eval_report(
    artifact_dir: Path,
    labels_dir: Path,
    out_path: Path,
    beta: float,
    dataset_name: str,
) -> Path:
    """Build an HTML evaluation report.

    Loads labeled samples, computes per-layer AUROC, fits logistic-regression
    probes, evaluates baselines, and writes a self-contained HTML file.

    Args:
        artifact_dir:  Trajectory artifact directory (.npz / .json files).
        labels_dir:    Label directory (_label.json files from M3).
        out_path:      Output path for the HTML report.
        beta:          Hopfield beta used during trajectory capture; also used
                       as the centre of the post-hoc beta sweep.
        dataset_name:  Human-readable dataset identifier for the report title.

    Returns:
        Path of the written HTML file.
    """
    artifact_dir = Path(artifact_dir)
    labels_dir   = Path(labels_dir)
    out_path     = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load samples ───────────────────────────────────────────────────────
    samples = load_all_features(artifact_dir, labels_dir)
    n = len(samples)
    log.info("build_eval_report: %d labeled samples, dataset=%s", n, dataset_name)

    sections: list[str] = []
    sections.append(
        f"<h1>Hallucination Detection Evaluation: {dataset_name}</h1>"
        f"<p>Samples: {n} | Beta: {beta}</p>"
    )

    if n == 0:
        sections.append("<p><strong>No labeled samples found.</strong></p>")
        html = _wrap_html(dataset_name, "\n".join(sections))
        out_path.write_text(html, encoding="utf-8")
        return out_path

    # ── per-layer AUROC ────────────────────────────────────────────────────
    sections.append("<h2>per-layer AUROC</h2>")

    auroc_keys = [k for k in _PER_LAYER_KEYS if k in samples[0]]
    table = per_layer_auroc_table(samples, auroc_keys)

    feat_key, best_l, best_auc = best_single_feature(table)
    sections.append(
        f"<p><strong>best_single_feature</strong>: {feat_key} @ layer {best_l}, "
        f"AUROC = {best_auc:.4f}</p>"
    )

    if auroc_keys:
        b64_curve = _plot_per_layer_auroc(table, auroc_keys)
        sections.append(_img_tag(b64_curve))
        b64_violin = _plot_violin(samples, feat_key, best_l)
        sections.append(_img_tag(b64_violin))

    # Show per-layer table for numeric rows only
    data_rows = table.drop("best_layer", errors="ignore")
    table_html_rows = [
        {"layer": i, **{k: float(data_rows.loc[i, k]) for k in auroc_keys}}
        for i in data_rows.index
    ]
    sections.append(_html_table(table_html_rows, "Per-layer AUROC table"))

    # ── Probe ──────────────────────────────────────────────────────────────
    sections.append("<h2>Logistic-Regression Probe</h2>")
    probe_results: list[dict] = []
    for variant_name, keys in [
        ("energy+entropy",  ["gen_energy_answer", "gen_entropy_answer"]),
        ("full feature set", auroc_keys),
    ]:
        valid_keys = [k for k in keys if k in samples[0]]
        if not valid_keys:
            continue
        try:
            res = fit_logreg_probe(samples, valid_keys)
            probe_results.append({
                "variant": variant_name,
                "mean_auroc": res["mean_auroc"],
                "std_auroc":  res["std_auroc"],
                "n_features": res["n_features"],
            })
        except ValueError as exc:
            sections.append(f"<p><em>Probe ({variant_name}) skipped: {exc}</em></p>")
    if probe_results:
        sections.append(_html_table(probe_results, "Probe results"))

    # ── Baselines ──────────────────────────────────────────────────────────
    sections.append("<h2>Baselines</h2>")
    baseline_rows: list[dict] = []
    for key in _SCALAR_BASELINE_KEYS:
        auroc, prauc = _scalar_auroc(samples, key)
        baseline_rows.append({"method": key, "auroc": auroc, "prauc": prauc})
    sections.append(_html_table(baseline_rows, "Scalar baseline AUROCs"))

    # ── Nested-model test: do Hopfield features add over baselines? ─────────
    # Compares an out-of-fold probe on [Hopfield + baselines] against one on
    # [baselines only] via a paired bootstrap of ΔAUROC (methodology §analysis).
    sections.append("<h2>Nested-model test (Hopfield features vs baselines-only)</h2>")
    hop_keys = [k for k in auroc_keys if k in samples[0]]
    avail_baselines = [
        k for k in _SCALAR_BASELINE_KEYS
        if not all(
            (s.get(k) is None) or (isinstance(s.get(k), float) and math.isnan(s[k]))
            for s in samples
        )
    ]
    if n >= _MIN_SAMPLES and hop_keys and avail_baselines:
        try:
            labels_b, base_oof = oof_probe_predictions(samples, avail_baselines)
            _, full_oof = oof_probe_predictions(samples, avail_baselines + hop_keys)
            nm = paired_bootstrap_delta_auroc(labels_b, full_oof, base_oof, n_boot=2000)
            sig = "yes" if (not math.isnan(nm["ci_low"]) and nm["ci_low"] > 0) else "no"
            sections.append(_html_table([{
                "auroc_baselines_only": nm["auroc_base"],
                "auroc_full":           nm["auroc_full"],
                "delta_auroc":          nm["delta"],
                "ci95_low":             nm["ci_low"],
                "ci95_high":            nm["ci_high"],
                "hopfield_adds_signal": sig,
            }], "Paired-bootstrap ΔAUROC (full − baselines-only)"))
            sections.append(
                f"<p><em>Baseline features: {', '.join(avail_baselines)} | "
                f"Hopfield features: {', '.join(hop_keys)} | "
                f"bootstrap resamples: 2000</em></p>"
            )
        except ValueError as exc:
            sections.append(f"<p><em>Nested-model test skipped: {exc}</em></p>")
    else:
        sections.append(
            "<p><em>Nested-model test skipped: needs ≥"
            f"{_MIN_SAMPLES} samples and ≥1 computed baseline sidecar "
            "(run python -m hopfield_llm.evaluation.baselines first).</em></p>"
        )

    # ── Method comparison table ────────────────────────────────────────────
    sections.append("<h2>Summary: method vs AUROC</h2>")
    comparison_rows: list[dict] = []
    comparison_rows.append({
        "method":  f"best_single_feature ({feat_key} L{best_l})",
        "auroc":   best_auc,
        "prauc":   float("nan"),
        "n_feat":  1,
    })
    for r in probe_results:
        comparison_rows.append({
            "method": f"probe_{r['variant']}",
            "auroc":  r["mean_auroc"],
            "prauc":  float("nan"),
            "n_feat": r["n_features"],
        })
    for r in baseline_rows:
        comparison_rows.append({
            "method": r["method"],
            "auroc":  r["auroc"],
            "prauc":  r["prauc"],
            "n_feat": 1,
        })
    sections.append(_html_table(comparison_rows, "Method comparison"))

    # ── Beta sweep (if scores available) ──────────────────────────────────
    has_scores = any(s.get("gen_scores") is not None for s in samples)
    if has_scores:
        sections.append("<h2>Beta sweep</h2>")
        sweep_betas = [b * f for b in [beta] for f in [0.1, 0.5, 1.0, 2.0, 5.0]]
        sweep_betas = sorted(set(round(x, 3) for x in sweep_betas if x > 0))
        try:
            sweep_df = beta_sweep_auroc(
                [s for s in samples if s.get("gen_scores") is not None],
                sweep_betas,
            )
            if not sweep_df.empty:
                sections.append(_img_tag(_plot_beta_sweep(sweep_df)))
                sweep_rows = [
                    {"beta": float(idx), **{k: float(v) for k, v in row.items()
                                            if k not in ("per_layer_auroc_energy", "per_layer_auroc_entropy")}}
                    for idx, row in sweep_df.iterrows()
                ]
                sections.append(_html_table(sweep_rows, "Beta sweep"))
        except ValueError as exc:
            sections.append(f"<p><em>Beta sweep skipped: {exc}</em></p>")

    # ── Write HTML ──────────────────────────────────────────────────────────
    html = _wrap_html(dataset_name, "\n".join(sections))
    out_path.write_text(html, encoding="utf-8")
    log.info("build_eval_report: written to %s", out_path)
    return out_path


def _wrap_html(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"  <meta charset='UTF-8'>\n"
        f"  <title>Eval: {title}</title>\n"
        "  <style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:1em}"
        "  table{border-collapse:collapse}th,td{padding:4px 8px}"
        "  img{margin:1em 0;display:block}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )
