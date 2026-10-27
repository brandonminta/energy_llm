"""Stage 4 — feature aggregation (subsec:feature_design, subsec:stage4).

Pooling is the MEAN over ALL generated positions 1..T_g — no answer-span
logic anywhere; no gold answers touch feature computation. Per layer l:

    delta_energy[l]       = mean_t E^(l,t) - E^(l,ref)        (eq:delta_energy)
    delta_entropy[l]      = mean_t H^(l,t) - H^(l,ref)        (eq:delta_entropy, diagnostic)
    delta_norm_entropy[l] = mean_t Hbar^(l,t) - Hbar^(l,ref)  (eq:delta_norm_entropy)
    mean_js[l]            = mean_t JS^(l,t)                   (eq:mean_js)
    mean_hellinger_sq[l]  = mean_t Hel^2^(l,t)                (eq:mean_hellinger)

with E = lse_term + quadratic_term (the two terms are stored separately and
summed only here). The feature vector is

    phi = [delta_energy || delta_norm_entropy || mean_js || mean_hellinger_sq]
        in R^{4L}                                             (eq:feature_vector)

delta_entropy (unnormalised) and the absolute generation-phase E / Hbar are
stored as diagnostics only and excluded from phi. Baselines from the same
decoding pass: length-normalised sequence log-probability and mean
token-level predictive entropy (subsec:an_contribution).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_llm import metrics
from energy_llm.io_artifacts import SampleManifest, load_json

log = logging.getLogger("energy_llm.features")

PHI_GROUPS = ["delta_energy", "delta_norm_entropy", "mean_js", "mean_hellinger_sq"]
DIAGNOSTIC_GROUPS = ["delta_entropy", "gen_energy_mean", "gen_norm_entropy_mean"]
BASELINE_COLUMNS = ["seq_logprob", "token_entropy_mean"]


def sample_features_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-layer features of one sample from its trajectory arrays."""
    ref_energy = arrays["ref_lse_term"] + arrays["ref_quadratic_term"]          # [L]
    gen_energy = arrays["gen_lse_term"] + arrays["gen_quadratic_term"]          # [L, T_g]
    return {
        "delta_energy": gen_energy.mean(axis=1) - ref_energy,
        "delta_entropy": arrays["gen_entropy"].mean(axis=1) - arrays["ref_entropy"],
        "delta_norm_entropy": arrays["gen_norm_entropy"].mean(axis=1) - arrays["ref_norm_entropy"],
        "mean_js": arrays["gen_js"].mean(axis=1),
        "mean_hellinger_sq": arrays["gen_hellinger_sq"].mean(axis=1),
        # diagnostics (stored, never in phi):
        "gen_energy_mean": gen_energy.mean(axis=1),
        "gen_norm_entropy_mean": arrays["gen_norm_entropy"].mean(axis=1),
    }


def sample_baselines_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "seq_logprob": float(np.mean(arrays["token_logprobs"])),
        "token_entropy_mean": float(np.mean(arrays["token_predictive_entropy"])),
    }


def feature_columns(n_layers: int, groups: list[str] | None = None) -> list[str]:
    groups = PHI_GROUPS if groups is None else groups
    return [f"{g}_l{l}" for g in groups for l in range(n_layers)]


def build_feature_table(
    trajectories_dir: str | Path,
    labels: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per completed sample: phi columns (group-major, layer-minor,
    matching eq:feature_vector), diagnostics, baselines, metadata, and the
    label when available. Failed samples carry NaN rows (median imputation
    in the probe pipeline handles them)."""
    trajectories_dir = Path(trajectories_dir)
    manifest = SampleManifest(trajectories_dir)
    rows: list[dict[str, Any]] = []
    n_layers: int | None = None

    for sid in sorted(manifest.complete_ids()):
        meta = load_json(trajectories_dir / f"{sid}.json")
        with np.load(trajectories_dir / f"{sid}.npz") as npz:
            arrays = {k: npz[k] for k in npz.files}
        L = arrays["ref_lse_term"].shape[0]
        if n_layers is None:
            n_layers = L
        elif n_layers != L:
            raise ValueError(f"inconsistent layer count: {sid} has {L}, expected {n_layers}")

        row: dict[str, Any] = {
            "sample_id": sid,
            "category": meta.get("category"),
            "source": meta.get("source"),
            "n_tokens_generated": meta.get("n_tokens_generated"),
            "failed": bool(meta.get("failed", False)),
        }
        feats = sample_features_from_arrays(arrays)
        for group in PHI_GROUPS + DIAGNOSTIC_GROUPS:
            for l in range(L):
                row[f"{group}_l{l}"] = feats[group][l]
        row.update(sample_baselines_from_arrays(arrays))
        if labels is not None:
            row["is_hallucination"] = labels.get(sid)
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("feature table: %d samples, L=%s", len(df), n_layers)
    return df


def phi_matrix(df: pd.DataFrame, n_layers: int) -> np.ndarray:
    """phi in R^{N x 4L}, column order per eq:feature_vector."""
    return df[feature_columns(n_layers)].to_numpy(dtype=np.float64)


def baseline_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[BASELINE_COLUMNS].to_numpy(dtype=np.float64)


def infer_n_layers(df: pd.DataFrame) -> int:
    layers = [int(c.rsplit("_l", 1)[1]) for c in df.columns if c.startswith("delta_energy_l")]
    if not layers:
        raise ValueError("no delta_energy_l* columns in feature table")
    return max(layers) + 1


# ----------------------------------------------------------------------
# Beta sensitivity: recompute features from cached score tensors
# (subsec:an_sensitivity) — no forward passes.
# ----------------------------------------------------------------------

def recompute_sample_arrays_from_scores(
    score_npz: dict[str, np.ndarray], beta: float
) -> dict[str, np.ndarray]:
    """Rebuild the full per-sample trajectory arrays at an arbitrary beta
    from the cached float16 retrieval-score tensors."""
    s_ref = score_npz["ref_retrieval_scores"].astype(np.float64)   # [L, K]
    s_gen = score_npz["gen_retrieval_scores"].astype(np.float64)   # [L, T_g, K]
    p_ref = metrics.retrieval_distribution(s_ref, beta)            # [L, K]
    p_gen = metrics.retrieval_distribution(s_gen, beta)            # [L, T_g, K]
    return {
        "ref_lse_term": metrics.lse_term(s_ref, beta),
        "ref_quadratic_term": score_npz["ref_quadratic_term"].astype(np.float64),
        "ref_entropy": metrics.entropy(p_ref),
        "ref_norm_entropy": metrics.norm_entropy(p_ref),
        "gen_lse_term": metrics.lse_term(s_gen, beta),
        # quadratic terms are beta-independent: cached verbatim, [L, T_g]
        "gen_quadratic_term": score_npz["gen_quadratic_term"].astype(np.float64),
        "gen_entropy": metrics.entropy(p_gen),
        "gen_norm_entropy": metrics.norm_entropy(p_gen),
        "gen_js": metrics.js_nats(p_gen, p_ref[:, None, :], axis=2),
        "gen_hellinger_sq": metrics.hellinger_sq(p_gen, p_ref[:, None, :], axis=2),
    }


def build_feature_table_at_beta(
    trajectories_dir: str | Path,
    beta: float,
    labels: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Feature table recomputed at ``beta`` from {id}_retrieval_scores.npz.
    Samples without a cached score tensor are skipped (logged)."""
    trajectories_dir = Path(trajectories_dir)
    manifest = SampleManifest(trajectories_dir)
    rows: list[dict[str, Any]] = []
    skipped = 0

    for sid in sorted(manifest.complete_ids()):
        score_path = trajectories_dir / f"{sid}_retrieval_scores.npz"
        if not score_path.exists():
            skipped += 1
            continue
        meta = load_json(trajectories_dir / f"{sid}.json")
        with np.load(score_path) as npz:
            score_arrays = {k: npz[k] for k in npz.files}
        arrays = recompute_sample_arrays_from_scores(score_arrays, beta)
        with np.load(trajectories_dir / f"{sid}.npz") as npz:
            arrays["token_logprobs"] = npz["token_logprobs"]
            arrays["token_predictive_entropy"] = npz["token_predictive_entropy"]

        L = arrays["ref_lse_term"].shape[0]
        row: dict[str, Any] = {"sample_id": sid, "category": meta.get("category")}
        feats = sample_features_from_arrays(arrays)
        for group in PHI_GROUPS:
            for l in range(L):
                row[f"{group}_l{l}"] = feats[group][l]
        row.update(sample_baselines_from_arrays(arrays))
        if labels is not None:
            row["is_hallucination"] = labels.get(sid)
        rows.append(row)

    if skipped:
        log.warning("beta recompute: %d samples skipped (no cached score tensors)", skipped)
    return pd.DataFrame(rows)
