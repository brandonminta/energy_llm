"""Compare all hallucination-detection baselines on a trajectory directory.

Loads per-sample files and produces an AUROC comparison table covering:
  - Hopfield ΔE / ΔH probes (from .npz + labels)
  - Token-entropy baselines (from _baselines.json)
  - HalluField score (from _hallufield.json)
  - Semantic entropy (from _semmentropy.json)
  - INTRA hidden-state probe (from _intra.npz, with PCA+LR)

Requires all baseline files to be pre-computed.  Missing files are silently
skipped and marked N/A in the table.

Usage:
    python scripts/local/compare_all_baselines.py \\
        --traj-dir exp04_results/trajectories \\
        --label-dir exp04_results/labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ── Loader helpers ────────────────────────────────────────────────────────────

def _load_npz_scalar(traj_dir: Path, sample_id: str, key: str) -> float | None:
    npz_path = traj_dir / f"{sample_id}.npz"
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)
    if key not in npz:
        return None
    arr = npz[key].astype(np.float32)   # [L, T] or [L]
    return float(np.nanmean(arr))


def _load_label(label_dir: Path, sample_id: str) -> int | None:
    lpath = label_dir / f"{sample_id}_label.json"
    if not lpath.exists():
        return None
    d = json.loads(lpath.read_text())
    v = d.get("is_hallucination")
    return int(v) if v is not None else None


def _auroc(y_true, scores) -> float:
    y = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    mask = ~np.isnan(s)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def _intra_probe_auroc(
    intra_features: list[np.ndarray],
    labels: list[int],
    n_components: int = 64,
    n_splits: int = 5,
) -> float:
    """Fit a PCA+LR probe on INTRA hidden states, return nested-CV AUROC."""
    X = np.vstack(intra_features)   # [N, L*d]
    y = np.array(labels, dtype=int)
    if X.shape[0] < 50 or len(np.unique(y)) < 2:
        return float("nan")

    # Impute NaN
    for j in range(X.shape[1]):
        nan_j = np.isnan(X[:, j])
        if nan_j.any():
            X[nan_j, j] = np.nanmedian(X[:, j])

    outer_cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in outer_cv.split(X, y):
        sc  = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_val = sc.transform(X[val_idx])
        n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_r  = pca.fit_transform(X_tr)
        X_val_r = pca.transform(X_val)
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_tr_r, y[tr_idx])
        try:
            aucs.append(roc_auc_score(y[val_idx], clf.predict_proba(X_val_r)[:, 1]))
        except ValueError:
            aucs.append(float("nan"))
    return float(np.nanmean(aucs))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traj-dir",   required=True)
    parser.add_argument("--label-dir",  default=None)
    parser.add_argument("--intra-ncomp", type=int, default=64,
                        help="PCA components for INTRA probe")
    args = parser.parse_args()

    traj_dir  = Path(args.traj_dir)
    label_dir = Path(args.label_dir) if args.label_dir else traj_dir.parent / "labels"

    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(t in f.name for t in ("calibration", "_label", "_baselines",
                                          "_hallufield", "_intra", "_scores", "_hpre",
                                          "_lam", "_semmentropy"))
    )
    print(f"Found {len(sample_files)} trajectory samples.\n")

    # Per-sample feature accumulation
    labels:             list[int]         = []
    delta_energy_sc:    list[float]       = []
    delta_entropy_sc:   list[float]       = []
    seq_logprob:        list[float]       = []
    token_entropy_mean: list[float]       = []
    token_entropy_max:  list[float]       = []
    hallufield_sc:      list[float]       = []
    semmentropy_sc:     list[float]       = []
    intra_last_feats:   list[np.ndarray]  = []
    intra_labels:       list[int]         = []

    for jf in sample_files:
        meta = json.loads(jf.read_text())
        sid  = meta.get("id", jf.stem)

        y = _load_label(label_dir, sid)
        if y is None:
            continue
        labels.append(y)

        # Hopfield scalars (mean across layers)
        e_gen  = _load_npz_scalar(traj_dir, sid, "gen_energy")
        e_pre  = _load_npz_scalar(traj_dir, sid, "prefill_energy")
        h_gen  = _load_npz_scalar(traj_dir, sid, "gen_entropy")
        h_pre  = _load_npz_scalar(traj_dir, sid, "prefill_entropy")
        delta_energy_sc.append(
            e_gen - e_pre if (e_gen is not None and e_pre is not None) else float("nan")
        )
        delta_entropy_sc.append(
            h_gen - h_pre if (h_gen is not None and h_pre is not None) else float("nan")
        )

        # Logit baselines
        bl_path = traj_dir / f"{sid}_baselines.json"
        if bl_path.exists():
            bl = json.loads(bl_path.read_text())
            seq_logprob.append(float(bl.get("seq_logprob_mean", float("nan"))))
            token_entropy_mean.append(float(bl.get("token_entropy_mean", float("nan"))))
            token_entropy_max.append(float(bl.get("token_entropy_max", float("nan"))))
        else:
            seq_logprob.append(float("nan"))
            token_entropy_mean.append(float("nan"))
            token_entropy_max.append(float("nan"))

        # HalluField
        hf_path = traj_dir / f"{sid}_hallufield.json"
        if hf_path.exists():
            hf = json.loads(hf_path.read_text())
            hallufield_sc.append(float(hf.get("hallufield_score", float("nan"))))
        else:
            hallufield_sc.append(float("nan"))

        # Semantic entropy
        se_path = traj_dir / f"{sid}_semmentropy.json"
        if se_path.exists():
            se = json.loads(se_path.read_text())
            semmentropy_sc.append(float(se.get("semantic_entropy", float("nan"))))
        else:
            semmentropy_sc.append(float("nan"))

        # INTRA hidden states
        intra_path = traj_dir / f"{sid}_intra.npz"
        if intra_path.exists():
            npz = np.load(intra_path)
            # Use last-token: [L, d] → flatten → [L*d]
            flat = npz["gen_hidden_last"].flatten()
            intra_last_feats.append(flat)
            intra_labels.append(y)

    y = np.array(labels, dtype=int)
    N  = len(y)
    n_hal = int(y.sum())
    n_ok  = N - n_hal
    print(f"Labeled samples: N={N}  HAL={n_hal}  OK={n_ok}")
    print(f"  Hopfield scalars:    {sum(1 for v in delta_energy_sc if not np.isnan(v))}/{N}")
    print(f"  Logit baselines:     {sum(1 for v in seq_logprob if not np.isnan(v))}/{N}")
    print(f"  HalluField:          {sum(1 for v in hallufield_sc if not np.isnan(v))}/{N}")
    print(f"  Semantic entropy:    {sum(1 for v in semmentropy_sc if not np.isnan(v))}/{N}")
    print(f"  INTRA (npz):         {len(intra_last_feats)}/{N}")
    print()

    # AUROC table
    rows = [
        ("ΔE scalar (mean over layers)",     _auroc(y, delta_energy_sc)),
        ("ΔH scalar (mean over layers)",     _auroc(y, delta_entropy_sc)),
        ("seq_logprob_mean (↑ = OK)",        _auroc(y, seq_logprob)),
        ("token_entropy_mean (↑ = HAL)",     _auroc(y, token_entropy_mean)),
        ("token_entropy_max  (↑ = HAL)",     _auroc(y, token_entropy_max)),
        ("HalluField score   (↑ = HAL)",     _auroc(y, hallufield_sc)),
        ("Semantic entropy   (↑ = HAL)",     _auroc(y, semmentropy_sc)),
        ("INTRA (PCA+LR, nested-CV)",
         _intra_probe_auroc(intra_last_feats, intra_labels, args.intra_ncomp)
         if intra_last_feats else float("nan")),
    ]

    print(f"{'Method':<40}  {'AUROC':>8}")
    print("-" * 52)
    for name, auc in rows:
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"{name:<40}  {auc_str:>8}")

    best_name, best_auc = max(rows, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
    print(f"\nBest: {best_name}  →  {best_auc:.4f}")

    # Cliff's δ: Hopfield ΔE vs best baseline
    se_auc = next(v for n, v in rows if "emantic" in n)
    de_auc = next(v for n, v in rows if "ΔE scalar" in n)
    if not (np.isnan(se_auc) or np.isnan(de_auc)):
        print(f"\nHopfield ΔE scalar vs Semantic Entropy:")
        print(f"  ΔE = {de_auc:.4f},  SE = {se_auc:.4f},  Δ = {de_auc - se_auc:+.4f}")


if __name__ == "__main__":
    main()
