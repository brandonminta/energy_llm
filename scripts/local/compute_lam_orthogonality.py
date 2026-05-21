"""Compute LAM orthogonality per layer from _scores.npz + banks.pt.

Gupta et al. (2025) predict that factual-recall tokens activate highly
orthogonal memory keys (well-separated attractors), while hallucination tokens
activate correlated keys (ambiguous region).

For each sample that has a _scores.npz file, and for each layer ℓ:
  1. Identify the top-K activated memory patterns (by score).
  2. Compute mean pairwise cosine similarity between their bank rows.
  3. LAM-orthogonality = 1 - mean_cos_sim  (high = orthogonal, low = correlated)

Outputs:
  {traj_dir}/{sample_id}_lam.json   per sample per layer vector
  {traj_dir}/lam_summary.json       overall statistics + HAL/OK comparison

Usage:
    python scripts/local/compute_lam_orthogonality.py \\
        --traj-dir exp04_results/trajectories \\
        --banks    outputs/<run>/banks_key_space_no_norm_q.pt \\
        --top-k    8 \\
        --split    gen          # 'gen' | 'prefill' | 'both'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_banks(banks_path: Path) -> dict[int, torch.Tensor]:
    artifact = torch.load(banks_path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and "banks" in artifact:
        raw = artifact["banks"]
    else:
        raw = artifact
    # Keys may be ints or strings
    return {int(k): v.float() for k, v in raw.items() if k != "model"}


def _cosine_matrix(vecs: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarities for a [K, d] matrix."""
    normed = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return normed @ normed.T  # [K, K]


def _mean_pairwise_cos(bank_rows: torch.Tensor, top_k_indices: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity between top-k bank rows."""
    selected = bank_rows[top_k_indices.astype(int)]  # [top_k, d]
    if selected.shape[0] < 2:
        return float("nan")
    c = _cosine_matrix(selected)  # [top_k, top_k]
    n = selected.shape[0]
    # Off-diagonal mean
    off_diag = (c.sum() - c.diag().sum()) / (n * (n - 1))
    return float(off_diag)


def compute_lam_for_sample(
    scores_npz_path: Path,
    banks: dict[int, torch.Tensor],
    top_k: int = 8,
    split: str = "gen",
) -> dict[int, float]:
    """Return LAM-orthogonality per layer for one sample.

    Args:
        scores_npz_path: Path to {sample_id}_scores.npz.
        banks:           Dict[layer_idx → Tensor[K, d]].
        top_k:           Number of top-activated patterns to examine.
        split:           "gen", "prefill", or "both" (mean over both).

    Returns:
        Dict[layer_idx → float] — mean LAM-orthogonality (1 - mean_cos_sim)
    """
    npz = np.load(scores_npz_path, allow_pickle=True)
    results: dict[int, float] = {}

    for l_idx, bank in banks.items():
        arrays = []
        if split in ("gen", "both") and "gen_scores" in npz:
            # [T_gen, K]
            arrays.append(npz["gen_scores"][l_idx].astype(np.float32))
        if split in ("prefill", "both") and "prefill_scores" in npz:
            arrays.append(npz["prefill_scores"][l_idx].astype(np.float32))

        if not arrays:
            results[l_idx] = float("nan")
            continue

        combined = np.concatenate(arrays, axis=0)  # [T_total, K]
        # Mean score per pattern across all tokens
        mean_scores = combined.mean(axis=0)         # [K]
        top_idx = np.argsort(mean_scores)[-top_k:]  # highest-scoring patterns

        cos_sim = _mean_pairwise_cos(bank, top_idx)
        results[l_idx] = 1.0 - cos_sim if not np.isnan(cos_sim) else float("nan")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--banks",    required=True, help="Path to banks_*.pt")
    parser.add_argument("--top-k",    type=int, default=8)
    parser.add_argument("--split",    default="gen", choices=["gen", "prefill", "both"])
    parser.add_argument("--label-dir", default=None,
                        help="If provided, load labels for HAL/OK breakdown")
    args = parser.parse_args()

    traj_dir  = Path(args.traj_dir)
    banks     = _load_banks(Path(args.banks))
    label_dir = Path(args.label_dir) if args.label_dir else traj_dir.parent / "labels"
    n_layers  = len(banks)

    print(f"Banks: {len(banks)} layers loaded from {args.banks}")
    print(f"Top-k: {args.top_k}, Split: {args.split}\n")

    score_files = sorted(traj_dir.glob("*_scores.npz"))
    if not score_files:
        print("ERROR: no *_scores.npz files found in", traj_dir)
        return

    hal_lam: list[np.ndarray] = []
    ok_lam:  list[np.ndarray] = []
    all_lam: list[np.ndarray] = []

    for sf in score_files:
        sample_id = sf.stem.replace("_scores", "")
        out_path  = traj_dir / f"{sample_id}_lam.json"

        if out_path.exists():
            lam_per_layer = json.loads(out_path.read_text())
            lam_arr = np.array([lam_per_layer.get(str(l), float("nan")) for l in range(n_layers)])
        else:
            lam_per_layer = compute_lam_for_sample(sf, banks, args.top_k, args.split)
            out_path.write_text(json.dumps(
                {str(k): v for k, v in lam_per_layer.items()}, indent=2
            ))
            lam_arr = np.array([lam_per_layer.get(l, float("nan")) for l in range(n_layers)])

        all_lam.append(lam_arr)

        # Load label if available
        label_path = label_dir / f"{sample_id}_label.json"
        if label_path.exists():
            lbl = json.loads(label_path.read_text())
            if lbl.get("is_hallucination") is True:
                hal_lam.append(lam_arr)
            elif lbl.get("is_hallucination") is False:
                ok_lam.append(lam_arr)

    # Summary
    summary = {
        "n_samples":  len(all_lam),
        "n_hal":      len(hal_lam),
        "n_ok":       len(ok_lam),
        "top_k":      args.top_k,
        "split":      args.split,
        "mean_lam_all_layers": float(np.nanmean(all_lam)),
    }

    if hal_lam and ok_lam:
        hal_arr = np.nanmean(hal_lam, axis=0)
        ok_arr  = np.nanmean(ok_lam,  axis=0)
        delta   = hal_arr - ok_arr  # positive = HAL more orthogonal
        best_l  = int(np.nanargmax(np.abs(delta)))
        summary["hal_mean_lam"] = float(np.nanmean(hal_arr))
        summary["ok_mean_lam"]  = float(np.nanmean(ok_arr))
        summary["best_layer"]   = best_l
        summary["delta_at_best_layer"] = float(delta[best_l])
        summary["hal_per_layer"] = hal_arr.tolist()
        summary["ok_per_layer"]  = ok_arr.tolist()
        print(f"HAL mean LAM-orthogonality: {summary['hal_mean_lam']:.4f}")
        print(f"OK  mean LAM-orthogonality: {summary['ok_mean_lam']:.4f}")
        print(f"Best layer: L{best_l}  Δ={summary['delta_at_best_layer']:.4f}")
        if delta[best_l] < 0:
            print("Direction: OK > HAL (factual recall is MORE orthogonal — matches Gupta 2025)")
        else:
            print("Direction: HAL > OK (unexpected — note in paper)")

    out_summary = traj_dir / "lam_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_summary}")
    print(f"Per-sample _lam.json files written to {traj_dir}")


if __name__ == "__main__":
    main()
