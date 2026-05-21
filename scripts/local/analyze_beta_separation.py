"""Analyse β calibration against the Ramsauer well-separation condition.

Ramsauer et al. (2020) Theorem 1 guarantees retrieval convergence in one step
when β > 1 / Δ_min, where Δ_min is the minimum L2-distance between any pair
of stored patterns divided by M² (the squared max-row-norm).

This script:
  1. Loads banks_*.pt
  2. For each layer, estimates Δ_min via random sampling of pattern pairs
  3. Computes the Ramsauer threshold β_min = 1 / Δ_min
  4. Compares against the calibrated β★ from calibration_beta.json
  5. Writes a comparison table and a plot

The exact Δ_min requires O(K²) pairs which is intractable for K=11008.
We use 10 000 random pairs per layer (sufficient for an order-of-magnitude
estimate of the separation regime).

Usage:
    python scripts/local/analyze_beta_separation.py \\
        --banks         outputs/<run>/banks_key_space_no_norm_q.pt \\
        --calibration   exp04_results/trajectories/calibration_beta.json \\
        --output        exp04_results/beta_separation.json \\
        --n-pairs       10000
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
    return {int(k): v.float() for k, v in raw.items() if k != "model"}


def _estimate_delta_min(
    bank: torch.Tensor,
    n_pairs: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate min and mean pairwise L2 distance between bank rows.

    Uses random sampling because K=11008 makes exhaustive O(K²) infeasible.

    Returns:
        (delta_min_est, delta_mean_est)
    """
    K = bank.shape[0]
    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, K, n_pairs)
    j_idx = rng.integers(0, K, n_pairs)
    same  = i_idx == j_idx
    i_idx[same] = (i_idx[same] + 1) % K

    diff  = bank[i_idx] - bank[j_idx]            # [n_pairs, d]
    dists = diff.norm(dim=1).numpy()              # [n_pairs]
    return float(dists.min()), float(dists.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--banks",       required=True, help="Path to banks_*.pt")
    parser.add_argument("--calibration", required=True,
                        help="Path to calibration_beta.json")
    parser.add_argument("--output",      default="beta_separation.json")
    parser.add_argument("--n-pairs",     type=int, default=10_000)
    args = parser.parse_args()

    banks  = _load_banks(Path(args.banks))
    calib  = json.loads(Path(args.calibration).read_text())
    beta_star = float(calib["beta_star"])

    print(f"Banks: {len(banks)} layers  |  β★ = {beta_star:.2f}")
    print(f"Estimating Δ_min from {args.n_pairs} random pairs per layer ...\n")

    results = []
    n_above = 0  # layers where β★ > Ramsauer threshold

    print(f"{'Layer':>6}  {'Δ_min':>10}  {'Δ_mean':>10}  {'M':>8}  "
          f"{'β_Ramsauer':>12}  {'β★/β_R':>9}  {'OK?':>5}")
    print("-" * 66)

    for l_idx in sorted(banks.keys()):
        bank = banks[l_idx]
        M    = float(bank.norm(dim=1).max())
        d_min, d_mean = _estimate_delta_min(bank, args.n_pairs)
        # Ramsauer condition: β > M² / Δ_min  (adapted for L2 distance)
        # In the original paper Δ is the minimum separation; the threshold
        # is β > 1/Δ_min for unit-norm patterns. For non-unit-norm: β > M²/Δ_min.
        beta_ramsauer = (M ** 2) / d_min if d_min > 1e-10 else float("inf")
        ratio = beta_star / beta_ramsauer if beta_ramsauer > 0 else 0.0
        ok = ratio > 1.0
        if ok:
            n_above += 1

        row = {
            "layer":         l_idx,
            "delta_min_est": round(d_min,  5),
            "delta_mean_est":round(d_mean, 5),
            "M":             round(M,      5),
            "beta_ramsauer": round(beta_ramsauer, 3),
            "beta_star":     beta_star,
            "ratio":         round(ratio, 3),
            "above_threshold": ok,
        }
        results.append(row)
        print(f"{l_idx:>6}  {d_min:>10.5f}  {d_mean:>10.5f}  {M:>8.4f}  "
              f"{beta_ramsauer:>12.2f}  {ratio:>9.3f}  {'YES' if ok else 'NO':>5}")

    n_layers = len(results)
    print(f"\n{n_above}/{n_layers} layers satisfy β★ > β_Ramsauer "
          f"(well-separation guarantee)")

    summary = {
        "beta_star":       beta_star,
        "n_layers":        n_layers,
        "n_above_threshold": n_above,
        "fraction_above":  round(n_above / n_layers, 3) if n_layers else 0,
        "n_pairs_sampled": args.n_pairs,
        "per_layer":       results,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {out_path}")

    if n_above < n_layers * 0.5:
        print("\nWARNING: β★ is below the Ramsauer threshold for most layers.")
        print("The calibrated β★ ≈ 68 targets H_norm = 0.55 (detection regime),")
        print("not the theoretical convergence guarantee. Discuss this gap in §3.")


if __name__ == "__main__":
    main()
