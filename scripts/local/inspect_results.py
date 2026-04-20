"""Inspect trajectory artifacts produced by run-trajectory.

Usage:
    python scripts/inspect_results.py runs/local_test/qwen25_1_5b_truthfulqa/trajectories
    python scripts/inspect_results.py <traj_dir> --samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────


def load_sample(traj_dir: Path, sample_id: str) -> tuple[dict, dict]:
    meta = json.loads((traj_dir / f"{sample_id}.json").read_text())
    npz  = dict(np.load(traj_dir / f"{sample_id}.npz"))
    return meta, npz


def finite_mean(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(v.mean()) if v.size else float("nan")


def layer_profile(arr: np.ndarray) -> str:
    """Compact per-layer mean string, e.g. '0.12 0.34 ...' (first 8 layers)."""
    show = arr[:8] if arr.ndim == 1 else arr[:, :].mean(axis=1)[:8]
    return "  ".join(f"{v:.3f}" for v in show)


# ── main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Inspect hopfield-llm trajectory artifacts")
    ap.add_argument("traj_dir", help="Directory produced by run-trajectory")
    ap.add_argument("--samples", type=int, default=3,
                    help="Number of sample details to print (default 3)")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.is_dir():
        sys.exit(f"Directory not found: {traj_dir}")

    json_files  = sorted(traj_dir.glob("*.json"))
    npz_files   = sorted(traj_dir.glob("*.npz"))
    metric_npz  = [f for f in npz_files if not f.stem.endswith("_hpre")]
    hpre_npz    = [f for f in npz_files if f.stem.endswith("_hpre")]

    if not json_files:
        sys.exit("No .json files found. Did run-trajectory complete?")

    # ── dataset-level summary ────────────────────────────────────────
    print("=" * 60)
    print(f"  Trajectory directory: {traj_dir}")
    print(f"  Samples:              {len(json_files)}")
    print(f"  Metric .npz files:    {len(metric_npz)}")
    print(f"  Diagnostic h_pre:     {len(hpre_npz)}")
    print("=" * 60)

    # Load all samples for aggregate stats
    all_prefill_energy:   list[np.ndarray] = []
    all_gen_energy:       list[np.ndarray] = []
    all_n_tokens:         list[int]        = []
    all_n_layers:         list[int]        = []
    missing_or_bad:       list[str]        = []

    for jf in json_files:
        sid = jf.stem
        npz_path = traj_dir / f"{sid}.npz"
        if not npz_path.exists():
            missing_or_bad.append(sid)
            continue
        try:
            meta, npz = load_sample(traj_dir, sid)
            all_prefill_energy.append(npz["prefill_energy"])   # [L]
            all_gen_energy.append(npz["gen_energy"])            # [L, T]
            all_n_tokens.append(int(meta["n_tokens_generated"]))
            all_n_layers.append(int(meta["n_layers"]))
        except Exception as exc:
            missing_or_bad.append(f"{sid} ({exc})")

    if missing_or_bad:
        print(f"\n  WARNING: {len(missing_or_bad)} samples could not be loaded:")
        for s in missing_or_bad[:5]:
            print(f"    {s}")

    # Consistency check
    n_ok = len(all_n_tokens)
    if n_ok == 0:
        sys.exit("No valid samples found.")

    L = all_n_layers[0]
    print(f"\n  Model / layers:       {L}")
    print(f"  Valid samples:        {n_ok} / {len(json_files)}")

    # Token stats
    tok = np.array(all_n_tokens)
    print(f"\n  Generated tokens:")
    print(f"    mean={tok.mean():.1f}  min={tok.min()}  max={tok.max()}")

    # Energy stats (prefill, mean across samples per layer)
    pe = np.vstack(all_prefill_energy)       # [N, L]
    pe_finite = np.where(np.isfinite(pe), pe, np.nan)
    layer_mean = np.nanmean(pe_finite, axis=0)   # [L]
    print(f"\n  Prefill energy (layer mean, first 8 layers):")
    print(f"    {layer_profile(layer_mean)}")
    print(f"    overall finite mean = {np.nanmean(layer_mean):.4f}")

    # Check for all-NaN layers (hook missed)
    nan_layers = [i for i in range(L) if np.all(np.isnan(pe_finite[:, i]))]
    if nan_layers:
        print(f"\n  WARNING: {len(nan_layers)} layers are all-NaN (hook may have missed them)")
        print(f"    layers: {nan_layers[:10]}")
    else:
        print(f"  All {L} layers captured cleanly.")

    # ── per-sample detail ────────────────────────────────────────────
    n_show = min(args.samples, len(json_files))
    print(f"\n{'─'*60}")
    print(f"  Sample details ({n_show} of {len(json_files)})")
    print(f"{'─'*60}")

    for jf in json_files[:n_show]:
        sid = jf.stem
        npz_path = traj_dir / f"{sid}.npz"
        if not npz_path.exists():
            continue
        try:
            meta, npz = load_sample(traj_dir, sid)
        except Exception as exc:
            print(f"\n  [{sid}]  ERROR: {exc}")
            continue

        T = len(npz["token_ids"])
        pe_mean = finite_mean(npz["prefill_energy"])
        ge_mean = finite_mean(npz["gen_energy"])

        print(f"\n  [{sid}]")
        print(f"    source:    {meta['source']}")
        print(f"    question:  {meta['question'][:80]}")
        print(f"    generated: {meta['generated_text'][:120]}")
        gold_str = " | ".join(meta["gold_answers"][:3])
        if len(meta["gold_answers"]) > 3:
            gold_str += f"  (+{len(meta['gold_answers'])-3} more)"
        print(f"    gold:      {gold_str[:100]}")
        print(f"    tokens generated: {T}")
        print(f"    prefill energy (mean over layers): {pe_mean:.4f}")
        print(f"    gen energy     (mean over L×T):   {ge_mean:.4f}")

        # Check shapes
        expected_shapes = {
            "prefill_energy": (L,),
            "gen_energy":     (L, T),
            "token_ids":      (T,),
        }
        shape_ok = all(
            npz[k].shape == expected_shapes[k]
            for k in expected_shapes
        )
        print(f"    shapes OK: {shape_ok}")

    print(f"\n{'='*60}")
    print("  Inspection complete.")
    if n_ok == len(json_files):
        print("  Status: ALL samples look good — ready for HPC.")
    else:
        print(f"  Status: {len(json_files)-n_ok} samples failed — check logs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
