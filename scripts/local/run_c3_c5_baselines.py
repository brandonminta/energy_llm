"""Run C3 (Semantic Entropy) and C5 (INTRA) baselines on a trajectory directory.

Both require GPU.  C3 also requires an NLI model (default: facebook/bart-large-mnli,
loaded on CPU to save VRAM — move to 'cuda' with --nli-device cuda if memory allows).

Usage:
    # Run both:
    python scripts/local/run_c3_c5_baselines.py \\
        --traj-dir exp04_results/trajectories \\
        --model qwen25_3b

    # INTRA only:
    python scripts/local/run_c3_c5_baselines.py \\
        --traj-dir exp04_results/trajectories \\
        --model qwen25_3b --skip-sementropy

    # Semantic entropy only:
    python scripts/local/run_c3_c5_baselines.py \\
        --traj-dir exp04_results/trajectories \\
        --model qwen25_3b --skip-intra
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--traj-dir",     required=True)
    p.add_argument("--model",        default=None,
                   help="Model alias (inferred from trajectory metadata if omitted)")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--no-4bit",      action="store_true")
    p.add_argument("--max-samples",  type=int, default=None)
    # C5 options
    p.add_argument("--skip-intra",   action="store_true",
                   help="Skip INTRA hidden-state feature extraction")
    # C3 options
    p.add_argument("--skip-sementropy", action="store_true",
                   help="Skip semantic entropy computation")
    p.add_argument("--n-completions",   type=int,   default=10)
    p.add_argument("--temperature",     type=float, default=0.5)
    p.add_argument("--max-new-tokens",  type=int,   default=50)
    p.add_argument("--nli-model",       default="facebook/bart-large-mnli")
    p.add_argument("--nli-device",      default="cpu",
                   help="Device for NLI model (cpu or cuda). Keep on cpu if VRAM is tight.")
    p.add_argument("--nli-threshold",   type=float, default=0.5,
                   help="Minimum entailment probability for bidirectional NLI check")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        sys.exit(f"ERROR: trajectory directory not found: {traj_dir}")

    # Resolve model alias
    model_alias = args.model
    if model_alias is None:
        sample_files = sorted(
            f for f in traj_dir.glob("*.json")
            if not any(t in f.name for t in ("calibration", "_label", "_baselines"))
        )
        if sample_files:
            meta0 = json.loads(sample_files[0].read_text())
            model_alias = meta0.get("model", "qwen25_3b")
        else:
            model_alias = "qwen25_3b"
        print(f"Model alias: '{model_alias}' (from trajectory metadata)")

    # Import project code
    try:
        from hopfield_llm.models.loader import HFLLM
        from hopfield_llm.evaluation.intra import run_intra_batch
        from hopfield_llm.evaluation.semantic_entropy import (
            SemanticEntropyScorer, run_semantic_entropy_batch,
        )
    except ImportError:
        sys.exit("ERROR: hopfield_llm not installed. Run:  pip install -e .")

    # Load LLM (needed for both C3 and C5)
    if not (args.skip_intra and args.skip_sementropy):
        print(f"\nLoading model '{model_alias}' (4bit={not args.no_4bit}, device={args.device})")
        llm = HFLLM(
            model_alias,
            device=args.device,
            load_in_4bit=not args.no_4bit,
        )
        llm.model.eval()
        print("Model loaded.\n")
    else:
        llm = None

    # ── C5: INTRA ──────────────────────────────────────────────────────────
    if not args.skip_intra:
        print("=" * 56)
        print("C5 — INTRA: extracting per-layer hidden states")
        print("=" * 56)
        n_done = run_intra_batch(
            llm, traj_dir,
            max_samples=args.max_samples,
        )
        print(f"INTRA done: {n_done} samples processed.\n")

    # ── C3: Semantic Entropy ────────────────────────────────────────────────
    if not args.skip_sementropy:
        print("=" * 56)
        print(f"C3 — Semantic Entropy: n={args.n_completions}, T={args.temperature}")
        print(f"     NLI model: {args.nli_model} on {args.nli_device}")
        print("=" * 56)
        scorer = SemanticEntropyScorer(
            nli_model_id=args.nli_model,
            device=args.nli_device,
            entailment_threshold=args.nli_threshold,
        )
        summaries = run_semantic_entropy_batch(
            llm, scorer, traj_dir,
            n=args.n_completions,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
        if summaries:
            import numpy as np
            entropies = [s["semantic_entropy"] for s in summaries
                         if not (s["semantic_entropy"] != s["semantic_entropy"])]
            print(f"\nSemantic entropy summary:")
            print(f"  n={len(entropies)}  "
                  f"mean={np.mean(entropies):.3f}  "
                  f"median={np.median(entropies):.3f}  "
                  f"max={max(entropies):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
