"""Compute logit-based baselines for all trajectory samples.

Runs a teacher-forced forward pass per sample to extract:
  seq_logprob_mean   — mean log-probability over generated tokens
  token_entropy_mean — mean token-level entropy
  token_entropy_max  — max token-level entropy

Writes {sample_id}_baselines.json next to each trajectory file.
Safe to interrupt and re-run: already-computed samples are skipped.

Usage:
    python scripts/local/run_baselines.py \\
        --traj-dir exp04_results/trajectories \\
        --model qwen25_3b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute seq_logprob/token_entropy baselines for trajectory samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--traj-dir", required=True,
                   help="Directory containing {sample_id}.json / .npz files")
    p.add_argument("--model", default=None,
                   help="Model alias (e.g. qwen25_3b). Defaults to the value stored in "
                        "the first trajectory .json file.")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantisation (use full precision / fp16)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stop after N samples (useful for quick tests)")
    p.add_argument("--device", default="cuda",
                   help="PyTorch device string")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        sys.exit(f"ERROR: trajectory directory not found: {traj_dir}")

    # Collect sample IDs
    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(tag in f.name for tag in ("calibration", "_label", "_baselines",
                                              "_scores", "_hpre"))
    )
    if not sample_files:
        sys.exit(f"ERROR: no trajectory .json files found in {traj_dir}")
    if args.max_samples:
        sample_files = sample_files[:args.max_samples]

    # Resolve model alias from first file if not specified
    model_alias = args.model
    if model_alias is None:
        meta0 = json.loads(sample_files[0].read_text())
        model_alias = meta0.get("model", "qwen25_3b")
        print(f"Model alias not specified; using '{model_alias}' from trajectory metadata.")

    # Import project code
    try:
        from hopfield_llm.models.loader import HFLLM
        from hopfield_llm.datasets.base import DataSample
        from hopfield_llm.evaluation.baselines import compute_logit_baselines, save_baselines
    except ImportError:
        sys.exit("ERROR: hopfield_llm not installed.  Run:  pip install -e .")

    print(f"\nLoading model '{model_alias}'  (4bit={not args.no_4bit}, device={args.device})")
    llm = HFLLM(
        model_alias,
        device=args.device,
        load_in_4bit=not args.no_4bit,
    )
    llm.model.eval()
    print(f"Model loaded.\n")

    total   = len(sample_files)
    done    = 0
    skipped = 0
    failed  = 0

    print(f"{'':>6}  {'sample_id':<40}  {'seq_lp':>8}  {'H_mean':>8}  {'H_max':>8}")
    print("-" * 76)

    for i, jf in enumerate(sample_files, 1):
        try:
            meta = json.loads(jf.read_text())
        except Exception as exc:
            print(f"  [{i:>4}/{total}]  SKIP (bad JSON: {exc})")
            failed += 1
            continue

        sample_id     = meta.get("id", jf.stem)
        baseline_path = traj_dir / f"{sample_id}_baselines.json"

        if baseline_path.exists():
            skipped += 1
            continue

        sample = DataSample(
            id=sample_id,
            source=meta.get("source", ""),
            question=meta.get("question", ""),
            gold_answers=meta.get("gold_answers", []),
            generated_text=meta.get("generated_text", ""),
        )

        try:
            bl = compute_logit_baselines(llm, sample)
        except Exception as exc:
            print(f"  [{i:>4}/{total}]  ERR  {sample_id}  ({exc})")
            failed += 1
            continue

        bl["id"] = sample_id
        save_baselines(bl, traj_dir, sample_id)
        done += 1

        print(
            f"  [{i:>4}/{total}]  {sample_id:<40}"
            f"  {bl['seq_logprob_mean']:>8.4f}"
            f"  {bl['token_entropy_mean']:>8.4f}"
            f"  {bl['token_entropy_max']:>8.4f}"
        )

    print(f"\nDone={done}  Skipped={skipped}  Failed={failed}  Total={total}")


if __name__ == "__main__":
    main()
