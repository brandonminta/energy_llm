"""CLI entry point for hopfield_llm pipeline stages."""

from __future__ import annotations

import argparse

from hopfield_llm.pipeline.stages import build_banks, run_trajectory
from hopfield_llm.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hopfield_llm pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage 1 — bank extraction
    p = subparsers.add_parser(
        "build-banks",
        help="Extract W_down[l] for all layers and save as a .pt artifact",
    )
    p.add_argument("--model", required=True, help="Model alias or HuggingFace model ID")
    p.add_argument("--output", required=True, help="Output path for banks .pt file")
    p.add_argument("--device", default=None)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # Stage 2 — trajectory collection
    p = subparsers.add_parser(
        "run-trajectory",
        help="Prefill + generation passes for each sample; save .npz and .json",
    )
    p.add_argument("--model", required=True, help="Model alias or HuggingFace model ID")
    p.add_argument(
        "--dataset", required=True,
        choices=["triviaqa", "nq", "truthfulqa"],
        help="Dataset to run",
    )
    p.add_argument("--banks", required=True, help="Path to banks .pt artifact")
    p.add_argument("--output", required=True, help="Output directory for artifacts")
    p.add_argument("--beta", type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument(
        "--energy-mode", default="dot", choices=["dot", "cosine"],
        dest="energy_mode",
    )
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=50, dest="max_new_tokens")
    p.add_argument(
        "--diagnostic-subset", type=int, default=20, dest="diagnostic_subset",
        help="Number of samples for which raw h_pre tensors are saved",
    )
    p.add_argument(
        "--shard-id", type=int, default=0, dest="shard_id",
        help="Shard index for this worker (0-indexed). Use $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument(
        "--num-shards", type=int, default=1, dest="num_shards",
        help="Total number of shards. 1 = no sharding (process all samples).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-banks":
        build_banks(
            model=args.model,
            output=args.output,
            device=args.device,
            load_in_4bit=not args.no_4bit,
        )
    elif args.command == "run-trajectory":
        run_trajectory(
            model=args.model,
            dataset=args.dataset,
            banks=args.banks,
            output=args.output,
            beta=args.beta,
            threshold=args.threshold,
            energy_mode=args.energy_mode,
            max_samples=args.max_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            diagnostic_subset=args.diagnostic_subset,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            device=args.device,
            load_in_4bit=not args.no_4bit,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
