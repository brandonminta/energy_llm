"""CLI entry point for hopfield_llm pipeline stages."""

from __future__ import annotations

import argparse

from hopfield_llm.pipeline.stages import (
    analyze_scores,
    build_memory_bank,
    extract_hidden,
    score_hidden_states,
    visualize,
)
from hopfield_llm.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hopfield_llm pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage 1
    p = subparsers.add_parser("build-memory", help="Extract MLP memory banks")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bank", default="gate")
    p.add_argument("--device", default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--non-strict", action="store_true")

    # Stage 2
    p = subparsers.add_parser("extract-hidden", help="Extract hidden states")
    p.add_argument("--memory", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pooling", default="mean", choices=["mean", "last", "max", "first"])
    p.add_argument("--separator", default="\n")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--device", default=None)

    # Stage 3
    p = subparsers.add_parser("score", help="Compute Hopfield energy scores")
    p.add_argument("--memory", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--beta", type=float, default=15.0)
    p.add_argument("--energy-mode", default="cosine", choices=["cosine", "dot"])
    p.add_argument("--active-threshold", type=float, default=0.1)
    p.add_argument("--score-metric", default="js",
                   choices=["kl_fwd", "kl_rev", "js", "hellinger",
                            "delta_entropy", "delta_norm_entropy", "delta_energy",
                            "delta_top_activation", "delta_mean_activation"])
    p.add_argument("--score-aggregation", default="mean", choices=["mean", "max", "weighted"])
    p.add_argument("--score-use-absolute", action="store_true")

    # Stage 4
    p = subparsers.add_parser("analyze", help="Summarize saved scores")
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)

    # Stage 5
    p = subparsers.add_parser("visualize", help="Render plots from analysis JSON")
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-memory":
        build_memory_bank(
            model=args.model, output=args.output, bank=args.bank,
            normalize=args.normalize, strict=not args.non_strict,
            device=args.device, load_in_4bit=not args.no_4bit,
        )
    elif args.command == "extract-hidden":
        extract_hidden(
            memory=args.memory, prompts=args.prompts, output=args.output,
            pooling=args.pooling, separator=args.separator,
            layers=args.layers, device=args.device,
        )
    elif args.command == "score":
        score_hidden_states(
            memory=args.memory, hidden=args.hidden, output=args.output,
            beta=args.beta, energy_mode=args.energy_mode,
            active_threshold=args.active_threshold,
            score_metric=args.score_metric,
            score_aggregation=args.score_aggregation,
            score_use_absolute=args.score_use_absolute,
        )
    elif args.command == "analyze":
        analyze_scores(scores=args.scores, output=args.output)
    elif args.command == "visualize":
        visualize(analysis=args.analysis, output=args.output)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
