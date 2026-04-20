"""CLI entry point for hopfield_llm pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from hopfield_llm.pipeline.analysis import analyze_trajectories
from hopfield_llm.pipeline.stages import build_banks, run_trajectory
from hopfield_llm.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="hopfield_llm pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Stage 1 — bank extraction ─────────────────────────────────────────────
    p = subparsers.add_parser("build-banks", help="Extract MLP weight banks → .pt")
    p.add_argument("--model",   required=True, help="Model alias or HuggingFace model ID")
    p.add_argument("--output",  required=True, help="Output path for banks .pt file")
    p.add_argument("--bank",    default="down",
                   choices=["gate", "up", "down", "gate_plus_up", "gate_up_concat"],
                   help="MLP projection to use as the memory bank")
    p.add_argument("--device",  default=None)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # ── Stage 2 — trajectory collection ──────────────────────────────────────
    p = subparsers.add_parser(
        "run-trajectory",
        help="Prefill + generation passes per sample; save .npz and .json",
    )
    p.add_argument("--model",    required=True)
    p.add_argument("--dataset",  required=True, choices=["triviaqa", "nq", "truthfulqa"])
    p.add_argument("--banks",    required=True, help="Path to banks .pt artifact")
    p.add_argument("--output",   required=True, help="Output directory for artifacts")
    p.add_argument("--beta",     type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--energy-mode", default="dot", choices=["dot", "cosine"], dest="energy_mode")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=50, dest="max_new_tokens")
    p.add_argument("--diagnostic-subset", type=int, default=20, dest="diagnostic_subset")
    p.add_argument("--shard-id",  type=int, default=0, dest="shard_id")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--device",   default=None)
    p.add_argument("--no-4bit",  action="store_true")

    # ── Stage 3 — analysis ────────────────────────────────────────────────────
    p = subparsers.add_parser("analyze", help="Aggregate trajectories → analysis JSON (CPU-only)")
    p.add_argument("--trajectories", required=True, help="Directory with .npz/.json files")
    p.add_argument("--output",       required=True, help="Output path for analysis JSON")
    p.add_argument(
        "--score-metric", default="js", dest="score_metric",
        choices=["js", "kl_fwd", "kl_rev", "hellinger",
                 "delta_energy", "delta_norm_entropy", "delta_entropy",
                 "delta_top_activation", "delta_mean_activation"],
    )
    p.add_argument("--score-aggregation", default="mean", dest="score_aggregation",
                   choices=["mean", "max", "weighted"])
    p.add_argument("--signal-zone", type=int, nargs=2, default=None, dest="signal_zone",
                   metavar=("START", "END"))

    # ── Stage 4 — visualization ───────────────────────────────────────────────
    p = subparsers.add_parser("visualize", help="Generate plots from analysis JSON (CPU-only)")
    p.add_argument("--analysis", required=True)
    p.add_argument("--output",   required=True, help="Output directory for plots")

    # ── Convenience — run full experiment from config ─────────────────────────
    p = subparsers.add_parser("run-experiment",
                               help="Run all pipeline stages from an experiment YAML")
    p.add_argument("--config",   required=True, help="Path to experiment YAML config")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values (e.g. --override beta=20.0 seed=0)")
    p.add_argument("--device",   default=None)
    p.add_argument("--no-4bit",  action="store_true")
    p.add_argument("--shard-id", type=int, default=None, dest="shard_id")
    p.add_argument("--num-shards", type=int, default=None, dest="num_shards")
    p.add_argument("--skip-banks", action="store_true", dest="skip_banks",
                   help="Skip bank extraction (reuse existing banks.pt in run dir)")

    return parser


def _run_experiment(args: argparse.Namespace) -> int:
    from hopfield_llm.pipeline.config import apply_overrides, load_experiment_config
    from hopfield_llm.utils.env import load_env_config
    from hopfield_llm.utils.tracking import ExperimentTracker
    from hopfield_llm.visualization.plots import visualize_analysis

    cfg = load_experiment_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
    if args.no_4bit:
        cfg.load_in_4bit = False
    if args.num_shards is not None:
        cfg.num_shards = args.num_shards

    env = load_env_config()
    device = args.device or env.device
    shard_id = args.shard_id if args.shard_id is not None else 0

    tracker = ExperimentTracker(base_dir=Path(cfg.base_dir) / cfg.name)
    run_dir = tracker.start(cfg.to_dict())

    banks_path    = run_dir / "banks.pt"
    traj_dir      = run_dir / "trajectories"
    analysis_path = run_dir / "analysis.json"
    plots_dir     = run_dir / "plots"

    try:
        if args.skip_banks and banks_path.exists():
            from hopfield_llm.utils.logging import get_logger
            get_logger("cli").info("Skipping bank extraction (--skip-banks)")
        else:
            build_banks(
                model=cfg.model_alias, output=str(banks_path),
                bank=cfg.bank, device=device, load_in_4bit=cfg.load_in_4bit,
            )
            tracker.log_metric("stage1_complete", 1.0)

        run_trajectory(
            model=cfg.model_alias, dataset=cfg.dataset_name,
            banks=str(banks_path), output=str(traj_dir),
            beta=cfg.beta, threshold=cfg.threshold, energy_mode=cfg.energy_mode,
            max_samples=cfg.max_samples, seed=cfg.seed,
            max_new_tokens=cfg.max_new_tokens, diagnostic_subset=cfg.diagnostic_subset,
            shard_id=shard_id, num_shards=cfg.num_shards,
            device=device, load_in_4bit=cfg.load_in_4bit,
        )
        tracker.log_metric("stage2_complete", 1.0)

        if cfg.num_shards <= 1:
            analyze_trajectories(
                traj_dir=str(traj_dir), output=str(analysis_path),
                score_metric=cfg.score_metric, score_aggregation=cfg.score_aggregation,
                signal_zone=cfg.signal_zone,
            )
            tracker.log_metric("stage3_complete", 1.0)

            visualize_analysis(str(analysis_path), str(plots_dir))
            tracker.log_metric("stage4_complete", 1.0)
        else:
            from hopfield_llm.utils.logging import get_logger
            get_logger("cli").info(
                "Multi-shard mode (shard %d/%d): skipping analysis. "
                "Run 'hopfield-llm analyze' after all shards complete.",
                shard_id, cfg.num_shards,
            )
    finally:
        tracker.finish()

    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-banks":
        build_banks(
            model=args.model, output=args.output,
            bank=args.bank, device=args.device, load_in_4bit=not args.no_4bit,
        )
    elif args.command == "run-trajectory":
        run_trajectory(
            model=args.model, dataset=args.dataset, banks=args.banks,
            output=args.output, beta=args.beta, threshold=args.threshold,
            energy_mode=args.energy_mode, max_samples=args.max_samples,
            seed=args.seed, max_new_tokens=args.max_new_tokens,
            diagnostic_subset=args.diagnostic_subset,
            shard_id=args.shard_id, num_shards=args.num_shards,
            device=args.device, load_in_4bit=not args.no_4bit,
        )
    elif args.command == "analyze":
        signal_zone = tuple(args.signal_zone) if args.signal_zone else None
        analyze_trajectories(
            traj_dir=args.trajectories, output=args.output,
            score_metric=args.score_metric, score_aggregation=args.score_aggregation,
            signal_zone=signal_zone,
        )
    elif args.command == "visualize":
        from hopfield_llm.visualization.plots import visualize_analysis
        visualize_analysis(args.analysis, args.output)
    elif args.command == "run-experiment":
        return _run_experiment(args)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
