"""CLI entry point for hopfield_llm pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from hopfield_llm.pipeline.analysis import analyze_trajectories
from hopfield_llm.pipeline.stages import build_banks, label_trajectory, run_trajectory
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
    p.add_argument("--bank",    default="up",
                   choices=["gate", "up", "down_values",
                            "gate_proj", "up_proj", "fc1", "w1", "w3"],
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
    p.add_argument("--calibrate-beta", action="store_true", dest="calibrate_beta",
                   help="Auto-calibrate β before the run to hit 55%% of max entropy")
    p.add_argument("--calibration-samples", type=int, default=20, dest="calibration_samples",
                   help="Number of samples used for beta calibration (default: 20)")
    p.add_argument("--beta-target", type=float, default=0.55, dest="beta_target",
                   help="Target norm_entropy fraction for beta calibration (default: 0.55)")

    # ── Stage 3 — analysis ────────────────────────────────────────────────────
    p = subparsers.add_parser("analyze", help="Aggregate trajectories → analysis JSON (CPU-only)")
    p.add_argument("--trajectories", required=True, help="Directory with .npz/.json files")
    p.add_argument("--output",       required=True, help="Output path for analysis JSON")
    p.add_argument(
        "--score-metric", default="delta_energy", dest="score_metric",
        choices=[
            "delta_energy", "delta_norm_entropy", "delta_entropy",
            "delta_top_activation", "delta_lse",
            # Temporal metrics over generation tokens
            "energy_slope", "energy_var", "energy_spread",
            # Retrieval-confidence metric (requires saved scores)
            "top_act_gap_mean",
            # Logit-lens metrics (require enable_logit_lens=True at capture)
            "delta_logit_entropy", "gen_logit_entropy_mean", "gen_logit_top_prob_min",
            # Shuffled-bank null control
            "delta_energy_null",
            "energy_shift_l1", "energy_shift_l2", "peak_delta_layer",
        ],
    )
    p.add_argument("--score-aggregation", default="mean", dest="score_aggregation",
                   choices=["mean", "max"])
    p.add_argument("--signal-zone", type=int, nargs=2, default=None, dest="signal_zone",
                   metavar=("START", "END"))
    p.add_argument("--pool-over-answer", action="store_true", dest="pool_over_answer",
                   help="Pool generation metrics only over the gold-answer span "
                        "(falls back to content tokens). Loads a tokenizer.")
    p.add_argument("--tokenizer-id", default=None, dest="tokenizer_id",
                   help="Override the model alias used to load a tokenizer "
                        "for --pool-over-answer.")

    # ── Stage 4 — visualization ───────────────────────────────────────────────
    p = subparsers.add_parser("visualize", help="Generate plots from analysis JSON (CPU-only)")
    p.add_argument("--analysis", required=True)
    p.add_argument("--output",   required=True, help="Output directory for plots")

    # ── Stage 5 — labeling (heuristic + LLM judge) ────────────────────────────
    p = subparsers.add_parser(
        "label",
        help="Label each sample with F1 heuristic + LLM judge (CPU-only API or local HF)",
    )
    p.add_argument("--trajectories", required=True, help="Trajectory directory (output of run-trajectory)")
    p.add_argument("--output",       required=True, help="Directory for {id}_labeled.json + calibration.json")
    p.add_argument("--judge-model",  default="gpt-4o-mini", dest="judge_model")
    p.add_argument("--api-base",     default="https://api.openai.com/v1", dest="api_base",
                   help="OpenAI-compatible API base URL; pass empty string '' to use a local HF model")
    p.add_argument("--api-key",      default=None, dest="api_key",
                   help="API key (defaults to OPENAI_API_KEY env var)")
    p.add_argument("--heuristic-threshold", type=float, default=0.3, dest="heuristic_threshold")
    p.add_argument("--calibration-subset-size", type=int, default=50, dest="calibration_subset_size")
    p.add_argument("--temperature",  type=float, default=0.0)

    # ── Stage 6 — evaluation report ───────────────────────────────────────────
    p = subparsers.add_parser(
        "evaluate",
        help="Per-layer AUROC + logistic probe + baselines → HTML report (CPU-only)",
    )
    p.add_argument("--trajectories", required=True, help="Trajectory directory")
    p.add_argument("--labels",       default=None,
                   help="Label directory ({id}_label.json); defaults to --trajectories")
    p.add_argument("--output",       required=True, help="Output HTML path")
    p.add_argument("--beta",         type=float, default=15.0,
                   help="Beta used during capture (centres the post-hoc beta sweep)")
    p.add_argument("--dataset-name", default="dataset", dest="dataset_name",
                   help="Display name in the report header")

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

    # In probe mode build_banks writes banks_{label}.pt, not banks.pt.
    _probe_bank_paths = [run_dir / f"banks_{cfg.primary_probe.label}.pt"]
    if cfg.secondary_probe is not None:
        _probe_bank_paths.append(run_dir / f"banks_{cfg.secondary_probe.label}.pt")
    _banks_exist = banks_path.exists() or all(p.exists() for p in _probe_bank_paths)

    try:
        if args.skip_banks and _banks_exist:
            from hopfield_llm.utils.logging import get_logger
            get_logger("cli").info("Skipping bank extraction (--skip-banks)")
        else:
            build_banks(
                model=cfg.model_alias, output=str(banks_path),
                bank=cfg.bank, device=device, load_in_4bit=cfg.load_in_4bit,
                primary_probe=cfg.primary_probe,
                secondary_probe=cfg.secondary_probe,
            )
            tracker.log_metric("stage1_complete", 1.0)

        _, calibrated_beta = run_trajectory(
            model=cfg.model_alias, dataset=cfg.dataset_name,
            banks=str(banks_path), output=str(traj_dir),
            beta=cfg.beta, threshold=cfg.threshold, energy_mode=cfg.energy_mode,
            max_samples=cfg.max_samples, seed=cfg.seed,
            max_new_tokens=cfg.max_new_tokens, diagnostic_subset=cfg.diagnostic_subset,
            shard_id=shard_id, num_shards=cfg.num_shards,
            device=device, load_in_4bit=cfg.load_in_4bit,
            scores_subset_size=cfg.scores_subset_size,
            score_capture=cfg.score_capture,
            score_topk=cfg.score_topk,
            enable_logit_lens=cfg.enable_logit_lens,
            enable_null_control=cfg.enable_null_control,
            calibrate_beta=cfg.calibrate_beta,
            calibration_samples=cfg.calibration_samples,
            beta_target=cfg.beta_target,
            primary_probe=cfg.primary_probe,
            secondary_probe=cfg.secondary_probe,
        )
        if calibrated_beta is not None:
            tracker.patch_config({"trajectory": {"beta": calibrated_beta}})
            tracker.log_metric("calibrated_beta", calibrated_beta)
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
            calibrate_beta=args.calibrate_beta,
            calibration_samples=args.calibration_samples,
            beta_target=args.beta_target,
        )
    elif args.command == "analyze":
        signal_zone = tuple(args.signal_zone) if args.signal_zone else None
        analyze_trajectories(
            traj_dir=args.trajectories, output=args.output,
            score_metric=args.score_metric, score_aggregation=args.score_aggregation,
            signal_zone=signal_zone,
            pool_over_answer=args.pool_over_answer,
            tokenizer_id=args.tokenizer_id,
        )
    elif args.command == "visualize":
        from hopfield_llm.visualization.plots import visualize_analysis
        visualize_analysis(args.analysis, args.output)
    elif args.command == "label":
        import os
        from hopfield_llm.labeling.llm_judge import LLMJudge
        api_base = args.api_base if args.api_base else None
        judge = LLMJudge(
            model_id=args.judge_model,
            api_base=api_base,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            temperature=args.temperature,
        )
        label_trajectory(
            traj_dir=args.trajectories,
            output=args.output,
            judge=judge,
            heuristic_threshold=args.heuristic_threshold,
            calibration_subset_size=args.calibration_subset_size,
        )
    elif args.command == "evaluate":
        from hopfield_llm.evaluation.report import build_eval_report
        build_eval_report(
            artifact_dir=Path(args.trajectories),
            labels_dir=Path(args.labels or args.trajectories),
            out_path=Path(args.output),
            beta=args.beta,
            dataset_name=args.dataset_name,
        )
    elif args.command == "run-experiment":
        return _run_experiment(args)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
