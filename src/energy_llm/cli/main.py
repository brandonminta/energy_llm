from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from energy_llm.datasets.truthfulqa import load_truthfulqa
from energy_llm.experiments.exp1_factual_vs_hallucinated import run_experiment_1
from energy_llm.memory.mlp_bank import extract_mlp_memory_bank
from energy_llm.models.hf_llm import HFLLM


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _resolve_config_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def _ensure_not_placeholder(config: dict[str, Any], path: Path) -> None:
    if config.get("status") == "placeholder":
        raise ValueError(f"Config {path} is a documented placeholder and is not runnable.")


def _load_experiment_bundle(
    experiment_config_path: Path,
    model_override: str | None,
    dataset_override: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_root = Path.cwd()
    experiment_config = _read_yaml(experiment_config_path)
    _ensure_not_placeholder(experiment_config, experiment_config_path)

    model_path = _resolve_config_path(
        model_override or experiment_config["model_config"],
        repo_root,
    )
    dataset_path = _resolve_config_path(
        dataset_override or experiment_config["dataset_config"],
        repo_root,
    )

    model_config = _read_yaml(model_path)
    dataset_config = _read_yaml(dataset_path)
    _ensure_not_placeholder(model_config, model_path)
    _ensure_not_placeholder(dataset_config, dataset_path)
    return experiment_config, model_config, dataset_config


def _run_truthfulqa_exp1(
    experiment_config: dict[str, Any],
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    beta_override: float | None,
    max_samples_override: int | None,
    output_dir_override: str | None,
) -> Path:
    if dataset_config.get("loader") != "truthfulqa":
        raise ValueError("The current CLI baseline only supports loader='truthfulqa'.")

    llm = HFLLM(
        model_id=model_config["model_alias"],
        device=model_config.get("device"),
        load_in_4bit=model_config.get("load_in_4bit", True),
        trust_remote_code=model_config.get("trust_remote_code", True),
        output_hidden_states=model_config.get("output_hidden_states", True),
    )

    bank_cfg = experiment_config.get("bank", {})
    banks, bank_metadata = extract_mlp_memory_bank(
        llm,
        bank=bank_cfg.get("name", "gate"),
        normalize=bank_cfg.get("normalize", False),
        strict=bank_cfg.get("strict", True),
        return_metadata=True,
    )

    dataset_max_samples = max_samples_override
    if dataset_max_samples is None:
        dataset_max_samples = dataset_config.get("max_samples")

    samples = load_truthfulqa(
        factual_strategy=dataset_config.get("factual_strategy", "best"),
        hallucinated_strategy=dataset_config.get("hallucinated_strategy", "random_incorrect"),
        categories=dataset_config.get("categories"),
        max_samples=dataset_max_samples,
        seed=dataset_config.get("seed", 42),
        verbose=dataset_config.get("verbose", True),
    )

    repr_cfg = experiment_config.get("representation", {})
    divergence_cfg = experiment_config.get("divergence", {})
    score_cfg = experiment_config.get("score", {})
    runtime_cfg = experiment_config.get("runtime", {})

    beta = beta_override if beta_override is not None else divergence_cfg.get("beta", 15.0)

    results, df = run_experiment_1(
        llm=llm,
        banks=banks,
        samples=samples,
        pooling=repr_cfg.get("pooling", "mean"),
        layers=repr_cfg.get("layers"),
        beta=beta,
        energy_mode=divergence_cfg.get("energy_mode", "cosine"),
        active_threshold=divergence_cfg.get("active_threshold", 0.1),
        separator=repr_cfg.get("separator", "\n"),
        signal_zone=score_cfg.get("signal_zone"),
        score_metric=score_cfg.get("metric", "js"),
        score_aggregation=score_cfg.get("aggregation", "mean"),
        score_use_absolute=score_cfg.get("use_absolute", False),
        skip_on_error=runtime_cfg.get("skip_on_error", True),
        verbose=runtime_cfg.get("verbose", True),
    )

    output_dir = Path(
        output_dir_override
        or experiment_config.get("output_dir")
        or f"runs/exp01/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)

    clean_bank_metadata = {
        str(layer_idx): {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in layer_info.items()
        }
        for layer_idx, layer_info in bank_metadata.items()
    }

    score_series = df["score"].dropna() if "score" in df.columns else np.array([])
    summary = {
        "experiment_name": experiment_config.get("name", "exp01_truthfulqa_baseline"),
        "model_alias": model_config["model_alias"],
        "model_id": llm.model_id,
        "n_samples": len(samples),
        "n_results": len(results),
        "n_success": int(sum(1 for r in results if r.error is None)),
        "n_failed": int(sum(1 for r in results if r.error is not None)),
        "score_mean": None if len(score_series) == 0 else float(score_series.mean()),
        "score_std": None if len(score_series) == 0 else float(score_series.std(ddof=0)),
        "beta": beta,
        "bank_name": bank_cfg.get("name", "gate"),
        "pooling": repr_cfg.get("pooling", "mean"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "bank_metadata.json").write_text(
        json.dumps(clean_bank_metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "experiment": experiment_config,
                "model": model_config,
                "dataset": dataset_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[energy_llm] Saved run outputs to {output_dir}")
    if len(score_series) != 0:
        print(
            "[energy_llm] Score mean/std: "
            f"{float(score_series.mean()):.4f} / {float(score_series.std(ddof=0)):.4f}"
        )

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="energy_llm command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_exp1 = subparsers.add_parser(
        "run-exp1",
        help="Run the implemented TruthfulQA factual-vs-hallucinated baseline.",
    )
    run_exp1.add_argument(
        "--config",
        default="configs/experiments/exp01_truthfulqa_baseline.yaml",
        help="Path to the experiment YAML config.",
    )
    run_exp1.add_argument(
        "--model-config",
        default=None,
        help="Optional model config override.",
    )
    run_exp1.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config override.",
    )
    run_exp1.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Optional beta override for divergence scoring.",
    )
    run_exp1.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional dataset sample cap for smoke tests or small runs.",
    )
    run_exp1.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-exp1":
        experiment_config_path = Path(args.config)
        experiment_config, model_config, dataset_config = _load_experiment_bundle(
            experiment_config_path=experiment_config_path,
            model_override=args.model_config,
            dataset_override=args.dataset_config,
        )
        _run_truthfulqa_exp1(
            experiment_config=experiment_config,
            model_config=model_config,
            dataset_config=dataset_config,
            beta_override=args.beta,
            max_samples_override=args.max_samples,
            output_dir_override=args.output_dir,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
