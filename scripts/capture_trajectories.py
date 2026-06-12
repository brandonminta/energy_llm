#!/usr/bin/env python
"""Stage 2 — trajectory capture. Resume-by-default; shardable for SLURM
array jobs; traps SIGUSR1/SIGTERM for graceful preemption.

Usage:
    python scripts/capture_trajectories.py --config configs/e1.yaml \
        [--shard-index K --num-shards N] [--limit 5] [--max-samples-per-job M]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.banks import load_banks
from energy_llm.config import load_config, require_beta_star
from energy_llm.data import load_samples
from energy_llm.models import load_model_and_tokenizer
from energy_llm.trajectory import run_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="dry-run cap on samples per shard")
    parser.add_argument("--max-samples-per-job", type=int, default=None,
                        help="exit (resumable) after this many samples; for "
                             "short-walltime partitions with self-resubmit")
    parser.add_argument("--score-dtype", default=None,
                        help="override trajectory.score_dtype (e.g. bfloat16 "
                             "on memory-tight MIG slices)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    require_beta_star(cfg)
    if args.score_dtype:
        cfg.trajectory.score_dtype = args.score_dtype
    if not cfg.snapshot_path.exists():
        # E3 path: beta_star comes from the YAML (inherited from E1); every
        # run still gets a frozen snapshot.
        cfg.save_snapshot()

    banks = load_banks(cfg.banks_path)
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.alias, dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit, device=cfg.model.device,
    )
    samples = load_samples(cfg.dataset.name, num_samples=cfg.dataset.num_samples,
                           seed=cfg.seed)
    report = run_trajectories(
        cfg, model, tokenizer, banks, samples,
        shard_index=args.shard_index, num_shards=args.num_shards,
        limit=args.limit, max_samples_per_job=args.max_samples_per_job,
    )
    print(f"[trajectories] {report}")
    if report["stopped_by_signal"]:
        sys.exit(0)  # clean, resumable exit on preemption


if __name__ == "__main__":
    main()
