#!/usr/bin/env python
"""beta* calibration — warm-up prefill pass on N_c training-split questions,
writes beta_star into the config snapshot read by all later stages.

Usage:
    python scripts/calibrate_beta.py --config configs/e1.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.banks import load_banks
from energy_llm.calibration import calibrate_beta
from energy_llm.config import load_config
from energy_llm.data import load_samples
from energy_llm.models import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true",
                        help="recalibrate even if beta_star is already set")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    if cfg.calibration.beta_star is not None and not args.force:
        print(f"[calibrate] beta* already calibrated: {cfg.calibration.beta_star} "
              f"(snapshot {cfg.snapshot_path}); use --force to redo")
        return

    banks = load_banks(cfg.banks_path)
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.alias, dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit, device=cfg.model.device,
    )
    samples = load_samples(cfg.dataset.name, num_samples=cfg.dataset.num_samples,
                           seed=cfg.seed)
    record = calibrate_beta(cfg, model, tokenizer, banks, samples)
    print(f"[calibrate] beta* = {record['beta_star']:.4f}  clamped={record['clamped']}")
    print(f"[calibrate] curve: "
          f"{dict(zip(record['beta_grid'], [round(v, 4) for v in record['mean_norm_entropy_curve']]))}")
    print(f"[calibrate] snapshot written: {cfg.snapshot_path}")


if __name__ == "__main__":
    main()
