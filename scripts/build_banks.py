#!/usr/bin/env python
"""Stage 1 — build the memory bank artifact (banks.pt).

Usage:
    python scripts/build_banks.py --config configs/e1.yaml [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.banks import build_banks
from energy_llm.config import load_config
from energy_llm.models import load_model_and_tokenizer, resolve_model_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if banks.pt already exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    if cfg.banks_path.exists() and not args.force:
        print(f"[build_banks] cache hit: {cfg.banks_path} — skipping (use --force to rebuild)")
        return

    # Bank extraction reads weights to CPU float32 (subsec:stage1); the model
    # never needs to live on a GPU for this stage.
    model, _ = load_model_and_tokenizer(
        cfg.model.alias, dtype="float32",
        load_in_4bit=cfg.model.load_in_4bit, device="cpu",
    )
    artifact = build_banks(
        model, model_id=resolve_model_id(cfg.model.alias),
        bank_id=cfg.bank.bank_id, output_path=cfg.banks_path,
    )
    print(f"[build_banks] wrote {cfg.banks_path}  "
          f"(bank={artifact['bank_id']}, L={artifact['n_layers']}, "
          f"dims={artifact['dims'][0]})")


if __name__ == "__main__":
    main()
