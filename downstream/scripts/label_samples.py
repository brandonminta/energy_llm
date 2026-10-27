#!/usr/bin/env python
"""Stage 3 — label trajectories with the OpenAI-compatible judge
(CPU-only; needs OPENAI_API_KEY). Resumable: already-labelled samples are
skipped.

Usage:
    python scripts/label_samples.py --config configs/e1.yaml [--limit 5]
    python scripts/label_samples.py --config configs/e1.yaml --make-human-csv
    python scripts/label_samples.py --config configs/e1.yaml --kappa results/e1/human_labels.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.config import load_config
from energy_llm.labeling import (
    judge_human_kappa,
    label_samples,
    make_human_subset_csv,
    make_judge_client,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--make-human-csv", action="store_true",
                        help="write the 60-sample human-labelling template CSV")
    parser.add_argument("--kappa", default=None, metavar="HUMAN_CSV",
                        help="compute Cohen's kappa against the filled human CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)

    if args.make_human_csv:
        path = make_human_subset_csv(
            cfg.labels_dir, cfg.out / "human_labels.csv",
            n=cfg.labeling.human_subset_size, seed=cfg.seed,
        )
        print(f"[label] human-labelling template written: {path} — fill the "
              f"human_label column with 0 (supported) or 1 (hallucination)")
        return

    if args.kappa:
        out = judge_human_kappa(cfg.labels_dir, args.kappa)
        print(f"[label] Cohen's kappa = {out['kappa']:.3f}  "
              f"(n={out['n_subset']}, raw agreement={out['raw_agreement']:.3f}, "
              f"passes kappa>=0.6: {out['passes_threshold']})")
        return

    client = make_judge_client(cfg.labeling.api_base)
    counts = label_samples(
        cfg.trajectories_dir, cfg.labels_dir, client,
        judge_model=cfg.labeling.judge_model,
        max_retries=cfg.labeling.max_retries, limit=args.limit,
    )
    print(f"[label] {counts}")


if __name__ == "__main__":
    main()
