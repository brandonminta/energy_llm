#!/usr/bin/env python
"""Stage 4 (aggregation) — build the feature table from trajectory + label
artifacts (CPU-only, order-independent over shards).

Usage:
    python scripts/build_features.py --config configs/e1.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energy_llm.config import load_config
from energy_llm.features import build_feature_table
from energy_llm.labeling import load_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    labels = load_labels(cfg.labels_dir) if cfg.labels_dir.exists() else None
    df = build_feature_table(cfg.trajectories_dir, labels=labels)
    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.features_dir / "features.csv"
    tmp = out.with_name(out.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    n_labelled = int(df["is_hallucination"].notna().sum()) if "is_hallucination" in df else 0
    print(f"[features] wrote {out}: {len(df)} samples, {n_labelled} labelled, "
          f"{len(df.columns)} columns")


if __name__ == "__main__":
    main()
