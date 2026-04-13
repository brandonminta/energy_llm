#!/usr/bin/env python3
"""Split a prompt JSON into N shards for parallel extraction.

Usage:
    python scripts/shard_prompts.py \
        data/interim/prompts/truthfulqa_paired.json \
        --output-dir data/interim/prompts/shards \
        --num-shards 10
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Shard prompts for parallel extraction")
    parser.add_argument("input", help="Input prompt JSON file")
    parser.add_argument("--output-dir", default="data/interim/prompts/shards",
                        help="Output directory for shards")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of shards to create")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all prompts
    with open(input_path) as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts from {input_path}")

    # Divide into shards
    shard_size = (len(prompts) + args.num_shards - 1) // args.num_shards
    shard_files = []

    for i in range(args.num_shards):
        start = i * shard_size
        end = min(start + shard_size, len(prompts))
        shard = prompts[start:end]

        if not shard:
            continue  # Skip empty shards

        shard_file = output_dir / f"shard_{i:03d}.json"
        with open(shard_file, "w") as f:
            json.dump(shard, f, indent=2)
        shard_files.append(shard_file)
        print(f"  Shard {i}: {len(shard):4d} prompts → {shard_file}")

    print(f"\n✓ Created {len(shard_files)} shards in {output_dir}")
    print(f"  Use with: sbatch --array=0-{len(shard_files)-1} scripts/slurm/extract_hidden_array.slurm")


if __name__ == "__main__":
    main()
