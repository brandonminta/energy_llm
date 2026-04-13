#!/usr/bin/env python3
"""Merge multiple hidden state artifacts into a single file.

Usage:
    python scripts/merge_hidden_states.py \
        data/cache/hidden_states/shard_*.pt \
        --output data/cache/hidden_states/truthfulqa_merged.pt
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge hidden state shards")
    parser.add_argument("inputs", nargs="+", help="Input .pt files (glob supported)")
    parser.add_argument("--output", required=True, help="Output .pt file")
    args = parser.parse_args()

    # Expand globs if needed
    input_files = []
    for pattern in args.inputs:
        matches = sorted(Path(".").glob(pattern))
        input_files.extend(matches)

    if not input_files:
        print(f"No input files matched: {args.inputs}")
        return 1

    print(f"Loading {len(input_files)} artifacts...")
    samples = []
    model_summary = None
    memory_summary = None
    pooling = None
    separator = None
    layers = None
    batch_size = None

    for i, fpath in enumerate(input_files):
        print(f"  [{i+1}/{len(input_files)}] {fpath.name}")
        artifact = torch.load(fpath, weights_only=False)

        # Validate metadata consistency
        if model_summary is None:
            model_summary = artifact["model"]
            memory_summary = artifact.get("memory_summary")
            pooling = artifact.get("pooling")
            separator = artifact.get("separator")
            layers = artifact.get("layers")
            batch_size = artifact.get("batch_size")
        else:
            assert artifact["model"] == model_summary, f"Model mismatch in {fpath}"
            assert artifact.get("pooling") == pooling, f"Pooling mismatch in {fpath}"

        samples.extend(artifact["samples"])

    print(f"Total samples merged: {len(samples)}")

    # Create merged artifact
    merged = {
        "artifact_type": "hidden_states",
        "model": model_summary,
        "memory_summary": memory_summary,
        "pooling": pooling,
        "separator": separator,
        "layers": layers,
        "batch_size": batch_size,
        "samples": samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"\n✓ Merged artifact saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    exit(main())
