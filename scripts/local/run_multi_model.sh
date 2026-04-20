#!/usr/bin/env bash
# Run trajectory for multiple models sequentially on one machine.
# For HPC, submit one scripts/hpc/submit_pipeline.sh per model instead.
#
# Usage: DATASET=triviaqa ./scripts/local/run_multi_model.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"


DATASET="${DATASET:-triviaqa}"
MODELS="${MODELS:-qwen25_3b phi3_mini llama32_3b}"

for model in $MODELS; do
    echo "=== $model / $DATASET ==="
    BANKS="data/cache/banks/${model}_banks.pt"
    OUT="outputs/multi_model/${model}_${DATASET}"
    mkdir -p data/cache/banks "$OUT"

    hopfield-llm build-banks --model "$model" --output "$BANKS"
    hopfield-llm run-trajectory \
        --model "$model" --dataset "$DATASET" \
        --banks "$BANKS" --output "$OUT"
    echo "  → $OUT"
done
