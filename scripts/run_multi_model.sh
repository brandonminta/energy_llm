#!/usr/bin/env bash
# Run trajectory for multiple models sequentially on one machine.
# For HPC, submit one jobs/submit_pipeline.sh per model instead.
#
# Usage: DATASET=triviaqa ./scripts/run_multi_model.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DATASET="${DATASET:-triviaqa}"
MODELS="${MODELS:-qwen25_3b phi3_mini llama32_3b}"

for model in $MODELS; do
    echo "=== $model / $DATASET ==="
    BANKS="data/cache/banks/${model}_banks.pt"
    OUT="runs/multi_model/${model}_${DATASET}"
    mkdir -p data/cache/banks "$OUT"

    hopfield-llm build-banks --model "$model" --output "$BANKS"
    hopfield-llm run-trajectory \
        --model "$model" --dataset "$DATASET" \
        --banks "$BANKS" --output "$OUT"
    echo "  → $OUT"
done
