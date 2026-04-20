#!/usr/bin/env bash
# Run a single-machine experiment (no SLURM).
# Processes the full dataset sequentially on one GPU.
#
# Usage:
#   ./scripts/local/run_exp01.sh
#   MODEL=qwen25_7b DATASET=nq MAX_SAMPLES=500 ./scripts/local/run_exp01.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-qwen25_3b}"
DATASET="${DATASET:-triviaqa}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BETA="${BETA:-15.0}"
ENERGY_MODE="${ENERGY_MODE:-dot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
OUT="outputs/exp01/${MODEL}_${DATASET}"
BANKS="data/cache/banks/${MODEL}_banks.pt"

mkdir -p "$OUT" data/cache/banks

echo "=== run_exp01: $MODEL / $DATASET ==="

echo "[1/2] build-banks..."
hopfield-llm build-banks --model "$MODEL" --output "$BANKS"

echo "[2/2] run-trajectory..."
hopfield-llm run-trajectory \
    --model          "$MODEL" \
    --dataset        "$DATASET" \
    --banks          "$BANKS" \
    --output         "$OUT" \
    --beta           "$BETA" \
    --energy-mode    "$ENERGY_MODE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}

echo "Done → $OUT"
