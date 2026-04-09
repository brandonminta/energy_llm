#!/bin/bash
#SBATCH --job-name=hopfield-hidden
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hidden_%j.out
#SBATCH --error=logs/hidden_%j.err
#
# Stage 2: Hidden-state extraction (forward pass)
#
# Resources justification:
#   - gpu partition (48h limit) — forward pass is the most expensive stage
#   - a100_2g.10gb (10 GB) — all models fit in 4-bit with room for activations
#   - 8 CPUs, 32 GB RAM — CPU for tokenisation + RAM buffer for artifact I/O
#   - 24h — ~818 samples × 2 passes × ~1s each ≈ 30 min for Qwen-3B,
#     but larger models or all_incorrect strategy can be much longer
#
set -euo pipefail

# ---- Configuration ----
MODEL="${MODEL:-qwen25_3b}"
BANK_PATH="${BANK_PATH:-$SCRATCH/hopfield_cache/banks/${MODEL}__gate.pt}"
PROMPTS="${PROMPTS:-data/prompts/truthfulqa_paired.json}"
POOLING="${POOLING:-mean}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_cache/hidden_states}"

# ---- Setup ----
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

mkdir -p "$OUTPUT_DIR" logs

OUTPUT="$OUTPUT_DIR/${MODEL}__${POOLING}.pt"

# Skip if already cached
if [ -f "$OUTPUT" ]; then
    echo "[forward_pass] Cache hit: $OUTPUT — skipping."
    exit 0
fi

echo "[forward_pass] Model=$MODEL Pooling=$POOLING"
echo "[forward_pass] Bank=$BANK_PATH"
echo "[forward_pass] Prompts=$PROMPTS"

python -m hopfield_llm.cli.run extract-hidden \
    --memory "$BANK_PATH" \
    --prompts "$PROMPTS" \
    --output "$OUTPUT" \
    --pooling "$POOLING"

echo "[forward_pass] Done: $OUTPUT"
