#!/bin/bash
#SBATCH --job-name=hopfield-banks
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/banks_%j.out
#SBATCH --error=logs/banks_%j.err
#
# Stage 1: MLP memory-bank extraction
#
# Resources justification:
#   - gpu-dev partition (96h limit, 1 GPU) — only loads model weights, no forward pass
#   - a100_1g.5gb (5 GB) — sufficient for all models in 4-bit (largest is ~4 GB)
#   - 4 CPUs, 16 GB RAM — weight dequantisation is CPU-bound for 4-bit models
#   - 2h — conservative; typical run is <30 min per model
#
set -euo pipefail

# ---- Configuration (override via environment variables) ----
MODEL="${MODEL:-qwen25_3b}"
BANK="${BANK:-gate}"
NORMALIZE="${NORMALIZE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_cache/banks}"

# ---- Setup ----
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

mkdir -p "$OUTPUT_DIR" logs

OUTPUT="$OUTPUT_DIR/${MODEL}__${BANK}.pt"

# Skip if already cached
if [ -f "$OUTPUT" ]; then
    echo "[extract_banks] Cache hit: $OUTPUT — skipping."
    exit 0
fi

NORM_FLAG=""
[ -n "$NORMALIZE" ] && NORM_FLAG="--normalize"

echo "[extract_banks] Model=$MODEL Bank=$BANK Output=$OUTPUT"

python -m hopfield_llm.cli.run build-memory \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --bank "$BANK" \
    $NORM_FLAG

echo "[extract_banks] Done: $OUTPUT"
