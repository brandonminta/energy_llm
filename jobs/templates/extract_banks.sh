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
# Stage 1 — extract W_down[l] for all layers and save banks.pt
#
# Override via environment variables before sbatch:
#   MODEL=llama31_8b ./jobs/submit_pipeline.sh
#
set -euo pipefail

MODEL="${MODEL:-qwen25_3b}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_cache/banks}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

mkdir -p "$OUTPUT_DIR" logs

OUTPUT="$OUTPUT_DIR/${MODEL}_banks.pt"

if [ -f "$OUTPUT" ]; then
    echo "[build_banks] Cache hit: $OUTPUT — skipping."
    exit 0
fi

echo "[build_banks] model=$MODEL  output=$OUTPUT"

hopfield-llm build-banks \
    --model "$MODEL" \
    --output "$OUTPUT"

echo "[build_banks] Done: $OUTPUT"
