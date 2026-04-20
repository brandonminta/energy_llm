#!/bin/bash
#SBATCH --job-name=hopfield-analyze
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#
# Stage 3 — Analyze trajectory outputs and generate plots (CPU-only).
#
# Runs after all trajectory shards complete.
#
set -euo pipefail

MODEL="${MODEL:-qwen25_3b}"
DATASET="${DATASET:-triviaqa}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_results/${MODEL}/${DATASET}}"
SCORE_METRIC="${SCORE_METRIC:-js}"
SCORE_AGGREGATION="${SCORE_AGGREGATION:-mean}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

export HOPFIELD_ENV=hpc

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
fi

mkdir -p logs

echo "============================================"
echo " analyze + visualize"
echo " model=${MODEL}  dataset=${DATASET}"
echo " trajectories=${OUTPUT_DIR}"
echo " score_metric=${SCORE_METRIC}"
echo "============================================"

hopfield-llm analyze \
    --trajectories "$OUTPUT_DIR" \
    --output       "$OUTPUT_DIR/analysis.json" \
    --score-metric "$SCORE_METRIC" \
    --score-aggregation "$SCORE_AGGREGATION"

hopfield-llm visualize \
    --analysis "$OUTPUT_DIR/analysis.json" \
    --output   "$OUTPUT_DIR/plots"

echo "[analyze] Done → $OUTPUT_DIR/analysis.json + $OUTPUT_DIR/plots/"
