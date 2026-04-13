#!/usr/bin/env bash
#SBATCH --job-name=hopfield-exp01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp01_%j.out
#SBATCH --error=logs/exp01_%j.err
#
# Single-job experiment (no array). Processes the full dataset on one GPU.
# For parallel runs use jobs/submit_pipeline.sh instead.
#
set -euo pipefail

MODEL="${MODEL:-qwen25_3b}"
DATASET="${DATASET:-triviaqa}"
BANKS_PATH="${BANKS_PATH:-$SCRATCH/hopfield_cache/banks/${MODEL}_banks.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_results/${MODEL}/${DATASET}}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
fi

mkdir -p "$OUTPUT_DIR" logs

echo "[exp01] build-banks model=${MODEL}"
hopfield-llm build-banks --model "$MODEL" --output "$BANKS_PATH"

echo "[exp01] run-trajectory dataset=${DATASET}"
hopfield-llm run-trajectory \
    --model    "$MODEL" \
    --dataset  "$DATASET" \
    --banks    "$BANKS_PATH" \
    --output   "$OUTPUT_DIR"

echo "[exp01] Done → $OUTPUT_DIR"
