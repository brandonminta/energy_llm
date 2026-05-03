#!/bin/bash
#SBATCH --job-name=hopfield-traj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/traj_%A_%a.out
#SBATCH --error=logs/traj_%A_%a.err
#SBATCH --array=0-7%4
#
# Stage 2 — run prefill + generation for each sample (array job, 8 shards)
#
# %A = job array ID, %a = task index (0-based shard_id)
# Concurrency limit: max 4 tasks running simultaneously.
#
# Override before sbatch:
#   MODEL=qwen25_7b DATASET=nq NUM_SHARDS=16 ./jobs/submit_pipeline.sh
#
set -euo pipefail

MODEL="${MODEL:-qwen25_3b}"
DATASET="${DATASET:-triviaqa}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BANKS_PATH="${BANKS_PATH:-$SCRATCH/hopfield_cache/banks/${MODEL}_banks.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_results/${MODEL}/${DATASET}}"
BETA="${BETA:-15.0}"
ENERGY_MODE="${ENERGY_MODE:-dot}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

mkdir -p "$OUTPUT_DIR" logs

echo "============================================"
echo " run_trajectory  shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
echo " model=${MODEL}  dataset=${DATASET}"
echo " banks=${BANKS_PATH}"
echo " output=${OUTPUT_DIR}"
echo "============================================"

hopfield-llm run-trajectory \
    --model        "$MODEL" \
    --dataset      "$DATASET" \
    --banks        "$BANKS_PATH" \
    --output       "$OUTPUT_DIR" \
    --beta         "$BETA" \
    --energy-mode  "$ENERGY_MODE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --shard-id     "$SLURM_ARRAY_TASK_ID" \
    --num-shards   "$NUM_SHARDS"

echo "[run_trajectory] shard ${SLURM_ARRAY_TASK_ID} done → $OUTPUT_DIR"
