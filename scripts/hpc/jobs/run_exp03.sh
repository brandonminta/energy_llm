#!/bin/bash
#SBATCH --job-name=hopfield-exp03
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp03_%j.out
#SBATCH --error=logs/exp03_%j.err
#
# exp03 — 500 TruthfulQA samples, Llama 3.2-3B, stages 1+2+3+4 in one job.
# Comparable to exp02 (Qwen2.5-3B) — same dataset, beta, seed, config.
# Stages: build-banks → run-trajectory (500 samples) → analyze → visualize
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp03.sh
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="$HOME/hopfield_results"

export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

echo "============================================"
echo " exp03 — TruthfulQA 500 samples, Llama 3.2-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Results: $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR" logs

hopfield-llm run-experiment \
    --config "$REPO_ROOT/configs/experiments/exp03_truthfulqa_500.yaml" \
    --override "base_dir=$RESULTS_DIR"

echo "============================================"
echo " Done: $(date)"
echo " Output: $RESULTS_DIR/exp03_truthfulqa_500/"
echo "============================================"
