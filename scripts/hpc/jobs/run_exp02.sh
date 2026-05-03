#!/bin/bash
#SBATCH --job-name=hopfield-exp02
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp02_%j.out
#SBATCH --error=logs/exp02_%j.err
#
# exp02 — 500 TruthfulQA samples, Qwen2.5-3B, stages 1+2+3 in one job.
# Stages: build-banks → run-trajectory (500 samples) → analyze → visualize
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp02.sh
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="$HOME/hopfield_results"

export HOPFIELD_ENV=hpc

# Activate environment
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

echo "============================================"
echo " exp02 — TruthfulQA 500 samples, Qwen2.5-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Results: $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR" logs

hopfield-llm run-experiment \
    --config "$REPO_ROOT/configs/experiments/exp02_truthfulqa_500.yaml" \
    --override "base_dir=$RESULTS_DIR"

echo "============================================"
echo " Done: $(date)"
echo " Output: $RESULTS_DIR/exp02_truthfulqa_500/"
echo "============================================"
