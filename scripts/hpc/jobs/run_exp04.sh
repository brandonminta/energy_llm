#!/bin/bash
#SBATCH --job-name=hopfield-exp04
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp04_%j.out
#SBATCH --error=logs/exp04_%j.err
#
# exp04 — 500 TruthfulQA samples, Qwen2.5-3B.
# Stages: build-banks → run-trajectory → analyze → visualize
#
# This is the main experiment: the clean energy-trajectory baseline after the
# chat-template fix.  Run the smoke test first to confirm the cluster works.
#
# If the job was interrupted, resubmit with --skip-banks to skip bank
# extraction (banks.pt already on disk) and the trajectory stage will
# automatically resume from the last completed sample.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp04.sh
#
# Resume after interruption:
#   sbatch scripts/hpc/jobs/run_exp04.sh --skip-banks   (pass to run-experiment)
#   (or add --skip-banks to the hopfield-llm call below)
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/hopfield_results}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

echo "============================================"
echo " exp04 — TruthfulQA 500 samples, Qwen2.5-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " HF_HOME: $HF_HOME"
echo " Results: $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR" logs

hopfield-llm run-experiment \
    --config "$REPO_ROOT/configs/experiments/exp04_qwen_chat_template_fix.yaml" \
    --override "base_dir=$RESULTS_DIR"

echo "============================================"
echo " exp04 Done: $(date)"
echo " Output: $RESULTS_DIR/exp04_qwen_chat_template_fix/"
echo "============================================"
