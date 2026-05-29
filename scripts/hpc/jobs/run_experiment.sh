#!/bin/bash
#SBATCH --job-name=hopfield-exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Parametrized single-job wrapper around `hopfield-llm run-experiment`.
# Replaces the per-experiment run_expNN.sh / run_expNN_resume.sh scripts: the
# config path is the first argument and everything after it is forwarded to
# run-experiment.
#
# run-experiment auto-resumes the trajectory stage (already-processed samples
# are skipped), so a resubmit after an OOM/time-out just needs --skip-banks.
#
# Submit from repo root (override the SLURM job name to taste):
#   mkdir -p logs
#   sbatch --job-name=final_qwen_tqa \
#          scripts/hpc/jobs/run_experiment.sh configs/experiments/final_qwen_truthfulqa.yaml
#
# Resume after interruption (banks already on disk):
#   sbatch scripts/hpc/jobs/run_experiment.sh configs/experiments/final_qwen_truthfulqa.yaml --skip-banks
#
set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <config.yaml> [extra run-experiment args...]}"
shift || true

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/hopfield_results}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

# Resolve the config relative to the repo root when not an absolute path.
case "$CONFIG" in
  /*) CONFIG_PATH="$CONFIG" ;;
  *)  CONFIG_PATH="$REPO_ROOT/$CONFIG" ;;
esac

echo "============================================"
echo " run-experiment"
echo " Config:  $CONFIG_PATH"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " HF_HOME: $HF_HOME"
echo " Results: $RESULTS_DIR"
echo " Extra:   $*"
echo "============================================"

mkdir -p "$RESULTS_DIR" logs

hopfield-llm run-experiment \
    --config "$CONFIG_PATH" \
    --override "base_dir=$RESULTS_DIR" \
    "$@"

echo "============================================"
echo " Done: $(date)  (config: $(basename "$CONFIG_PATH"))"
echo "============================================"
