#!/bin/bash
#SBATCH --job-name=energy-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#
# Smoke test: extract + merge the first N questions in one process (no array,
# no requeue), then print dtype, TF==greedy %, ||x_hat|| deviation, peak GPU
# memory, s/question, mean H_bar per layer at beta*, empty/truncated counts.
# Needs a calibrated beta*. Submit with hpc/submit.sh smoke.
set -euo pipefail
CONFIG="${1:?usage: smoke.sbatch <config.yaml> <input.jsonl> [N]}"
INPUT="${2:?}"
N="${3:-20}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
activate_env

echo "[smoke] job=$SLURM_JOB_ID node=$(hostname) N=$N config=$CONFIG"
echo "[smoke] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
python -m energy_llm smoke --config "$CONFIG" --input "$INPUT" --n "$N"
