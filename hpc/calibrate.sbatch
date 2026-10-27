#!/bin/bash
#SBATCH --job-name=energy-calibrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.err
#
# beta* calibration on the warm-up JSONL. Submit with hpc/submit.sh calibrate
# (the #SBATCH lines are fallbacks; hpc/env.sh values override them).
set -uo pipefail
CONFIG="${1:?usage: calibrate.sbatch <config.yaml> <warmup.jsonl>}"
WARMUP="${2:?usage: calibrate.sbatch <config.yaml> <warmup.jsonl>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
activate_env

echo "[calibrate] job=$SLURM_JOB_ID node=$(hostname) config=$CONFIG warmup=$WARMUP"
echo "[calibrate] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
run_requeueable "$SLURM_JOB_ID" python -m energy_llm calibrate --config "$CONFIG" --input "$WARMUP"
