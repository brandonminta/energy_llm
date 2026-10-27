#!/bin/bash
#SBATCH --job-name=energy-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# One array task = one shard (its own directory; nothing shared is written).
# Submit with hpc/submit.sh extract, which sets --array from the shard plan.
#   USR1 (SIGNAL_LEAD_S before walltime) -> Python finishes the current id,
#   closes files, exits 75 -> scontrol requeue this task.
#   TERM (preemption) -> same clean exit; SLURM requeues via --requeue.
set -uo pipefail
CONFIG="${1:?usage: extract_array.sbatch <config.yaml> <input.jsonl> <name>}"
INPUT="${2:?}"
NAME="${3:?}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
activate_env

TASK="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[extract] task=$TASK shard=$SLURM_ARRAY_TASK_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"
echo "[extract] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
run_requeueable "$TASK" python -m energy_llm extract --config "$CONFIG" --input "$INPUT" \
    --name "$NAME" --shard-index "$SLURM_ARRAY_TASK_ID"
