#!/bin/bash
#SBATCH --job-name=energy-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#
# Final merge, chained with --dependency=afterok on the extraction array.
set -euo pipefail
CONFIG="${1:?usage: merge.sbatch <config.yaml> <input.jsonl> <name>}"
INPUT="${2:?}"
NAME="${3:?}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source hpc/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
activate_env

echo "[merge] job=$SLURM_JOB_ID node=$(hostname) name=$NAME"
python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME"
python -m energy_llm merge --config "$CONFIG" --input "$INPUT" --name "$NAME"
