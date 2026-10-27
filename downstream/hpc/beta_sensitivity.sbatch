#!/bin/bash
#SBATCH --job-name=energy-beta-sensitivity
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/beta_sensitivity_%j.out
#SBATCH --error=logs/beta_sensitivity_%j.err
#
# Beta sensitivity analysis — recomputes phi over a beta grid from the cached
# retrieval-score tensors; no forward passes, no GPU required. Resource
# directives preserved verbatim from the proven working CPU job template.
#
# Submit from repo root:
#   sbatch hpc/beta_sensitivity.sbatch configs/e1.yaml
#
set -euo pipefail

CONFIG="${1:?usage: sbatch hpc/beta_sensitivity.sbatch <config.yaml>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

cd "$REPO_ROOT"
mkdir -p logs

echo "[beta_sensitivity] config=$CONFIG node=$(hostname) date=$(date)"
python scripts/run_beta_sensitivity.py --config "$CONFIG"
echo "[beta_sensitivity] done: $(date)"
