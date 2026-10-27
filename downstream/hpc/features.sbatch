#!/bin/bash
#SBATCH --job-name=energy-features
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/features_%j.out
#SBATCH --error=logs/features_%j.err
#
# Stage 4 — feature aggregation + probe + statistics (CPU-only). Runs after
# all Stage 2 shards complete and Stage 3 labelling is done. Resource
# directives preserved verbatim from the previous working CPU job
# (scripts/hpc/templates/analyze.sh).
#
# Submit from repo root:
#   sbatch hpc/features.sbatch configs/e1.yaml
#
set -euo pipefail

CONFIG="${1:?usage: sbatch hpc/features.sbatch <config.yaml>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

cd "$REPO_ROOT"
mkdir -p logs

echo "[features] config=$CONFIG node=$(hostname) date=$(date)"
python scripts/build_features.py --config "$CONFIG"
python scripts/run_probe.py --config "$CONFIG"
echo "[features] done: $(date)"
