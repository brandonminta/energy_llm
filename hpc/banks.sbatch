#!/bin/bash
#SBATCH --job-name=energy-banks
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/banks_%j.out
#SBATCH --error=logs/banks_%j.err
#
# Stage 1 — memory bank extraction (banks.pt). Resource directives preserved
# verbatim from the previous working setup (scripts/hpc/templates/
# extract_banks.sh). The stage itself runs on CPU (weights are read to CPU
# float32); the GRES is kept only because it is what gets scheduled.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch hpc/banks.sbatch configs/e1.yaml
#
set -euo pipefail

CONFIG="${1:?usage: sbatch hpc/banks.sbatch <config.yaml>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

cd "$REPO_ROOT"
mkdir -p logs

echo "[banks] config=$CONFIG node=$(hostname) date=$(date)"
python scripts/build_banks.py --config "$CONFIG"
echo "[banks] done: $(date)"
