#!/usr/bin/env bash
#SBATCH --job-name=energy-llm-datasets
#SBATCH --output=slurm-datasets-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
  source "$HOME/micromamba/etc/profile.d/micromamba.sh"
  micromamba activate "${MAMBA_ENV_NAME:-energy-llm}"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
python -m energy_llm.cli.main run-exp1 --config configs/experiments/exp01_truthfulqa_baseline.yaml "$@"
