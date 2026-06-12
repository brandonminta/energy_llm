#!/bin/bash
#SBATCH --job-name=energy-calibrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/calibrate_%j.out
#SBATCH --error=logs/calibrate_%j.err
#
# beta* calibration — N_c=30 prefill passes, writes beta_star into the
# config snapshot (results/<exp>/config_snapshot.yaml) that every later
# stage reads. Resource directives preserved verbatim from the previous
# working full-GPU job (scripts/hpc/jobs/run_experiment.sh); the job
# finishes in minutes.
#
# Submit from repo root:
#   sbatch hpc/calibrate.sbatch configs/e1.yaml
#
set -euo pipefail

CONFIG="${1:?usage: sbatch hpc/calibrate.sbatch <config.yaml>}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

cd "$REPO_ROOT"
mkdir -p logs

echo "[calibrate] config=$CONFIG node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
python scripts/calibrate_beta.py --config "$CONFIG"
echo "[calibrate] done: $(date)"
