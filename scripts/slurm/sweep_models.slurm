#!/usr/bin/env bash
#SBATCH --job-name=energy-llm-models
#SBATCH --output=slurm-models-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
  source "$HOME/micromamba/etc/profile.d/micromamba.sh"
  micromamba activate "${MAMBA_ENV_NAME:-energy-llm}"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
for model_cfg in configs/models/qwen25_3b.yaml configs/models/phi3_mini.yaml configs/models/llama32_3b.yaml; do
  model_name="$(basename "$model_cfg" .yaml)"
  python -m energy_llm.cli.main run-exp1 \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --model-config "$model_cfg" \
    --output-dir "runs/slurm_multi_model/${model_name}"
done
