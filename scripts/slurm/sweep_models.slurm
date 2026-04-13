#!/usr/bin/env bash
#SBATCH --job-name=hopfield-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/models_%j.out
#SBATCH --error=logs/models_%j.err
#
# Sequential model sweep on a single GPU.
# For concurrent model runs submit one jobs/submit_pipeline.sh per model.
#
set -euo pipefail

DATASET="${DATASET:-triviaqa}"
MODELS="${MODELS:-qwen25_3b phi3_mini llama32_3b}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
fi

for model in $MODELS; do
    BANKS="$SCRATCH/hopfield_cache/banks/${model}_banks.pt"
    OUT="$SCRATCH/hopfield_results/${model}/${DATASET}"
    mkdir -p "$OUT" logs

    echo "=== $model / $DATASET ==="
    hopfield-llm build-banks --model "$model" --output "$BANKS"
    hopfield-llm run-trajectory \
        --model "$model" --dataset "$DATASET" \
        --banks "$BANKS" --output "$OUT"
    echo "  → $OUT"
done
