#!/usr/bin/env bash
#SBATCH --job-name=hopfield-datasets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/datasets_%j.out
#SBATCH --error=logs/datasets_%j.err
#
# Run trajectory across all three datasets for one model.
#
set -euo pipefail

MODEL="${MODEL:-qwen25_3b}"
BANKS="$SCRATCH/hopfield_cache/banks/${MODEL}_banks.pt"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
fi

mkdir -p logs
hopfield-llm build-banks --model "$MODEL" --output "$BANKS"

for dataset in triviaqa nq truthfulqa; do
    OUT="$SCRATCH/hopfield_results/${MODEL}/${dataset}"
    mkdir -p "$OUT"
    echo "=== $MODEL / $dataset ==="
    hopfield-llm run-trajectory \
        --model "$MODEL" --dataset "$dataset" \
        --banks "$BANKS" --output "$OUT"
    echo "  → $OUT"
done
