#!/usr/bin/env bash
#SBATCH --job-name=hopfield-exp01
#SBATCH --output=slurm-exp01-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
    micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PROMPTS="${PROMPTS:?Set PROMPTS=path/to/truthfulqa_paired.json}"
BANK_PATH="data/cache/banks/qwen25_3b__gate.pt"
HIDDEN_PATH="data/cache/hidden_states/qwen25_3b__mean.pt"
OUT_DIR="runs/exp01_truthfulqa_baseline"

python -m hopfield_llm.cli.run build-memory \
    --model qwen25_3b \
    --output "$BANK_PATH" \
    --bank gate

python -m hopfield_llm.cli.run extract-hidden \
    --memory "$BANK_PATH" \
    --prompts "$PROMPTS" \
    --output "$HIDDEN_PATH"

python -m hopfield_llm.cli.run score \
    --memory "$BANK_PATH" \
    --hidden "$HIDDEN_PATH" \
    --output "${OUT_DIR}/scores.pt" \
    --beta 15.0 \
    --energy-mode cosine \
    --active-threshold 0.1 \
    --score-metric js \
    --score-aggregation mean

python -m hopfield_llm.cli.run analyze \
    --scores "${OUT_DIR}/scores.pt" \
    --output "${OUT_DIR}/analysis.json"

python -m hopfield_llm.cli.run visualize \
    --analysis "${OUT_DIR}/analysis.json" \
    --output "${OUT_DIR}/plots"
