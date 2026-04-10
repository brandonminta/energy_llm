#!/usr/bin/env bash
# Sweep over Hopfield beta values: build bank + hidden states once, then sweep score stage.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

MODEL="${MODEL:-qwen25_3b}"
BANK="${BANK:-gate}"
PROMPTS="${PROMPTS:?Set PROMPTS=path/to/truthfulqa_paired.json}"

BANK_PATH="data/cache/banks/${MODEL}__${BANK}.pt"
HIDDEN_PATH="data/cache/hidden_states/${MODEL}__mean.pt"

# Stage 1: Build memory bank (once)
echo "[sweep_beta] Stage 1: Building memory bank..."
python -m hopfield_llm.cli.run build-memory \
    --model "$MODEL" \
    --output "$BANK_PATH" \
    --bank "$BANK"

# Stage 2: Extract hidden states (once — beta has no effect here)
echo "[sweep_beta] Stage 2: Extracting hidden states..."
python -m hopfield_llm.cli.run extract-hidden \
    --memory "$BANK_PATH" \
    --prompts "$PROMPTS" \
    --output "$HIDDEN_PATH"

# Stages 3–5: Score → analyze → visualize for each beta
for beta in 5 10 15 20; do
    echo "[sweep_beta] beta=${beta}: score → analyze → visualize..."
    OUT_DIR="runs/beta_sweep/beta_${beta}"
    python -m hopfield_llm.cli.run score \
        --memory "$BANK_PATH" \
        --hidden "$HIDDEN_PATH" \
        --output "${OUT_DIR}/scores.pt" \
        --beta "$beta" \
        "$@"
    python -m hopfield_llm.cli.run analyze \
        --scores "${OUT_DIR}/scores.pt" \
        --output "${OUT_DIR}/analysis.json"
    python -m hopfield_llm.cli.run visualize \
        --analysis "${OUT_DIR}/analysis.json" \
        --output "${OUT_DIR}/plots"
done
