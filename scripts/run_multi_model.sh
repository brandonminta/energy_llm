#!/usr/bin/env bash
# Run the full 5-stage pipeline for each model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

BANK="${BANK:-gate}"
PROMPTS="${PROMPTS:?Set PROMPTS=path/to/truthfulqa_paired.json}"

for model in qwen25_3b phi3_mini llama32_3b; do
    echo "[multi_model] model=${model}: running full pipeline..."
    OUT_DIR="runs/multi_model/${model}"
    BANK_PATH="data/cache/banks/${model}__${BANK}.pt"
    HIDDEN_PATH="data/cache/hidden_states/${model}__mean.pt"

    python -m hopfield_llm.cli.run build-memory \
        --model "$model" \
        --output "$BANK_PATH" \
        --bank "$BANK"

    python -m hopfield_llm.cli.run extract-hidden \
        --memory "$BANK_PATH" \
        --prompts "$PROMPTS" \
        --output "$HIDDEN_PATH"

    python -m hopfield_llm.cli.run score \
        --memory "$BANK_PATH" \
        --hidden "$HIDDEN_PATH" \
        --output "${OUT_DIR}/scores.pt" \
        "$@"

    python -m hopfield_llm.cli.run analyze \
        --scores "${OUT_DIR}/scores.pt" \
        --output "${OUT_DIR}/analysis.json"

    python -m hopfield_llm.cli.run visualize \
        --analysis "${OUT_DIR}/analysis.json" \
        --output "${OUT_DIR}/plots"
done
