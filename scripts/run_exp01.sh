#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Stage 1: Build memory bank
echo "[run_exp01] Stage 1: Building memory bank..."
python -m hopfield_llm.cli.run build-memory \
    --model qwen25_3b \
    --output data/cache/banks/qwen25_3b__gate.pt \
    --bank gate

# Stage 2: Extract hidden states
# (requires a prompts JSON file — generate from TruthfulQA or provide manually)
# python -m hopfield_llm.cli.run extract-hidden \
#     --memory data/cache/banks/qwen25_3b__gate.pt \
#     --prompts data/prompts/truthfulqa_paired.json \
#     --output data/cache/hidden_states/qwen25_3b__mean.pt

echo "[run_exp01] Bank extraction complete."
echo "[run_exp01] For the full pipeline, use: ./jobs/submit_pipeline.sh"
