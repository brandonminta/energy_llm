#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

for model_cfg in \
  configs/models/qwen25_3b.yaml \
  configs/models/phi3_mini.yaml \
  configs/models/llama32_3b.yaml; do
  model_name="$(basename "$model_cfg" .yaml)"
  python -m energy_llm.cli.main run-exp1 \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --model-config "$model_cfg" \
    --output-dir "runs/multi_model/${model_name}" \
    "$@"
done
