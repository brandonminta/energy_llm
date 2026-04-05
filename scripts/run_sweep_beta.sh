#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

for beta in 5 10 15 20; do
  python -m energy_llm.cli.main run-exp1 \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --beta "$beta" \
    --output-dir "runs/beta_sweep/beta_${beta}" \
    "$@"
done
