#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m energy_llm.cli.main run-exp1 --config configs/experiments/exp01_truthfulqa_baseline.yaml "$@"
