#!/usr/bin/env bash
# Beta is now a flag on run-trajectory; each run produces independent artifacts.
# Run separate trajectory collections for each beta value.
#
# Usage: MODEL=qwen25_3b DATASET=triviaqa ./scripts/local/run_sweep_beta.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"


MODEL="${MODEL:-qwen25_3b}"
DATASET="${DATASET:-triviaqa}"
BANKS="data/cache/banks/${MODEL}_banks.pt"

mkdir -p data/cache/banks
hopfield-llm build-banks --model "$MODEL" --output "$BANKS"

for beta in 5 10 15 20; do
    OUT="outputs/beta_sweep/beta_${beta}/${MODEL}_${DATASET}"
    echo "[beta=$beta] → $OUT"
    hopfield-llm run-trajectory \
        --model "$MODEL" --dataset "$DATASET" \
        --banks "$BANKS" --output "$OUT" \
        --beta "$beta"
done
