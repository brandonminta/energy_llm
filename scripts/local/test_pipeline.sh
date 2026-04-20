#!/usr/bin/env bash
# Local end-to-end test.
# Runs build-banks then run-trajectory on a small slice of TruthfulQA.
#
# Usage:
#   ./scripts/local/test_pipeline.sh                     # 1.5B model, 20 samples
#   MODEL=qwen25_3b N=50 ./scripts/local/test_pipeline.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ----- configuration -------------------------------------------------
MODEL="${MODEL:-qwen25_1_5b}"   # safe default for 4 GB VRAM
DATASET="${DATASET:-truthfulqa}"
N="${N:-20}"                    # number of samples
MAX_TOKENS="${MAX_TOKENS:-50}"  # tokens to generate per sample
DIAG="${DIAG:-5}"               # samples for which to also save raw h_pre
# ---------------------------------------------------------------------

OUT="outputs/local_test/${MODEL}_${DATASET}"
BANKS="$OUT/banks.pt"
TRAJ="$OUT/trajectories"

mkdir -p "$OUT"

echo "================================================"
echo " hopfield-llm  local test"
echo " model:   $MODEL"
echo " dataset: $DATASET  ($N samples)"
echo " output:  $OUT"
echo "================================================"
echo ""

# ---- Stage 1: bank extraction ---------------------------------------
echo "[1/2] build-banks..."
hopfield-llm build-banks \
    --model  "$MODEL" \
    --output "$BANKS"
echo "      saved: $BANKS"
echo ""

# ---- Stage 2: trajectory collection --------------------------------
echo "[2/2] run-trajectory ($N samples, $MAX_TOKENS tokens each)..."
hopfield-llm run-trajectory \
    --model              "$MODEL" \
    --dataset            "$DATASET" \
    --banks              "$BANKS" \
    --output             "$TRAJ" \
    --max-samples        "$N" \
    --max-new-tokens     "$MAX_TOKENS" \
    --diagnostic-subset  "$DIAG"
echo "      saved: $TRAJ"
echo ""

# ---- Quick file check -----------------------------------------------
NPZ_COUNT=$(find "$TRAJ" -name "*.npz" ! -name "*_hpre.npz" | wc -l)
JSON_COUNT=$(find "$TRAJ" -name "*.json" | wc -l)
HPRE_COUNT=$(find "$TRAJ" -name "*_hpre.npz" | wc -l)

echo "================================================"
echo " artifacts"
echo "   metric npz:     $NPZ_COUNT"
echo "   metadata json:  $JSON_COUNT"
echo "   hpre npz:       $HPRE_COUNT  (diagnostic)"
echo "================================================"
echo ""
echo "Next step — inspect results:"
echo "  python scripts/local/inspect_results.py $TRAJ"
