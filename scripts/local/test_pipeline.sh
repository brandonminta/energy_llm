#!/usr/bin/env bash
# Local end-to-end smoke test.
#
# Exercises the headline pipeline path (gated_key bank, normalised banks,
# normalize_query=false, answer-span pooling) on a small slice of TruthfulQA
# with a ≤1.5B model, then runs the analysis + visualisation stages and the
# token-uncertainty baselines.  Stages 5–6 (label + evaluate) are noted at the
# end: labelling needs a judge model/API key and the probe needs ≥50 labelled
# samples, so they are run separately on a full trajectory directory.
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
# ---------------------------------------------------------------------

OUT="outputs/local_test/${MODEL}_${DATASET}"
BANKS="$OUT/banks.pt"
TRAJ="$OUT/trajectories"
ANALYSIS="$OUT/analysis.json"
PLOTS="$OUT/plots"

mkdir -p "$OUT"

echo "================================================"
echo " hopfield-llm  local smoke test"
echo " model:   $MODEL"
echo " dataset: $DATASET  ($N samples)"
echo " output:  $OUT"
echo "================================================"

# ---- Stage 1: bank extraction (gated_key, normalised) ---------------
echo "[1/5] build-banks (gated_key, --normalize)..."
hopfield-llm build-banks \
    --model  "$MODEL" \
    --output "$BANKS" \
    --bank   gated_key \
    --normalize

# ---- Stage 2: trajectory collection (normalize_query=false) ---------
echo "[2/5] run-trajectory ($N samples)..."
hopfield-llm run-trajectory \
    --model           "$MODEL" \
    --dataset         "$DATASET" \
    --banks           "$BANKS" \
    --output          "$TRAJ" \
    --max-samples     "$N" \
    --max-new-tokens  "$MAX_TOKENS" \
    --no-normalize-query

# ---- Stage 3: analysis (answer-span pooling) ------------------------
echo "[3/5] analyze (--pool-over-answer)..."
hopfield-llm analyze \
    --trajectories "$TRAJ" \
    --output       "$ANALYSIS" \
    --pool-over-answer \
    --tokenizer-id "$MODEL"

# ---- Stage 4: visualisation -----------------------------------------
echo "[4/5] visualize..."
hopfield-llm visualize --analysis "$ANALYSIS" --output "$PLOTS"

# ---- Baselines (token-uncertainty + P(True)) ------------------------
echo "[5/5] baselines (token uncertainty + P(True))..."
python -m hopfield_llm.evaluation.baselines --trajectories "$TRAJ" --model "$MODEL"

# ---- Artifact check -------------------------------------------------
NPZ_COUNT=$(find "$TRAJ" -name "*.npz" ! -name "*_hpre.npz" ! -name "*_scores.npz" | wc -l)
JSON_COUNT=$(find "$TRAJ" -maxdepth 1 -name "*.json" ! -name "*_baselines.json" | wc -l)
BL_COUNT=$(find "$TRAJ" -name "*_baselines.json" | wc -l)

echo "================================================"
echo " artifacts"
echo "   metric npz:      $NPZ_COUNT"
echo "   metadata json:   $JSON_COUNT"
echo "   baseline json:   $BL_COUNT"
echo "   analysis:        $ANALYSIS"
echo "   plots:           $PLOTS"
echo "================================================"
echo ""
echo "Stages 5–6 (label + evaluate) run on a full trajectory directory:"
echo "  export OPENAI_API_KEY=sk-...   # or --api-base '' for a local HF judge"
echo "  hopfield-llm label    --trajectories $TRAJ --output $OUT/labels"
echo "  hopfield-llm evaluate --trajectories $TRAJ --labels $OUT/labels --output $OUT/report.html"
