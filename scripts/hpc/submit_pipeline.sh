#!/bin/bash
#
# Submit the 2-stage hopfield-llm pipeline to SLURM.
#
# Stage 1 (single job):  build_banks  — extract W_down[l] for all layers
# Stage 2 (array job):   run_trajectory — prefill + generation, N shards in parallel
#
# Usage:
#   ./jobs/submit_pipeline.sh                              # defaults
#   MODEL=qwen25_7b DATASET=nq NUM_SHARDS=16 \
#     ./jobs/submit_pipeline.sh                           # larger run
#
# Key env vars (all optional):
#   MODEL          model alias (default: qwen25_3b)
#   DATASET        triviaqa | nq | truthfulqa (default: triviaqa)
#   NUM_SHARDS     array width for Stage 2 (default: 8)
#   BETA           Hopfield inverse temperature (default: 15.0)
#   ENERGY_MODE    dot | cosine (default: dot)
#   MAX_NEW_TOKENS tokens to generate per sample (default: 50)
#   OUTPUT_DIR     trajectory output dir (default: $SCRATCH/hopfield_results/MODEL/DATASET)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/templates"

export MODEL="${MODEL:-qwen25_3b}"
export DATASET="${DATASET:-triviaqa}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export BETA="${BETA:-15.0}"
export ENERGY_MODE="${ENERGY_MODE:-dot}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_results/${MODEL}/${DATASET}}"
export BANKS_PATH="${BANKS_PATH:-$SCRATCH/hopfield_cache/banks/${MODEL}_banks.pt}"
export SCORE_METRIC="${SCORE_METRIC:-delta_energy}"
export SCORE_AGGREGATION="${SCORE_AGGREGATION:-mean}"

mkdir -p logs

echo "================================================"
echo " hopfield-llm pipeline submission"
echo " Model:       $MODEL"
echo " Dataset:     $DATASET"
echo " Shards:      $NUM_SHARDS"
echo " Beta:        $BETA  energy_mode: $ENERGY_MODE"
echo " Banks:       $BANKS_PATH"
echo " Output:      $OUTPUT_DIR"
echo "================================================"

# Stage 1: bank extraction (single GPU, skips if banks.pt exists)
JOB1=$(sbatch --parsable "$TEMPLATE_DIR/extract_banks.sh")
echo "[1/3] build_banks  → job $JOB1"

# Stage 2: trajectory array (depends on Stage 1, one task per shard)
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%4"
JOB2=$(sbatch --parsable \
    --dependency=afterok:"$JOB1" \
    --array="$ARRAY_SPEC" \
    "$TEMPLATE_DIR/forward_pass.sh")
echo "[2/3] run_trajectory array $ARRAY_SPEC → job $JOB2 (after $JOB1)"

# Stage 3: analysis + visualization (CPU-only, after all shards complete)
JOB3=$(sbatch --parsable \
    --dependency=afterok:"$JOB2" \
    "$TEMPLATE_DIR/analyze.sh")
echo "[3/3] analyze + visualize → job $JOB3 (after $JOB2)"

echo ""
echo "Pipeline submitted: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u \$USER"
echo "Output:   $OUTPUT_DIR"
echo "Cancel:   scancel $JOB1 $JOB2 $JOB3"
