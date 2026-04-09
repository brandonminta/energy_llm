#!/bin/bash
#
# Master pipeline submission script with SLURM dependency chaining.
#
# Usage:
#   ./jobs/submit_pipeline.sh                           # defaults
#   MODEL=phi3mini BETA=20.0 ./jobs/submit_pipeline.sh  # custom
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/templates"

# Propagate configuration via environment
export MODEL="${MODEL:-qwen25_3b}"
export BANK="${BANK:-gate}"
export BETA="${BETA:-15.0}"
export METRIC="${METRIC:-js}"
export POOLING="${POOLING:-mean}"

mkdir -p logs

echo "========================================="
echo " Hopfield-LLM Pipeline Submission"
echo " Model:   $MODEL"
echo " Bank:    $BANK"
echo " Beta:    $BETA"
echo " Metric:  $METRIC"
echo " Pooling: $POOLING"
echo "========================================="

# Stage 1: Extract banks
JOB1=$(sbatch --parsable "$TEMPLATE_DIR/extract_banks.sh")
echo "[1/3] Bank extraction: Job $JOB1"

# Stage 2: Forward pass (depends on stage 1)
JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" "$TEMPLATE_DIR/forward_pass.sh")
echo "[2/3] Hidden states:   Job $JOB2 (after $JOB1)"

# Stages 3-5: Score + Analyze + Visualize (depends on stage 2)
JOB3=$(sbatch --parsable --dependency=afterok:"$JOB2" "$TEMPLATE_DIR/compute_metrics.sh")
echo "[3/3] Metrics:         Job $JOB3 (after $JOB2)"

echo ""
echo "Pipeline submitted: $JOB1 -> $JOB2 -> $JOB3"
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $JOB1 $JOB2 $JOB3"
