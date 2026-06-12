#!/bin/bash
#
# Submit the GPU stages of one experiment with SLURM dependencies:
#   banks -> calibrate -> trajectory array (8 shards)
#
# Stage 3 (labelling) is NOT submitted: it needs OPENAI_API_KEY and is run
# interactively (notebooks/01_labeling_openai.ipynb or
# scripts/label_samples.py) after the trajectories land. Stage 4 is then
#   sbatch hpc/features.sbatch <config>
#
# Usage (from repo root):
#   mkdir -p logs
#   ./hpc/submit_pipeline.sh configs/e1.yaml
#   NUM_SHARDS=16 ./hpc/submit_pipeline.sh configs/e2.yaml
#
set -euo pipefail

CONFIG="${1:?usage: submit_pipeline.sh <config.yaml>}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs

echo "================================================"
echo " energy_llm pipeline submission"
echo " Config: $CONFIG    Shards: $NUM_SHARDS"
echo "================================================"

JOB1=$(sbatch --parsable "$SCRIPT_DIR/banks.sbatch" "$CONFIG")
echo "[1/3] banks        -> job $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" \
       "$SCRIPT_DIR/calibrate.sbatch" "$CONFIG")
echo "[2/3] calibrate    -> job $JOB2 (after $JOB1)"

ARRAY_SPEC="0-$((NUM_SHARDS - 1))%4"
JOB3=$(sbatch --parsable --dependency=afterok:"$JOB2" \
       --array="$ARRAY_SPEC" \
       "$SCRIPT_DIR/trajectories.sbatch" "$CONFIG")
echo "[3/3] trajectories -> job $JOB3 array $ARRAY_SPEC (after $JOB2)"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Then: label (notebook 01 or scripts/label_samples.py), and"
echo "      sbatch hpc/features.sbatch $CONFIG"
echo "Cancel:   scancel $JOB1 $JOB2 $JOB3"
