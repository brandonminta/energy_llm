#!/bin/bash
#SBATCH --job-name=hopfield-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#
# Smoke test — 20 samples, Qwen2.5-3B, all 4 stages.
# Same hardware as the real experiments; only max_samples differs.
# Purpose: confirm the pipeline runs end-to-end on the cluster
# (env, imports, GPU access, HF weights, disk writes) before committing
# to a 5-hour job.
#
# Expected runtime: ~10-15 min (includes model load + 20 forward passes)
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_quick_test.sh
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/hopfield_results}"

# HuggingFace caches model weights here — shared across all experiments
# so weights are downloaded only once.  Point to scratch if available.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

echo "============================================"
echo " Smoke test — 20 samples, Qwen2.5-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " HF_HOME: $HF_HOME"
echo " Results: $RESULTS_DIR"
echo "============================================"

mkdir -p "$RESULTS_DIR" logs

hopfield-llm run-experiment \
    --config "$REPO_ROOT/configs/experiments/exp04_qwen_chat_template_fix.yaml" \
    --override \
        "base_dir=$RESULTS_DIR" \
        "name=smoke_test_exp04" \
        "max_samples=20" \
        "diagnostic_subset=5" \
        "max_new_tokens=20"

echo "============================================"
echo " Smoke test PASSED: $(date)"
echo " Check: $RESULTS_DIR/smoke_test_exp04/"
echo "============================================"
