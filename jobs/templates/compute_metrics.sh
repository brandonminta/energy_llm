#!/bin/bash
#SBATCH --job-name=hopfield-metrics
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/metrics_%j.out
#SBATCH --error=logs/metrics_%j.err
#
# Stages 3-5: Score + Analyze + Visualize
#
# Resources justification:
#   - gpu-dev (96h limit) — scoring is matrix operations, not full inference
#   - a100_1g.5gb — GPU useful for batched matmul but not strictly required
#   - 16 CPUs, 64 GB RAM — parallelisable scoring over cached hidden states,
#     RAM for loading large score artifacts with many samples
#   - 8h — conservative; typical scoring of 818 samples takes <15 min
#
set -euo pipefail

# ---- Configuration ----
MODEL="${MODEL:-qwen25_3b}"
BANK_PATH="${BANK_PATH:-$SCRATCH/hopfield_cache/banks/${MODEL}__gate.pt}"
HIDDEN_PATH="${HIDDEN_PATH:-$SCRATCH/hopfield_cache/hidden_states/${MODEL}__mean.pt}"
BETA="${BETA:-15.0}"
METRIC="${METRIC:-js}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/hopfield_results}"

# ---- Setup ----
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HOPFIELD_ENV=hpc

mkdir -p "$OUTPUT_DIR" logs

SCORES="$OUTPUT_DIR/${MODEL}__beta${BETA}__${METRIC}__scores.pt"
ANALYSIS="$OUTPUT_DIR/${MODEL}__beta${BETA}__${METRIC}__analysis.json"
PLOTS="$OUTPUT_DIR/${MODEL}__beta${BETA}__${METRIC}__plots/"

echo "[compute_metrics] Scoring..."
python -m hopfield_llm.cli.run score \
    --memory "$BANK_PATH" \
    --hidden "$HIDDEN_PATH" \
    --output "$SCORES" \
    --beta "$BETA" \
    --score-metric "$METRIC"

echo "[compute_metrics] Analyzing..."
python -m hopfield_llm.cli.run analyze \
    --scores "$SCORES" \
    --output "$ANALYSIS"

echo "[compute_metrics] Visualizing..."
python -m hopfield_llm.cli.run visualize \
    --analysis "$ANALYSIS" \
    --output "$PLOTS"

echo "[compute_metrics] Done. Results in $OUTPUT_DIR"
