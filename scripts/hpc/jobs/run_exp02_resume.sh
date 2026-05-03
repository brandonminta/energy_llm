#!/bin/bash
#SBATCH --job-name=hopfield-exp02-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp02_resume_%j.out
#SBATCH --error=logs/exp02_resume_%j.err
#
# Resume exp02 — continúa desde los 347 samples ya procesados en el directorio existente.
# El pipeline detecta automáticamente los JSON existentes y los salta.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp02_resume.sh
#
set -euo pipefail

RUN_DIR="$HOME/hopfield_results/exp02_truthfulqa_500/20260424_153142_qwen25_3b_truthfulqa_up_b15.0_a2dfcc"

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

export HOPFIELD_ENV=hpc

echo "============================================"
echo " exp02 RESUME — TruthfulQA 500, Qwen2.5-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Run dir: $RUN_DIR"
echo " Traj done: $(find "$RUN_DIR/trajectories" -name '*.json' | wc -l) / 500"
echo "============================================"

# Stage 2 — complete the remaining ~153 samples (skips the 347 already done)
hopfield-llm run-trajectory \
    --model qwen25_3b \
    --dataset truthfulqa \
    --banks "$RUN_DIR/banks.pt" \
    --output "$RUN_DIR/trajectories" \
    --beta 15.0 \
    --threshold 0.1 \
    --energy-mode dot \
    --max-samples 500 \
    --seed 42 \
    --max-new-tokens 50 \
    --diagnostic-subset 20

echo "--- Trajectories complete: $(find "$RUN_DIR/trajectories" -name '*.json' | wc -l) / 500"

# Stage 3 — analyze
hopfield-llm analyze \
    --trajectories "$RUN_DIR/trajectories" \
    --output "$RUN_DIR/analysis.json" \
    --score-metric js_gen_to_prompt_per_layer \
    --score-aggregation mean

# Stage 4 — visualize
hopfield-llm visualize \
    --analysis "$RUN_DIR/analysis.json" \
    --output "$RUN_DIR/plots"

echo "============================================"
echo " Done: $(date)"
echo " Output: $RUN_DIR"
echo "============================================"
