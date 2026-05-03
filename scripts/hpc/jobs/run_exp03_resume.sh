#!/bin/bash
#SBATCH --job-name=hopfield-exp03-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp03_resume_%j.out
#SBATCH --error=logs/exp03_resume_%j.err
#
# Resume exp03 — modelo ya cacheado, banks ya construidos.
# Continúa desde los 4 samples ya procesados en el directorio existente.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp03_resume.sh
#
set -euo pipefail

RUN_DIR="$HOME/hopfield_results/exp03_truthfulqa_500/20260424_191203_llama32_3b_truthfulqa_up_b15.0_0ea592"

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

export HOPFIELD_ENV=hpc

echo "============================================"
echo " exp03 RESUME — TruthfulQA 500, Llama 3.2-3B"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Run dir: $RUN_DIR"
echo " Traj done: $(find "$RUN_DIR/trajectories" -name '*.json' ! -name '*scores*' | wc -l) / 500"
echo "============================================"

# Stage 2 — banks ya existen, modelo ya cacheado, skip automático de los 4 samples ya procesados
hopfield-llm run-trajectory \
    --model llama32_3b \
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

echo "--- Trajectories: $(find "$RUN_DIR/trajectories" -name '*.json' ! -name '*scores*' | wc -l) / 500"

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
