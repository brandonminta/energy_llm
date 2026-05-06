#!/bin/bash
#SBATCH --job-name=hopfield-exp04-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --nodelist=compute-0-1
#SBATCH --output=logs/exp04_resume_%j.out
#SBATCH --error=logs/exp04_resume_%j.err
#
# Resume exp04 after OOM kill at sample 294/500.
#
# Points directly at the existing run directory so the trajectory stage
# auto-skips the 294 already-processed samples.
# Banks are already extracted; calibrated β=68.16 is used directly.
#
# Usage (from repo root):
#   mkdir -p logs
#   sbatch scripts/hpc/jobs/run_exp04_resume.sh
#
set -euo pipefail

RUN_DIR="$HOME/hopfield_results/exp04_qwen_chat_template_fix/20260506_142057_qwen25_3b_truthfulqa_up_b15.0_cd3721"
TRAJ_DIR="$RUN_DIR/trajectories"
BANKS="$RUN_DIR/banks.pt"
ANALYSIS="$RUN_DIR/analysis.json"
PLOTS_DIR="$RUN_DIR/plots"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

echo "============================================"
echo " exp04 RESUME — remaining samples"
echo " Node:    $(hostname)"
echo " Date:    $(date)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Run dir: $RUN_DIR"
echo "============================================"

mkdir -p logs

# Stage 2 — trajectory (auto-resumes, skips already-processed samples)
# β=68.16 from prior calibration; no need to recalibrate
hopfield-llm run-trajectory \
    --model qwen25_3b \
    --dataset truthfulqa \
    --max-samples 500 \
    --seed 42 \
    --banks "$BANKS" \
    --output "$TRAJ_DIR" \
    --beta 68.16 \
    --energy-mode dot \
    --max-new-tokens 50 \
    --diagnostic-subset 0

# Stage 3 — analyze (runs over all 500 completed samples)
hopfield-llm analyze \
    --trajectories "$TRAJ_DIR" \
    --output "$ANALYSIS" \
    --score-metric delta_energy

# Stage 4 — visualize
hopfield-llm visualize \
    --analysis "$ANALYSIS" \
    --output "$PLOTS_DIR"

echo "============================================"
echo " exp04 Resume Done: $(date)"
echo " Results: $RUN_DIR"
echo "============================================"
