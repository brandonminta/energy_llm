#!/bin/bash
#SBATCH --job-name=energy-verify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#
# Verification only (never a production extraction): checks the pipeline
# against the model's own computation and against dev/reference.py on a few
# questions, prints PASS/FAIL per check, writes verify_report.json, exits 1 on
# any FAIL. Needs a calibrated beta*. Submit with hpc/submit.sh verify.
# Walltime: the float64 reference takes ~75 s per layer and question at
# Qwen2.5-3B size (measured on a laptop core); defaults (2 questions x 3
# layers) need ~8 min of it, plus model load and the other checks.
set -euo pipefail
CONFIG="${1:?usage: verify.sbatch <config.yaml> <input.jsonl> [N]}"
INPUT="${2:?}"
N="${3:-4}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
activate_env

echo "[verify] job=$SLURM_JOB_ID node=$(hostname) N=$N config=$CONFIG"
echo "[verify] gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
python -m energy_llm verify --config "$CONFIG" --input "$INPUT" --n "$N"
