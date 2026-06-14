#!/bin/bash
#SBATCH --job-name=energy-traj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/traj_%A_%a.out
#SBATCH --error=logs/traj_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#
# Stage 2 — trajectory capture, SLURM array over non-overlapping shards.
# Resource directives (partition/GRES/cpus/mem/time/array) preserved
# verbatim from the previous working setup (scripts/hpc/templates/
# forward_pass.sh). Added for robustness:
#   --signal=B:USR1@120  SLURM warns the batch shell 120 s before walltime;
#                        the trap below forwards USR1 to Python, which
#                        finishes the current sample, flushes, and exits in
#                        a resumable state.
#   --requeue            preempted jobs are requeued automatically.
#
# Resume-by-default: completed samples are skipped, so resubmitting after a
# kill/OOM/timeout is always safe and is a no-op for finished samples.
#
# MEMORY NOTE for the 10 GB a100_2g.10gb MIG slice: a 3B model in BF16
# (~6.2 GB) plus float32 banks (~3.2 GB for Qwen2.5-3B) is tight. If Stage 2
# hits GPU OOM here, set SCORE_DTYPE=bfloat16 below (the energy/divergence
# math still runs in float64 on CPU; only the bank matmul precision drops),
# or run on a full A100 (use the calibrate.sbatch directives).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch hpc/trajectories.sbatch configs/e1.yaml
#   NUM_SHARDS must match the array width: --array=0-(N-1)%4
#
set -euo pipefail

CONFIG="${1:?usage: sbatch hpc/trajectories.sbatch <config.yaml>}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SCORE_DTYPE="${SCORE_DTYPE:-float32}"
MAX_SAMPLES_PER_JOB="${MAX_SAMPLES_PER_JOB:-}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"

cd "$REPO_ROOT"
mkdir -p logs

echo "[traj] shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} config=$CONFIG node=$(hostname)"
echo "[traj] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

EXTRA_ARGS=()
if [ -n "$MAX_SAMPLES_PER_JOB" ]; then
    EXTRA_ARGS+=(--max-samples-per-job "$MAX_SAMPLES_PER_JOB")
fi

# Forward SLURM's pre-walltime USR1 (and TERM on preemption) to Python so it
# can finish the current sample and flush before the hard kill.
PYTHON_PID=0
forward() { [ "$PYTHON_PID" -ne 0 ] && kill -"$1" "$PYTHON_PID" 2>/dev/null || true; }
trap 'forward USR1' USR1
trap 'forward TERM' TERM

MAX_RETRIES=3
ATTEMPT=0
STATUS=1

while [ $ATTEMPT -lt $MAX_RETRIES ] && [ $STATUS -ne 0 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "[traj] attempt $ATTEMPT/$MAX_RETRIES shard=${SLURM_ARRAY_TASK_ID} $(date)"

    python scripts/capture_trajectories.py \
        --config "$CONFIG" \
        --shard-index "$SLURM_ARRAY_TASK_ID" \
        --num-shards "$NUM_SHARDS" \
        --score-dtype "$SCORE_DTYPE" \
        "${EXTRA_ARGS[@]}" &
    PYTHON_PID=$!
    wait "$PYTHON_PID"
    STATUS=$?

    if [ $STATUS -ne 0 ] && [ $ATTEMPT -lt $MAX_RETRIES ]; then
        echo "[traj] attempt $ATTEMPT failed (status $STATUS), waiting 60s before retry"
        sleep 60
    fi
done

echo "[traj] shard ${SLURM_ARRAY_TASK_ID} exited with status $STATUS at $(date)"

# --- OPTIONAL self-resubmit pattern for short-walltime partitions ---------
# Pair with MAX_SAMPLES_PER_JOB to chunk work across jobs. Disabled by
# default to avoid accidental job storms; export SELF_RESUBMIT=1 to enable.
if [ "${SELF_RESUBMIT:-0}" = "1" ] && [ "$STATUS" -eq 0 ]; then
    PENDING=$(python - "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$NUM_SHARDS" <<'EOF'
import sys
sys.path.insert(0, ".")
from energy_llm.config import load_config
from energy_llm.data import load_samples, shard
from energy_llm.io_artifacts import SampleManifest
cfg = load_config(sys.argv[1])
samples = shard(load_samples(cfg.dataset.name, cfg.dataset.num_samples, cfg.seed),
                int(sys.argv[2]), int(sys.argv[3]))
print(len(SampleManifest(cfg.trajectories_dir).pending([s.sample_id for s in samples])))
EOF
)
    if [ "$PENDING" -gt 0 ]; then
        echo "[traj] $PENDING samples pending — self-resubmitting shard ${SLURM_ARRAY_TASK_ID}"
        sbatch --array="${SLURM_ARRAY_TASK_ID}" "$0" "$CONFIG"
    fi
fi

exit "$STATUS"
