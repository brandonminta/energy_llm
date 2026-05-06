#!/bin/bash
#
# Interactive session — allocates a GPU node and activates the environment.
# Use for CPU-only work (analyze, visualize, label, evaluate, debugging).
#
# Must use a GPU node (gpu-dev) even for CPU work because the CPU partition
# nodes run an older glibc that is incompatible with pyarrow/bitsandbytes.
#
# Usage:
#   bash scripts/hpc/jobs/interactive_cpu.sh
#
# Optional overrides:
#   CPUS=8 MEM=16G TIME=01:00:00 bash scripts/hpc/jobs/interactive_cpu.sh
#
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
TIME="${TIME:-00:30:00}"
PARTITION="${PARTITION:-gpu-dev}"
GPU="${GPU:-a100_1g.5gb}"        # smallest slice on compute-0-0, allocates fast
NODE="${NODE:-compute-0-0}"      # has 24 a100_1g.5gb slices, least contention

echo "Requesting: $PARTITION | $NODE | $CPUS CPUs | $MEM RAM | $TIME | GPU: $GPU"
echo "Waiting for allocation..."

salloc -p "$PARTITION" -c "$CPUS" --mem="$MEM" --time="$TIME" \
    --gres=gpu:"$GPU":1 \
    --nodelist="$NODE" \
    --job-name=hopfield-interactive \
    bash --rcfile <(cat <<'EOF'
eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "${MAMBA_ENV_NAME:-hopfield-llm}"
export HOPFIELD_ENV=hpc
echo ""
echo "============================================"
echo " Interactive session ready"
echo " Node: $(hostname)"
echo " Env:  hopfield-llm (activated)"
echo " Type 'exit' to release the allocation"
echo "============================================"
echo ""
EOF
)
