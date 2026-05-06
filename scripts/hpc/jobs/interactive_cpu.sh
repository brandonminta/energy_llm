#!/bin/bash
#
# Interactive CPU session — allocates a node and activates the environment.
# Use this for CPU-only work: analyze, visualize, label, evaluate, debugging.
#
# Usage:
#   bash scripts/hpc/jobs/interactive_cpu.sh
#
# Optional overrides:
#   CPUS=8 MEM=16G TIME=01:00:00 bash scripts/hpc/jobs/interactive_cpu.sh
#
CPUS="${CPUS:-4}"
MEM="${MEM:-8G}"
TIME="${TIME:-00:30:00}"
PARTITION="${PARTITION:-cpu-dev}"

echo "Requesting: $PARTITION | $CPUS CPUs | $MEM RAM | $TIME"
echo "Waiting for allocation..."

salloc -p "$PARTITION" -c "$CPUS" --mem="$MEM" --time="$TIME" \
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
