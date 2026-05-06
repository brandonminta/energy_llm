#!/bin/bash
#
# Interactive session — opens a shell directly on a compute node.
# Use for CPU-only work: analyze, visualize, label, evaluate, debugging.
#
# Usage:
#   bash scripts/hpc/jobs/interactive_cpu.sh
#
# Optional overrides:
#   CPUS=8 MEM=32G TIME=01:00:00 bash scripts/hpc/jobs/interactive_cpu.sh
#
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
TIME="${TIME:-00:30:00}"
PARTITION="${PARTITION:-gpu-dev}"
GPU="${GPU:-a100-sxm4-40gb}"
NODE="${NODE:-compute-0-1}"

echo "Requesting: $PARTITION | $NODE | $CPUS CPUs | $MEM RAM | $TIME"
echo "Waiting for allocation..."

# Write the activation script to $HOME (NFS-mounted, visible on all nodes).
# /tmp is node-local so the compute node can't read a file created there.
INIT="$HOME/.hopfield_interactive_init.sh"
cat > "$INIT" << EOF
eval "\$(\$HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate "\${MAMBA_ENV_NAME:-hopfield-llm}"
export HOPFIELD_ENV=hpc
echo ""
echo "============================================"
echo " Interactive session ready"
echo " Node: \$(hostname)"
echo " Env:  hopfield-llm (activated)"
echo " Type 'exit' to release the allocation"
echo "============================================"
echo ""
EOF

srun \
    -p "$PARTITION" \
    -c "$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --gres=gpu:"$GPU":1 \
    --nodelist="$NODE" \
    --job-name=hopfield-interactive \
    --pty bash --init-file "$INIT"

rm -f "$INIT"
