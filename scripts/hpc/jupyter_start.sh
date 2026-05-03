#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/brandon.minta__yachaytech.edu.ec/energy_llm/logs/jupyter_%j.out
#SBATCH --error=/home/brandon.minta__yachaytech.edu.ec/energy_llm/logs/jupyter_%j.err
#
# Start JupyterLab on a compute node and print the SSH tunnel command.
#
set -euo pipefail

PORT=8888
NODE=$(hostname)
LOGIN=login1.hpc.cedia.edu.ec

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate hopfield-llm

echo "============================================"
echo " JupyterLab"
echo " Node:  $NODE"
echo " Port:  $PORT"
echo ""
echo " Run this on your LOCAL machine to tunnel:"
echo ""
echo "   ssh -L ${PORT}:${NODE}:${PORT} ${USER}@${LOGIN}"
echo ""
echo " Then open: http://localhost:${PORT}"
echo "============================================"

jupyter lab \
    --no-browser \
    --port=$PORT \
    --ip=0.0.0.0 \
    --notebook-dir=/home/brandon.minta__yachaytech.edu.ec/energy_llm
