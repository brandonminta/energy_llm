#!/bin/bash
#SBATCH --job-name=hopfield-fix-analyze
#SBATCH --partition=cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/brandon.minta__yachaytech.edu.ec/energy_llm/logs/fix_analyze_%j.out
#SBATCH --error=/home/brandon.minta__yachaytech.edu.ec/energy_llm/logs/fix_analyze_%j.err
#
# Re-run stage 3 (analyze + visualize) for exp02 and exp03 with the correct
# score metric (delta_energy instead of js, which requires distributions not
# stored in the .npz files).
#
set -euo pipefail

RUNDIR_02="/home/brandon.minta__yachaytech.edu.ec/hopfield_results/exp02_truthfulqa_500/20260424_153142_qwen25_3b_truthfulqa_up_b15.0_a2dfcc"
RUNDIR_03="/home/brandon.minta__yachaytech.edu.ec/hopfield_results/exp03_truthfulqa_500/20260424_191203_llama32_3b_truthfulqa_up_b15.0_0ea592"

export HOPFIELD_ENV=hpc

eval "$($HOME/.local/bin/micromamba shell hook --shell bash)"
micromamba activate hopfield-llm

echo "============================================"
echo " Fix: re-analyze exp02 + exp03"
echo " score_metric: delta_energy"
echo " Node: $(hostname)"
echo " Date: $(date)"
echo "============================================"

for DIR in "$RUNDIR_02" "$RUNDIR_03"; do
    EXP=$(basename "$(dirname "$DIR")")
    echo ""
    echo "--- $EXP ---"

    hopfield-llm analyze \
        --trajectories "$DIR/trajectories" \
        --output       "$DIR/analysis.json" \
        --score-metric delta_energy \
        --score-aggregation mean

    hopfield-llm visualize \
        --analysis "$DIR/analysis.json" \
        --output   "$DIR/plots"

    echo "Done → $DIR/analysis.json"
done

echo ""
echo "============================================"
echo " Finished: $(date)"
echo "============================================"
