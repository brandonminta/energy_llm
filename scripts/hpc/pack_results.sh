#!/bin/bash
#
# Pack exp02 + exp03 results into a single tar.gz, excluding heavy files
# (banks.pt, download_subset*, *.pt).
#
# Usage:  bash pack_results.sh
# Output: ~/hopfield_results/exp02_exp03_results.tar.gz
#
set -euo pipefail

OUTPUT=~/hopfield_results/exp02_exp03_results.tar.gz

RUNDIR_02="hopfield_results/exp02_truthfulqa_500/20260424_153142_qwen25_3b_truthfulqa_up_b15.0_a2dfcc"
RUNDIR_03="hopfield_results/exp03_truthfulqa_500/20260424_191203_llama32_3b_truthfulqa_up_b15.0_0ea592"

echo "Packing results (excluding banks + download_subset)..."

tar -czvf "$OUTPUT" \
    --exclude="*.pt" \
    --exclude="download_subset" \
    --exclude="download_subset.tar.gz" \
    -C "$HOME" \
    "$RUNDIR_02/trajectories" \
    "$RUNDIR_02/analysis.json" \
    "$RUNDIR_02/plots" \
    "$RUNDIR_02/config.yaml" \
    "$RUNDIR_02/git_info.json" \
    "$RUNDIR_03/trajectories" \
    "$RUNDIR_03/analysis.json" \
    "$RUNDIR_03/plots" \
    "$RUNDIR_03/config.yaml" \
    "$RUNDIR_03/git_info.json"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "Done → $OUTPUT  ($SIZE)"
echo ""
echo "Download with:"
echo "  scp ${USER}@login1.hpc.cedia.edu.ec:$OUTPUT ."
