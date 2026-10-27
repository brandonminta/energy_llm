#!/bin/bash
# Submit calibration, or an extraction array + chained merge, from the repo root.
#
#   hpc/submit.sh calibrate <config.yaml> <warmup.jsonl>
#   hpc/submit.sh extract   <config.yaml> <input.jsonl> <name> [--only-missing]
#
# AFTER=<jobid> adds --dependency=afterok:<jobid> to the extraction array
# (e.g. the calibration job). Resource values come from hpc/env.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/env.sh
activate_env
mkdir -p logs
mapfile -t ACCT < <(account_args)

MODE="${1:?usage: submit.sh calibrate|extract ...}"; shift
case "$MODE" in
calibrate)
    CONFIG="${1:?config}"; WARMUP="${2:?warmup.jsonl}"
    JOB=$(sbatch --parsable "${ACCT[@]}" --partition="$PARTITION_GPU" --gres="$GRES" \
          --cpus-per-task="$CPUS_GPU" --mem="$MEM_CALIBRATE" --time="$TIME_CALIBRATE" \
          --signal="B:USR1@$SIGNAL_LEAD_S" hpc/calibrate.sbatch "$CONFIG" "$WARMUP")
    echo "calibrate -> job $JOB"
    ;;
extract)
    CONFIG="${1:?config}"; INPUT="${2:?input.jsonl}"; NAME="${3:?name}"; FLAG="${4:-}"
    if [ "$FLAG" = "--only-missing" ]; then
        SHARDS=$(python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME" --missing)
        if [ -z "$SHARDS" ]; then
            echo "no missing shards"
        fi
    else
        N=$(python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME" --num-shards)
        SHARDS="0-$((N - 1))"
    fi
    DEP=()
    [ -n "${AFTER:-}" ] && DEP=(--dependency=afterok:"$AFTER")
    if [ -n "$SHARDS" ]; then
        ARRAY=$(sbatch --parsable "${ACCT[@]}" "${DEP[@]}" --array="${SHARDS}%${MAX_CONCURRENT}" \
                --partition="$PARTITION_GPU" --gres="$GRES" --cpus-per-task="$CPUS_GPU" \
                --mem="$MEM_EXTRACT" --time="$TIME_EXTRACT" --signal="B:USR1@$SIGNAL_LEAD_S" \
                hpc/extract_array.sbatch "$CONFIG" "$INPUT" "$NAME")
        echo "extract  -> array job $ARRAY  shards=$SHARDS (gres=$GRES)"
        DEP=(--dependency=afterok:"$ARRAY")
    fi
    MERGE=$(sbatch --parsable "${ACCT[@]}" "${DEP[@]}" --partition="$PARTITION_CPU" \
            --cpus-per-task="$CPUS_MERGE" --mem="$MEM_MERGE" --time="$TIME_MERGE" \
            hpc/merge.sbatch "$CONFIG" "$INPUT" "$NAME")
    echo "merge    -> job $MERGE (afterok on the array)"
    ;;
*)
    echo "unknown mode $MODE" >&2; exit 2 ;;
esac
