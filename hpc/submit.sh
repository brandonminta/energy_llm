#!/bin/bash
# Submit calibration, a smoke test, or an extraction array + chained merge,
# from the repo root.
#
#   hpc/submit.sh calibrate <config.yaml> <warmup.jsonl>
#   hpc/submit.sh smoke     <config.yaml> <input.jsonl> [N]
#   hpc/submit.sh extract   <config.yaml> <input.jsonl> <name> [--only-missing]
#
# AFTER=<jobid> adds --dependency=afterok:<jobid> to the extraction array
# (e.g. the calibration job). --only-missing is the same as
# hpc/missing_shards.sh ... --resubmit. Resource values come from hpc/env.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/env.sh
activate_env
mkdir -p logs
mapfile -t ACCT < <(account_args)

MODE="${1:?usage: submit.sh calibrate|smoke|extract ...}"; shift
case "$MODE" in
calibrate)
    CONFIG="${1:?config}"; WARMUP="${2:?warmup.jsonl}"
    JOB=$(sbatch --parsable "${ACCT[@]}" --partition="$PARTITION_GPU" --gres="$GRES" \
          --cpus-per-task="$CPUS_GPU" --mem="$MEM_CALIBRATE" --time="$TIME_CALIBRATE" \
          --signal="B:USR1@$SIGNAL_LEAD_S" hpc/calibrate.sbatch "$CONFIG" "$WARMUP")
    echo "calibrate -> job $JOB"
    ;;
smoke)
    CONFIG="${1:?config}"; INPUT="${2:?input.jsonl}"; N="${3:-20}"
    JOB=$(sbatch --parsable "${ACCT[@]}" --partition="$PARTITION_GPU" --gres="$GRES" \
          --cpus-per-task="$CPUS_GPU" --mem="$MEM_EXTRACT" --time="$TIME_SMOKE" \
          hpc/smoke.sbatch "$CONFIG" "$INPUT" "$N")
    echo "smoke -> job $JOB (logs/smoke_$JOB.out)"
    ;;
extract)
    CONFIG="${1:?config}"; INPUT="${2:?input.jsonl}"; NAME="${3:?name}"
    if [ "${4:-}" = "--only-missing" ]; then
        exec hpc/missing_shards.sh "$CONFIG" "$INPUT" "$NAME" --resubmit
    fi
    EXT_DIR=$(python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME" --extraction-dir)
    N=$(python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME" --num-shards)
    SHARDS="0-$((N - 1))"
    ARRAY=$(submit_extract_array "$CONFIG" "$INPUT" "$NAME" "$SHARDS" "${AFTER:+afterok:$AFTER}")
    MERGE=$(submit_merge "$CONFIG" "$INPUT" "$NAME" "afterok:$ARRAY")
    record_submission "$EXT_DIR" "$ARRAY" "$MERGE" "$SHARDS"
    echo "extract -> array job $ARRAY shards=$SHARDS (gres=$GRES)"
    echo "merge   -> job $MERGE (afterok:$ARRAY)"
    ;;
*)
    echo "unknown mode $MODE" >&2; exit 2 ;;
esac
