#!/bin/bash
# Report shard progress for an extraction. With --resubmit:
#   1. cancel the still-pending merge job(s) of earlier submissions (after a
#      task runs out of requeues they sit in DependencyNeverSatisfied forever);
#   2. relaunch only the unfinished shards that have no active task left
#      (a shard is never given two concurrent writers);
#   3. submit a new merge with afterok on the relaunched array AND on the
#      earlier tasks that are still pending/running.
#
#   hpc/missing_shards.sh <config.yaml> <input.jsonl> <name> [--resubmit]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/env.sh
activate_env
mkdir -p logs
CONFIG="${1:?config}"; INPUT="${2:?input.jsonl}"; NAME="${3:?name}"
status() { python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME" "$@"; }

status
[ "${4:-}" = "--resubmit" ] || exit 0

EXT_DIR=$(status --extraction-dir)
LOG="$EXT_DIR/submissions.log"
MISSING=$(status --missing)

ARRAYS=(); MERGES=()
if [ -f "$LOG" ]; then
    mapfile -t ARRAYS < <(sed -n 's/.* array=\([0-9][0-9]*\) .*/\1/p' "$LOG")
    mapfile -t MERGES < <(sed -n 's/.* merge=\([0-9][0-9]*\) .*/\1/p' "$LOG")
fi

# 1. cancel pending merges of earlier submissions
for M in "${MERGES[@]}"; do
    STATE=$(squeue -h -j "$M" -o "%T|%r" 2>/dev/null || true)
    if [[ "$STATE" == PENDING* ]]; then
        echo "cancelling pending merge $M (${STATE#*|})"
        scancel "$M"
    fi
done

# 2. active tasks of earlier arrays: never resubmit those shards
ACTIVE_TASKS=(); ACTIVE_SHARDS=" "
for A in "${ARRAYS[@]}"; do
    while read -r JOBID STATE; do
        [ -z "${JOBID:-}" ] && continue
        TASK="${JOBID##*_}"
        [[ "$TASK" =~ ^[0-9]+$ ]] || continue          # unexpanded ranges are skipped
        ACTIVE_TASKS+=("$JOBID"); ACTIVE_SHARDS+="$TASK "
        echo "shard $TASK still active as $JOBID ($STATE)"
    done < <(squeue -h -r -j "$A" -o "%i %T" 2>/dev/null || true)
done

TODO=()
IFS=',' read -ra MISS <<< "$MISSING"
for K in "${MISS[@]}"; do
    [ -z "$K" ] && continue
    [[ "$ACTIVE_SHARDS" == *" $K "* ]] || TODO+=("$K")
done

DEPS=""
if [ "${#TODO[@]}" -gt 0 ]; then
    SHARDS=$(IFS=','; echo "${TODO[*]}")
    ARRAY=$(submit_extract_array "$CONFIG" "$INPUT" "$NAME" "$SHARDS")
    echo "relaunched shards $SHARDS -> array job $ARRAY"
    DEPS="$ARRAY"
else
    SHARDS=""; ARRAY=""
    echo "no shard to relaunch"
fi
for T in "${ACTIVE_TASKS[@]}"; do DEPS="${DEPS:+$DEPS:}$T"; done

# 3. new merge
MERGE=$(submit_merge "$CONFIG" "$INPUT" "$NAME" "${DEPS:+afterok:$DEPS}")
record_submission "$EXT_DIR" "$ARRAY" "$MERGE" "$SHARDS"
echo "merge -> job $MERGE${DEPS:+ (afterok:$DEPS)}"
