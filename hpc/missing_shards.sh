#!/bin/bash
# Report shard progress for an extraction; with --resubmit, relaunch only the
# unfinished shards (plus a chained merge).
#
#   hpc/missing_shards.sh <config.yaml> <input.jsonl> <name> [--resubmit]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/env.sh
activate_env
CONFIG="${1:?config}"; INPUT="${2:?input.jsonl}"; NAME="${3:?name}"
python -m energy_llm status --config "$CONFIG" --input "$INPUT" --name "$NAME"
if [ "${4:-}" = "--resubmit" ]; then
    hpc/submit.sh extract "$CONFIG" "$INPUT" "$NAME" --only-missing
fi
