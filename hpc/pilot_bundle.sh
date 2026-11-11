#!/bin/bash
# Collect what the pilot review needs from ONE verify job and ONE smoke job
# into ~/pilot_bundle_<date>.txt, in one file to send or paste:
#   sacct of both jobs, head + tail of their .out, tail of their .err, and the
#   JSON reports (verify_report.json, smoke_report.json, manifest.json) of the
#   run directories THOSE jobs wrote. The run directory is read from the
#   "report: <dir>/<x>_report.json" line of each job's log; nothing is
#   guessed with find.
#
#   hpc/pilot_bundle.sh <verify_jobid> <smoke_jobid>
#
# Login node: bash >= 4.2, coreutils/grep/sed, no Python packages (JSON is
# copied as is, pretty-printed with the stdlib's python3 -m json.tool when
# python3 exists). Logs are read from logs/ under the repository root
# (hpc/submit.sh writes them there); override with LOG_DIR=..., TAIL_N=...
set -uo pipefail

VJOB="${1:-}"
SJOB="${2:-}"
if [ -z "$VJOB" ] || [ -z "$SJOB" ]; then
    echo "usage: hpc/pilot_bundle.sh <verify_jobid> <smoke_jobid>" >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
TAIL_N="${TAIL_N:-150}"
OUT="$HOME/pilot_bundle_$(date +%Y%m%d_%H%M%S).txt"

section() { printf '\n==================== %s ====================\n' "$*"; }

show_file() {   # show_file PATH [tail|head|all] [N]
    local path="$1" mode="${2:-all}" n="${3:-$TAIL_N}"
    if [ ! -f "$path" ]; then
        echo "(missing: $path)"
        return 0
    fi
    case "$mode" in
        tail) tail -n "$n" "$path" ;;
        head) head -n "$n" "$path" ;;
        *) cat "$path" ;;
    esac
}

show_json() {   # show_json PATH
    local path="$1" pretty=""
    if [ ! -f "$path" ]; then
        echo "(missing: $path)"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        pretty="$(python3 -m json.tool "$path" 2>/dev/null)" || pretty=""
    fi
    if [ -n "$pretty" ]; then
        printf '%s\n' "$pretty"
    else
        cat "$path"
    fi
}

run_dir_of() {   # run_dir_of LOG REPORT_NAME -> the directory printed by that job, or empty
    local log="$1" name="$2"
    [ -f "$log" ] || return 0
    grep -o "report: [^ ]*/$name" "$log" | tail -n 1 | sed -e 's/^report: //' -e "s#/$name\$##"
}

job_block() {   # job_block KIND JOBID REPORT_NAME
    local kind="$1" job="$2" name="$3" out err dir
    out="$LOG_DIR/${kind}_${job}.out"
    err="$LOG_DIR/${kind}_${job}.err"
    section "$kind job $job: $out (first 20 lines)"
    show_file "$out" head 20
    section "$kind job $job: $out (last $TAIL_N lines)"
    show_file "$out" tail
    section "$kind job $job: $err (last $TAIL_N lines)"
    show_file "$err" tail
    dir="$(run_dir_of "$out" "$name")"
    section "$kind job $job: run directory"
    if [ -z "$dir" ]; then
        echo "(no 'report: .../$name' line in $out: the job did not finish its report)"
        return 0
    fi
    echo "$dir"
    section "$kind job $job: $dir/$name"
    show_json "$dir/$name"
    section "$kind job $job: $dir/manifest.json"
    show_json "$dir/manifest.json"
}

{
    section "pilot bundle $(date -Is) host=$(hostname) user=${USER:-?}"
    echo "repo: $REPO_ROOT"
    git -C "$REPO_ROOT" log -1 --format='commit %H %ad %s' 2>/dev/null || echo "(git log unavailable)"
    git -C "$REPO_ROOT" status --short 2>/dev/null | head -n 20
    section "sacct -j $VJOB,$SJOB"
    if command -v sacct >/dev/null 2>&1; then
        sacct -j "$VJOB,$SJOB" \
            --format=JobID%18,JobName%16,Partition%10,State%14,ExitCode,Elapsed,MaxRSS,ReqMem,NodeList%16,Start,End \
            2>&1
    else
        echo "(sacct not available)"
    fi
    job_block verify "$VJOB" verify_report.json
    job_block smoke "$SJOB" smoke_report.json
} > "$OUT" 2>&1

echo "$OUT ($(wc -c < "$OUT") bytes)"
