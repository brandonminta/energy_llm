# Sourced by every script in hpc/. Override any value by exporting it first.
#
# Resource values are the ones from the previously working jobs (gpu
# partition, a100-sxm4-40gb, 8 CPUs; cpu partition for CPU-only work). The
# only change approved for this restructure: extraction walltime 3 h.
# submit.sh passes these to sbatch on the command line, which overrides the
# #SBATCH fallback lines inside the .sbatch files.

# ---- REQUIRED ------------------------------------------------------------
# Scratch root: model cache and all run outputs live here. No default.
: "${SCRATCH_ROOT:?set SCRATCH_ROOT to your scratch directory (export SCRATCH_ROOT=...)}"
export SCRATCH_ROOT

# ---- Environment ---------------------------------------------------------
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_cache}"
MAMBA_BIN="${MAMBA_BIN:-$HOME/.local/bin/micromamba}"
MAMBA_ENV="${MAMBA_ENV:-hopfield-llm}"

# ---- GPU jobs (calibrate, extract) ---------------------------------------
PARTITION_GPU="${PARTITION_GPU:-gpu}"
# MIG trial: GRES=gpu:a100_2g.10gb:1 — that profile name comes from an old
# comment, not from a working job: check it with `sinfo -o '%P %G'`. On a
# 10 GB slice the float32 bank (~3 GB) does not fit beside the bf16 model;
# extraction.bank_device: auto then places it on the CPU automatically.
GRES="${GRES:-gpu:a100-sxm4-40gb:1}"
CPUS_GPU="${CPUS_GPU:-8}"
MEM_EXTRACT="${MEM_EXTRACT:-64G}"
TIME_EXTRACT="${TIME_EXTRACT:-03:00:00}"
MEM_CALIBRATE="${MEM_CALIBRATE:-80G}"
TIME_CALIBRATE="${TIME_CALIBRATE:-06:00:00}"

# ---- CPU job (merge) -----------------------------------------------------
PARTITION_CPU="${PARTITION_CPU:-cpu}"
CPUS_MERGE="${CPUS_MERGE:-4}"
MEM_MERGE="${MEM_MERGE:-16G}"
TIME_MERGE="${TIME_MERGE:-01:00:00}"

# ---- Not present in the previous jobs: set only if your cluster requires them
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"

# ---- Robustness ----------------------------------------------------------
SIGNAL_LEAD_S="${SIGNAL_LEAD_S:-120}"   # USR1 this many seconds before walltime
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"   # array throttle (%N)
MAX_REQUEUES="${MAX_REQUEUES:-5}"       # per task, via scontrol requeue

activate_env() {
    eval "$("$MAMBA_BIN" shell hook --shell bash)"
    micromamba activate "$MAMBA_ENV"
}

account_args() {
    local a=()
    [ -n "$ACCOUNT" ] && a+=(--account="$ACCOUNT")
    [ -n "$QOS" ] && a+=(--qos="$QOS")
    [ "${#a[@]}" -gt 0 ] && printf '%s\n' "${a[@]}"
    return 0
}

# Run one energy-llm step in the background, forward USR1/TERM to it, and
# requeue this job (once per USR1 / pending exit, up to MAX_REQUEUES) when
# Python reports work left (exit 75). This is the ONLY retry mechanism.
#   run_requeueable <job-id-for-scontrol> python -m energy_llm ...
run_requeueable() {
    local job="$1"; shift
    local pid=0 sig="" nsig=0 before status
    trap 'sig=USR1; nsig=$((nsig+1)); [ "$pid" -ne 0 ] && kill -USR1 "$pid" 2>/dev/null' USR1
    trap 'sig=TERM; nsig=$((nsig+1)); [ "$pid" -ne 0 ] && kill -TERM "$pid" 2>/dev/null' TERM
    "$@" &
    pid=$!
    while :; do
        before=$nsig
        wait "$pid"; status=$?
        if [ "$nsig" -ne "$before" ]; then           # wait was interrupted by a trap
            if kill -0 "$pid" 2>/dev/null; then continue; fi
            wait "$pid"; status=$?
            [ "$status" -eq 127 ] && status=75       # already reaped: assume work left
        fi
        break
    done
    echo "[$(date -Is)] python exit=$status signal=${sig:-none} restart=${SLURM_RESTART_COUNT:-0}"
    if [ "$status" -eq 75 ] && [ "$sig" != "TERM" ]; then
        if [ "${SLURM_RESTART_COUNT:-0}" -lt "$MAX_REQUEUES" ]; then
            echo "[$(date -Is)] work left: scontrol requeue $job"
            scontrol requeue "$job"
            return 0
        fi
        echo "[$(date -Is)] MAX_REQUEUES=$MAX_REQUEUES reached; leaving $job unfinished"
    fi
    # TERM = preemption (SLURM requeues via --requeue) or scancel (stays cancelled)
    return "$status"
}
