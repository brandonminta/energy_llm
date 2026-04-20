#!/bin/bash
#
# HPC Setup Script — hopfield-llm
#
# Run once on the HPC cluster to set up the environment.
#
# Usage:
#   bash setup_hpc.sh                    # fresh install
#   bash setup_hpc.sh --update           # git pull + reinstall
#
set -euo pipefail

# ---- Configuration ----
REPO_URL="${REPO_URL:-}"  # Set to your GitHub URL, or leave empty for local-only
REPO_DIR="${REPO_DIR:-$HOME/hopfield-llm}"
ENV_NAME="${ENV_NAME:-hopfield-llm}"
SCRATCH_DIR="${SCRATCH:-/tmp}/hopfield_cache"
RESULTS_DIR="${SCRATCH:-/tmp}/hopfield_results"

UPDATE_ONLY=false
if [ "${1:-}" = "--update" ]; then
    UPDATE_ONLY=true
fi

echo "========================================="
echo " Hopfield-LLM HPC Setup"
echo " Repo:    $REPO_DIR"
echo " Env:     $ENV_NAME"
echo " Scratch: $SCRATCH_DIR"
echo "========================================="

# ---- Step 1: Load modules ----
echo "[1/6] Loading modules..."
module load cuda 2>/dev/null || echo "  Warning: cuda module not found (may already be in PATH)"
module load python 2>/dev/null || echo "  Warning: python module not found (may already be in PATH)"

# ---- Step 2: Clone or update repo ----
echo "[2/6] Setting up repository..."
if [ "$UPDATE_ONLY" = true ] && [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull --ff-only
    echo "  Updated existing repo."
elif [ -n "$REPO_URL" ] && [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    echo "  Cloned from $REPO_URL"
elif [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    echo "  Using existing repo at $REPO_DIR"
else
    echo "  ERROR: REPO_URL is empty and $REPO_DIR does not exist."
    echo "  Set REPO_URL or copy the repo manually."
    exit 1
fi

# ---- Step 3: Create virtual environment ----
echo "[3/6] Creating Python environment..."
if command -v micromamba &>/dev/null; then
    micromamba create -n "$ENV_NAME" python=3.12 -y 2>/dev/null || true
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "$ENV_NAME"
elif command -v conda &>/dev/null; then
    conda create -n "$ENV_NAME" python=3.12 -y 2>/dev/null || true
    conda activate "$ENV_NAME"
else
    echo "  No conda/micromamba found — using system Python + venv"
    python3 -m venv "$REPO_DIR/.venv"
    source "$REPO_DIR/.venv/bin/activate"
fi

echo "  Python: $(python --version)"
echo "  Pip:    $(pip --version)"

# ---- Step 4: Install dependencies ----
echo "[4/6] Installing dependencies..."
pip install --quiet -r requirements-hpc.txt
pip install --quiet -e .
echo "  Installed hopfield_llm in editable mode."

# ---- Step 5: Create data directories ----
echo "[5/6] Creating data directories..."
mkdir -p "$SCRATCH_DIR"/{banks,hidden_states,distributions}
mkdir -p "$RESULTS_DIR"
mkdir -p "$REPO_DIR/logs"
echo "  Cache:   $SCRATCH_DIR"
echo "  Results: $RESULTS_DIR"

# ---- Step 6: Environment variables ----
echo "[6/6] Setting environment variables..."
export HOPFIELD_ENV=hpc
export REPO_ROOT="$REPO_DIR"

# Write to a sourceable file for SLURM jobs
cat > "$REPO_DIR/.env.hpc" <<EOF
export HOPFIELD_ENV=hpc
export REPO_ROOT="$REPO_DIR"
export SCRATCH_DIR="$SCRATCH_DIR"
export RESULTS_DIR="$RESULTS_DIR"
EOF

echo ""
echo "========================================="
echo " Setup complete!"
echo ""
echo " Activate environment:"
echo "   micromamba activate $ENV_NAME"
echo "   source $REPO_DIR/.env.hpc"
echo ""
echo " Verify installation:"
echo "   python -c 'import hopfield_llm; print(hopfield_llm.__version__)'"
echo "   hopfield-llm --help"
echo ""
echo " Submit pipeline:"
echo "   cd $REPO_DIR && ./scripts/hpc/submit_pipeline.sh"
echo "========================================="

# Quick verification
echo ""
echo "Running verification..."
python -c "
import hopfield_llm
print(f'  hopfield_llm v{hopfield_llm.__version__} imported OK')
from hopfield_llm.models.profiles import MODEL_PROFILES
print(f'  {len(MODEL_PROFILES)} model profiles loaded')
from hopfield_llm.utils.env import load_env_config
cfg = load_env_config()
print(f'  Environment: {cfg.environment}')
print(f'  Batch size:  {cfg.default_batch_size}')
print('  All checks passed.')
"
