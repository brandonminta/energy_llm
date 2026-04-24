# Setup

## Local Development (Linux with RTX 2050)

### Prerequisites

- Python 3.10+ (3.12 recommended)
- CUDA toolkit (for GPU inference)
- micromamba or conda (recommended) or plain venv

### Installation

```bash
# Clone
git clone <your-repo-url> hopfield-llm
cd hopfield-llm

# Create environment
micromamba create -n hopfield-llm python=3.12 -y
micromamba activate hopfield-llm

# Install
pip install -r requirements.txt
pip install -e .

# Verify
python -c "import hopfield_llm; print(hopfield_llm.__version__)"
hopfield-llm --help
```

### Notes

- `bitsandbytes` is included for 4-bit quantization on Linux GPU. If unavailable, models load in fp16/fp32.
- The RTX 2050 (4 GB VRAM) supports Qwen2.5-1.5B and Qwen2.5-3B in 4-bit. Larger models will fail or fall back to CPU.
- Batch size should stay at 1 for 4 GB VRAM.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOPFIELD_ENV` | auto-detect | `local` or `hpc` |
| `CUDA_VISIBLE_DEVICES` | all | Restrict visible GPUs |

## HPC Installation (SLURM Cluster)

### Automated Setup

```bash
# Copy setup script to the cluster home, then:
bash setup_hpc.sh
```

This script will:
1. Load CUDA and Python modules (cluster-specific)
2. Create a micromamba/conda environment with `requirements-hpc.txt`
3. Install `hopfield_llm` in editable mode
4. Create cache and results directories on `$SCRATCH`
5. Write `.env.hpc` with all necessary environment variables
6. Run a verification test

### Manual Setup

```bash
# SSH into the cluster login node
ssh user@cluster.edu

# Load modules (cluster-specific, e.g., XSEDE/HPC)
module load cuda/12.0 python/3.12

# Create environment
micromamba create -n hopfield-llm python=3.12 -y
micromamba activate hopfield-llm

# Install dependencies (HPC variant includes distributed tools)
pip install -r requirements-hpc.txt
pip install -e .

# Create directories on fast storage (SCRATCH, not home)
mkdir -p $SCRATCH/hopfield_cache/{banks,hidden_states,distributions}
mkdir -p $SCRATCH/hopfield_results

# Set environment variables
export HOPFIELD_ENV=hpc
export REPO_ROOT=$(pwd)
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
export HOPFIELD_SCRATCH=$SCRATCH/hopfield_cache
```

### Verification

```bash
# Check import
python -c "import hopfield_llm; print('✓ Import successful')"

# Check environment auto-detection
python -c "
from hopfield_llm.utils.env import load_env_config
cfg = load_env_config()
print(f'Environment: {cfg.environment}')
print(f'Default batch size: {cfg.default_batch_size}')
print(f'Cache dir: {cfg.cache_dir}')
"

# Check GPU access (allocate a GPU node interactively)
salloc -p gpu -N 1 -c 4 --gres=gpu:1 --mem=32G --time=00:15:00
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
exit

# Submit a batch job
sbatch scripts/hpc/submit_pipeline.sh
```

### Environment Detection

The framework auto-detects local vs HPC via:
1. `HOPFIELD_ENV` env var (override)
2. Presence of `SLURM_JOB_ID` (HPC detection)
3. Available VRAM (local: ≤8GB → small models; HPC: ≥20GB → large models)

See `src/hopfield_llm/utils/env.py` for configuration details.
