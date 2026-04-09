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
# Copy setup script to the cluster, then:
bash setup_hpc.sh
```

This will:
1. Load CUDA and Python modules
2. Create a micromamba/conda environment with `requirements-hpc.txt`
3. Install `hopfield_llm` in editable mode
4. Create cache and results directories on `$SCRATCH`
5. Write `.env.hpc` with all environment variables
6. Run a verification test

### Manual Setup

```bash
# On the cluster
module load cuda python
micromamba create -n hopfield-llm python=3.12 -y
micromamba activate hopfield-llm
pip install -r requirements-hpc.txt
pip install -e .

# Create directories
mkdir -p $SCRATCH/hopfield_cache/{banks,hidden_states,distributions}
mkdir -p $SCRATCH/hopfield_results

# Set environment
export HOPFIELD_ENV=hpc
export REPO_ROOT=$(pwd)
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
```

### Verification

```bash
# Check import
python -c "import hopfield_llm; print(hopfield_llm.__version__)"

# Check environment detection
python -c "
from hopfield_llm.utils.env import load_env_config
cfg = load_env_config()
print(f'Environment: {cfg.environment}, Batch: {cfg.default_batch_size}')
"

# Check GPU access (on a GPU node)
salloc -p gpu-dev -n 1 -c 2 --gres=gpu:a100_1g.5gb:1 --mem=8G --time=00:10:00
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
exit
```
