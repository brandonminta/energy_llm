# HPC Workflow

Complete workflow from setup to results download.

## 1. Initial Setup (one time)

```bash
# On the cluster login node
cd $HOME
git clone <your-repo-url> hopfield-llm
cd hopfield-llm
bash setup_hpc.sh
```

## 2. Activate Environment (each session)

```bash
micromamba activate hopfield-llm
source .env.hpc
```

## 3. Submit the Pipeline

### Default (Qwen-3B, TruthfulQA)

```bash
./jobs/submit_pipeline.sh
```

### Custom Model/Parameters

```bash
MODEL=phi3mini BETA=20.0 METRIC=kl_fwd ./jobs/submit_pipeline.sh
```

### Individual Stages

```bash
# Only bank extraction
sbatch jobs/templates/extract_banks.sh

# Only forward pass (bank must already exist)
BANK_PATH=$SCRATCH/hopfield_cache/banks/qwen25_3b__gate.pt sbatch jobs/templates/forward_pass.sh
```

## 4. Monitor Jobs

```bash
# Check queue
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Job efficiency after completion
seff <JOB_ID>

# View logs
tail -f logs/hidden_<JOB_ID>.out

# Job accounting
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
```

## 5. Cancel Jobs

```bash
# Cancel one job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel a pipeline (IDs from submit output)
scancel <JOB1> <JOB2> <JOB3>
```

## 6. Download Results

```bash
# From your local machine:
rsync -avz --progress \
    user@hpc-cluster:$SCRATCH/hopfield_results/ \
    ./data/results/

# Or specific experiment:
rsync -avz --progress \
    user@hpc-cluster:$SCRATCH/hopfield_results/qwen25_3b__beta15.0__js__* \
    ./data/results/
```

## 7. Troubleshooting

### Job fails immediately (State: FAILED)

Check the error log:
```bash
cat logs/hidden_<JOB_ID>.err
```

Common causes:
- **ModuleNotFoundError**: Environment not activated. Add `source .env.hpc` to the SLURM script.
- **CUDA out of memory**: Reduce batch size or use a larger GPU partition.
- **FileNotFoundError**: Upstream artifact missing. Run stages in order.

### Job killed (State: OUT_OF_MEMORY)

The job exceeded its RAM allocation:
```bash
sacct -j <JOB_ID> --format=MaxRSS
```
Increase `--mem` in the SLURM script.

### Job pending for a long time

```bash
squeue -u $USER --start  # shows estimated start time
sprio -u $USER           # shows job priority
```

If stuck, try:
- Smaller GPU partition (gpu-dev instead of gpu)
- Lower resource request (fewer CPUs/RAM)
- Off-peak hours

### GPU not detected

```bash
# Check allocation
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# If empty, the job may not have GPU access
# Verify --gres flag in SLURM script
```

### Cache corruption

If an artifact file is corrupted (e.g. job killed mid-write):
```bash
# Delete the corrupted file and rerun
rm $SCRATCH/hopfield_cache/hidden_states/qwen25_3b__mean.pt
# Resubmit the stage
sbatch jobs/templates/forward_pass.sh
```

## GPU Partitions Reference

| Partition | Max GPUs | Max Time | GPU Types | Best For |
|-----------|----------|----------|-----------|----------|
| `gpu-dev` | 1 | 96h | a100_1g.5gb | Bank extraction, debugging |
| `gpu` | 2 | 48h | a100_2g.10gb | Full experiments |
| `gpu-max` | 4 | 24h | a100-sxm4-40gb | Large sweeps |

### Allocation Examples

```bash
# Minimal (bank extraction)
salloc -p gpu-dev -n 1 -c 4 --mem=16G --gres=gpu:a100_1g.5gb:1 --time=02:00:00

# Standard (single-model experiment)
salloc -p gpu -n 1 -c 8 --mem=32G --gres=gpu:a100_2g.10gb:1 --time=24:00:00

# Large (Mistral-7B in fp16)
salloc -p gpu -n 1 -c 8 --mem=64G --gres=gpu:a100_3g.20gb:1 --time=24:00:00
```
