# Usage

## CLI — pipeline stages

```bash
# Stage 1: extract MLP memory banks
hopfield-llm build-banks \
    --model qwen25_3b \
    --bank  down \           # gate | up | down | gate_plus_up | gate_up_concat
    --output banks.pt

# Stage 2: run prefill + generation for each sample
hopfield-llm run-trajectory \
    --model   qwen25_3b \
    --dataset truthfulqa \
    --banks   banks.pt \
    --output  traj/

# Stage 3: aggregate into analysis JSON (CPU-only)
hopfield-llm analyze \
    --trajectories traj/ \
    --output       analysis.json \
    --score-metric delta_energy   # or: js | kl_fwd | hellinger | ...

# Stage 4: generate plots (CPU-only)
hopfield-llm visualize \
    --analysis analysis.json \
    --output   plots/

# All stages from a YAML config
hopfield-llm run-experiment \
    --config configs/experiments/exp01_truthfulqa_baseline.yaml \
    --override beta=20.0 dataset_name=triviaqa
```

---

## Changing the memory bank

The `--bank` flag controls which MLP projection forms the memory:

```bash
# Default: W_down (the down-projection)
hopfield-llm build-banks --model qwen25_3b --bank down --output banks_down.pt

# Gate projection
hopfield-llm build-banks --model qwen25_3b --bank gate --output banks_gate.pt

# Composite: normalised average of gate and up projections
hopfield-llm build-banks --model qwen25_3b --bank gate_plus_up --output banks_gup.pt
```

Banks are stored as a dict `{layer_idx → Tensor[K, D]}`.  The rest of the
pipeline (hooks, metrics, analysis) is unaffected by this choice.

---

## Changing the query extractor

The query extractor selects which activation position becomes the retrieval
query.  Pass a callable `[T, d_m] → [d_m]` as `query_fn`:

```python
from hopfield_llm.hooks.capture import capture_prefill
from hopfield_llm.queries.extractors import mean_tokens, last_token, make_positional

# Default: last question token
result = capture_prefill(llm, question, banks, query_fn=last_token)

# Alternative: mean over all question tokens
result = capture_prefill(llm, question, banks, query_fn=mean_tokens)

# Alternative: a specific token index
result = capture_prefill(llm, question, banks, query_fn=make_positional(0))
```

---

## Python API

```python
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.memory.banks import extract_banks
from hopfield_llm.hooks.capture import capture_prefill, capture_generation
from hopfield_llm.metrics.energy import compute_energy
from hopfield_llm.pipeline.analysis import analyze_trajectories

# Load model
llm = HFLLM("qwen25_3b", load_in_4bit=True)

# Build memory banks
banks = extract_banks(llm, bank="down")

# Prefill pass
prefill = capture_prefill(llm, "Who wrote Hamlet?", banks, beta=15.0)
print(prefill.energy)   # [L] float32

# Generation pass
gen = capture_generation(llm, "Who wrote Hamlet?", banks, max_new_tokens=30)
print(gen.generated_text)
print(gen.energy)       # [L, T] float32

# Analyze saved trajectories
analyze_trajectories(
    traj_dir="outputs/exp01/trajectories",
    output="outputs/exp01/analysis.json",
    score_metric="delta_energy",
)
```

---

## Adding a new dataset

Subclass `BaseDataset` and yield `DataSample` objects:

```python
from hopfield_llm.datasets.base import BaseDataset, DataSample
from typing import Iterator

class MyDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "mydataset"

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)
```

Then register it in `pipeline/stages.py` inside `run_trajectory`.

---

## Experiment tracking

`ExperimentTracker` creates a timestamped run directory with:
```
outputs/{name}/{timestamp}_{model}_{dataset}_{bank}_{beta}_{hash}/
    config.yaml          — full config snapshot
    git_info.json        — commit, branch, dirty flag
    environment.yaml     — GPU name, VRAM, duration
    metrics_log.json     — stage completion timestamps
    trajectories/        — per-sample .npz + .json
    analysis.json
    plots/
```

To replay or inspect a previous run:
```python
from hopfield_llm.utils.tracking import ExperimentTracker
info = ExperimentTracker.load("outputs/exp01/...")
print(info["config"])
```

---

## Security: Hugging Face tokens

Models are downloaded from the HuggingFace Hub on first use.  If you need
a private model (e.g. Llama 3), provide your access token via the environment:

```bash
export HF_TOKEN=hf_...
```

Never hardcode tokens in configs or scripts.  The `.env` file (if you use one)
is gitignored by default.  See `.env.example` for the expected format.
