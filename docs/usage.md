# Usage

## CLI Pipeline

The CLI exposes five stages that can be run independently:

```bash
# Stage 1: Extract MLP memory banks
hopfield-llm build-memory --model qwen25_3b --output bank.pt

# Stage 2: Extract hidden states from prompts
hopfield-llm extract-hidden --memory bank.pt --prompts prompts.json --output hidden.pt

# Stage 3: Compute divergence scores
hopfield-llm score --memory bank.pt --hidden hidden.pt --output scores.pt

# Stage 4: Aggregate statistics
hopfield-llm analyze --scores scores.pt --output analysis.json

# Stage 5: Generate plots
hopfield-llm visualize --analysis analysis.json --output plots/
```

## Python API

```python
from hopfield_llm.models import HFLLM
from hopfield_llm.extraction import extract_mlp_memory_bank, extract_segment_hidden_states
from hopfield_llm.metrics import compute_layer_divergences, hallucination_score
from hopfield_llm.extraction.hidden_states import align_banks_to_layers

# Load model
llm = HFLLM("qwen25_3b", load_in_4bit=True)

# Extract banks
banks = extract_mlp_memory_bank(llm, bank="gate")

# Extract hidden states
seg = extract_segment_hidden_states(
    llm,
    question="What is the capital of France?",
    answer="Paris is the capital of France.",
    pooling="mean",
)

# Compare factual vs hallucinated
seg_f = extract_segment_hidden_states(llm, question=q, answer=factual_answer)
seg_h = extract_segment_hidden_states(llm, question=q, answer=hallucinated_answer)

aligned = align_banks_to_layers(banks, seg_f.layer_indices)
div = compute_layer_divergences(seg_f.h_answer, seg_h.h_answer, aligned)
score = hallucination_score(div, metric="js")
```

## Batched Extraction

For GPU-rich environments (A100), use batched extraction for ~4-8x speedup:

```python
from hopfield_llm.extraction import extract_hidden_states_batch

samples = [(q1, a1), (q2, a2), (q3, a3), ...]
results = extract_hidden_states_batch(llm, samples, batch_size=8)
```

## Adding a New Dataset

Implement `BaseHallucinationDataset` and produce `DataSample` instances:

```python
from hopfield_llm.datasets.base import BaseHallucinationDataset, DataSample

class MyDataset(BaseHallucinationDataset):
    def __init__(self, path: str):
        self._samples = self._load(path)

    @property
    def name(self) -> str:
        return "my_dataset"

    def __iter__(self):
        return iter(self._samples)

    def __len__(self):
        return len(self._samples)

    def _load(self, path):
        # Load your data and produce DataSample instances
        samples = []
        for row in load_my_data(path):
            samples.append(DataSample(
                sample_id=f"my_{row['id']}_factual",
                question=row["question"],
                answer=row["correct_answer"],
                label=1,
                source_dataset="my_dataset",
                metadata={"category": row.get("category", "unknown")},
            ))
            samples.append(DataSample(
                sample_id=f"my_{row['id']}_hallucinated",
                question=row["question"],
                answer=row["wrong_answer"],
                label=0,
                source_dataset="my_dataset",
                metadata={"category": row.get("category", "unknown")},
            ))
        return samples
```

## Experiment Tracking

```python
from hopfield_llm.utils.tracking import ExperimentTracker
from pathlib import Path

tracker = ExperimentTracker(base_dir=Path("experiments/runs"))
run_dir = tracker.start(config={"model": "qwen25_3b", "dataset": "truthfulqa", ...})

# During experiment
tracker.log_metric("score_mean", 0.42, step=100)

# After experiment
tracker.finish(metrics_df=df)

# Later: load for replay
info = ExperimentTracker.load(run_dir)
print(info["config"])
```
