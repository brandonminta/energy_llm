# energy_llm — extraction module

Extracts Modern-Hopfield retrieval metrics from the SwiGLU gate of
Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct, per question and per layer, for
a hallucination-detection study. The module **ends at the metrics**: no
labelling, judges, probes or statistics (those live in `downstream/`, outside
the package). It is dataset-agnostic: the input is a JSONL of
`{"id", "question"}` records; every other field is ignored and never
propagated. All outputs are indexed by `id`.

The specification → code mapping, and the list of deliberate departures from
the thesis `.tex`, is in [`docs/MAPPING.md`](docs/MAPPING.md).

## What is computed

Per layer, from the residual `x` at the input of `post_attention_layernorm`
(float32): `x̂ = x/√(mean(x²)+eps)`, bank = unit rows of `W̃ = W_g·γ`,
`s = bank·x̂/√D`. The bank is never materialised: only γ and `‖W̃_i‖` are
kept, and `s = (W_g.float() @ (γ⊙x̂)) / (‖W̃‖·√D)` converts one layer's `W_g`
to float32 at a time, then frees it. Then `p = softmax(β*s)`, `E = -(1/β*)·logsumexp(β*s)`,
`H̄ = H(p)/log D_ff`, margin = top1 − top2. Phases: the last prompt token and
the response tokens (every stop token excluded). Features per layer, β ∈
{β*/2, β*, 2β*} and variant (`mean` over response positions, or `first`
response position): `energy_prompt, energy_resp, delta_energy,
norm_entropy_prompt, norm_entropy_resp, delta_norm_entropy, js,
hellinger_sq, jaccard_topk, margin_prompt, margin_resp`.

**Empty responses** (the model emits a stop token first) are valid samples,
not failures: greedy decoding would reproduce them on every retry, and
dropping them would bias the comparison between datasets. They are marked
`empty_response = true` in `answers.parquet` and `features.parquet`, counted
as completed, and never appear in `failed.jsonl`. Prompt-phase features are
computed normally; response-phase features (`energy_resp`, `delta_*`, `js`,
`hellinger_sq`, `jaccard_topk`, `margin_resp`), response baselines and
`hidden.h5:response_last` are NaN by design.

## Layout

```
energy_llm/            the package (CLI: energy-llm / python -m energy_llm)
  cli.py               calibrate | extract | merge | status | smoke
  config.py            per-run YAML (model + revision, prompt, beta grid, ...)
  model.py             loading at a pinned revision, bf16 check, clean greedy config
  bank.py              gamma + row norms of W_g * gamma (bank never materialised), bank.h5
  prompting.py         chat template, system instruction, role mask
  capture.py           greedy + teacher forcing + hooks + all metrics per question
  metrics.py           float32 primitives
  calibrate.py         beta* on a warm-up JSONL
  extract.py           sharded, resumable runner
  merge.py             final files + manifest; shard status
  smoke.py             extract + merge on N questions in one process, summary
  signals.py storage.py manifest.py
configs/               qwen25_3b.yaml, llama32_3b.yaml (versioned, revisions pinned)
hpc/                   env.sh, prefetch_models.sh, submit.sh, *.sbatch (calibrate, extract_array,
                       merge, smoke), missing_shards.sh
tests/                 CPU-only, tiny random Qwen2 + Llama models
docs/                  thesis .tex (unchanged) + MAPPING.md
downstream/            NOT part of the module: labelling, probes, stats, notebooks
```

## Setup

```bash
micromamba activate hopfield-llm          # cluster env name (hpc/env.sh: MAMBA_ENV)
pip install -r requirements.txt && pip install -e ".[dev]"
pytest                                    # CPU only, ~30 s
```

`requirements.txt` is the single pinned dependency set. The cluster env
previously had `transformers==4.45.2`, which ignores `dtype=` and loads
float32 — update it with the command above. Loading fails unless the
parameters are bfloat16; the actual dtype is recorded in every manifest.

Optional check against real tokenizers (Llama needs `HF_TOKEN`):
`ENERGY_LLM_TOKENIZER_DIRS=<qwen_dir>:<llama_dir> pytest tests/test_real_tokenizer.py`.

## Running on SLURM

Every command runs from the repo root. `SCRATCH_ROOT` is **required**
(no default); the HF cache and all outputs go under it.

```bash
export SCRATCH_ROOT=/path/to/your/scratch

# 0. once, on a login node with internet (Llama is gated: export HF_TOKEN first)
hpc/prefetch_models.sh configs/qwen25_3b.yaml configs/llama32_3b.yaml

# 1. calibrate beta* (one per model) on your warm-up JSONL
hpc/submit.sh calibrate configs/qwen25_3b.yaml data/warmup.jsonl
#    -> $SCRATCH_ROOT/energy_llm/runs/qwen25_3b/calibration/beta_star.json

# 2. smoke test (one process, 20 questions, 30 min walltime): check the summary
hpc/submit.sh smoke configs/qwen25_3b.yaml data/truthfulqa.jsonl 20
#    -> logs/smoke_<job>.out: dtype, % TF == greedy, max ||x_hat|| deviation,
#       peak GPU memory, s/question, mean H_bar per layer at beta*, empty / truncated

# 3. extract: array job over shards + merge chained with afterok
AFTER=<calibrate_job_id> hpc/submit.sh extract configs/qwen25_3b.yaml data/truthfulqa.jsonl truthfulqa
#    (AFTER is optional; omit it once beta_star.json exists)

# 4. progress / missing shards; relaunch only the unfinished ones (+ a new merge)
hpc/missing_shards.sh configs/qwen25_3b.yaml data/truthfulqa.jsonl truthfulqa
hpc/missing_shards.sh configs/qwen25_3b.yaml data/truthfulqa.jsonl truthfulqa --resubmit

# 5. merge by hand (the chained merge job runs the same command)
python -m energy_llm merge --config configs/qwen25_3b.yaml --input data/truthfulqa.jsonl --name truthfulqa
```

Resuming needs no special command: resubmitting (step 2 or 3) skips every id
already in a shard's completion index. Interactive equivalents:

```bash
python -m energy_llm calibrate --config C --input warmup.jsonl
python -m energy_llm extract   --config C --input eval.jsonl --name N --shard-index K
python -m energy_llm status    --config C --input eval.jsonl --name N [--missing|--num-shards|--json]
python -m energy_llm merge     --config C --input eval.jsonl --name N [--allow-incomplete]
python -m energy_llm smoke     --config C --input eval.jsonl [--n 20]
```

### Robustness

- **Shards**: contiguous blocks of `extraction.samples_per_shard` ids; one
  array task per shard, each writing only its own directory.
- **Atomic, idempotent**: each id's outputs go to one file (tmp + fsync +
  rename); the id is appended to `completed.jsonl` only afterwards.
- **Failures** are real errors only (exception, OOM, unexpected NaN/inf).
  They are never marked complete: they go to `failed.jsonl` with the attempt
  number and are retried by the next run, up to `extraction.max_attempts`.
  Empty responses are not failures (see above).
- **Signals**: `--signal=B:USR1@120` and SIGTERM are trapped; Python finishes
  the current id, closes its files and exits 75; the batch script then
  `scontrol requeue`s the task (up to `MAX_REQUEUES`). That is the only retry
  mechanism. Preemption is requeued by SLURM itself (`--requeue`).
- **Guards**: a shard refuses to resume under a different input, config or
  β*; `extract` refuses a β* calibrated under another model/prompt/generation
  setting, and any evaluation question that is in the warm-up set.
- **Offline compute nodes**: `HF_HUB_OFFLINE=1`; models pre-downloaded to
  `$SCRATCH_ROOT/hf_cache`.
- **Orphaned merge**: if a task runs out of requeues, its merge waits forever
  (`DependencyNeverSatisfied`). Every submission is logged in
  `<extraction_dir>/submissions.log`; `missing_shards.sh … --resubmit`
  cancels the pending merge of earlier submissions, relaunches only the
  unfinished shards with no task still active (never two writers on one
  shard), and submits a new merge with `afterok` on the relaunched array and
  on the still-active earlier tasks.
- **Logs**: `logs/extract_<array>_<task>.out` (appended across requeues).
- **Memory**: peak GPU memory per shard run is recorded in `runs.jsonl` and
  in `manifest.json:shard_runs[].peak_gpu_mem_gb`.
- Batch size is 1 (padding changes bf16 results).

### Shard sizing

Target: ~2 h per shard including model load, inside the 3 h walltime. The
default `samples_per_shard: 1000` is an unmeasured, conservative estimate.
Run the smoke test first: its seconds/question and model load time give the
size directly (`(7200 − load) / s_per_question`). After the first real shard,
`hpc/missing_shards.sh …` prints the same suggestion from measured runs. Changing it changes
the shard plan, so use it for a **new** extraction name.

### MIG profile

`GRES` is a variable in `hpc/env.sh`. For the `a100_2g.10gb` trial:
`GRES=gpu:a100_2g.10gb:1 hpc/submit.sh smoke …` first (check the exact name
with `sinfo -o '%P %G'`) and read the peak-memory line. The bank is never
materialised: beyond the bf16 model and activations, only one layer's `W_g`
in float32 (~90 MB for Qwen2.5-3B) is resident at a time.

## Outputs (`$SCRATCH_ROOT/energy_llm/runs/<model>/extract/<name>/`)

| File | Content |
|---|---|
| `manifest.json` | full config + hash, β* and calibration record, code commit, library versions, hardware and peak GPU memory per shard run, actual model dtype, checks (TF mismatches, ‖x̂‖ deviation, truncations, empty responses), exhausted ids |
| `answers.parquet` | id, prompt, response, prompt/generated token ids, lengths, `empty_response`, `truncated`, `stop_token_id`, per-token log-prob and entropy, TF mismatches |
| `features.parquet` | long format: `id, layer, beta, beta_multiplier, variant, empty_response` + 11 features |
| `traces.h5` | `/<id>/{energy, norm_entropy, margin}` `[L, T]` at β*, `/<id>/role_mask` `[T]` (0 template, 1 instruction, 2 question, 3 response) |
| `baselines.parquet` | mean / min token log-probability, mean token entropy (response tokens) |
| `hidden.h5` | `/<id>/{prompt_last, response_last}` `[layers, D]` float16 |
| `bank.h5` | γ and `‖W̃_i‖` per layer |

## What moved or was removed in the restructure

- **Moved to `downstream/`** (kept intact, not installed, written for the
  previous artefact format): labelling (`labeling.py`, `label_samples.py`,
  `label_human.py`), probe / CV / statistics (`probe.py`, `stats.py`,
  `run_probe.py`), the old feature table and β-sensitivity scripts, the
  TruthfulQA/TriviaQA loaders (`downstream/prep/`, to export the JSONL
  inputs), all notebooks, the E1–E3 configs, their SLURM jobs and tests, and
  the audit documents of the old runs.
- **Removed** (approved): `scripts/build_banks.py`, `calibrate_beta.py`,
  `capture_trajectories.py` (replaced by the CLI); `hpc/banks.sbatch`,
  `calibrate.sbatch`, `trajectories.sbatch`, `submit_pipeline.sh` (replaced;
  their resource values are preserved in `hpc/env.sh`; the bash retry loop
  is gone); `requirements-hpc.txt` (conflicting pins); `.env.example`
  (stale); `pytest_local.log`.
- **Rewritten**: bank (γ folded in), hook location, metrics, prompting,
  calibration, runner, output format. Value/`w2` banks and 4-bit loading
  were dropped (no experiment may use them; reported runs are bf16).
