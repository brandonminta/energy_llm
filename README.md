# energy_llm — extraction module

Extracts Modern-Hopfield retrieval metrics from the SwiGLU gate of
Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct, per question and per layer, for
a hallucination-detection study. The module **ends at the metrics**: no
labelling, judges, probes or statistics (those live in the analysis
repository; the legacy E1–E3 code is in git at tag `thesis-e1e3`). It is dataset-agnostic: the input is a JSONL of
`{"id", "question"}` records; every other field is ignored and never
propagated. All outputs are indexed by `id`.

Three documents describe it:

| document | read it for |
|---|---|
| this README | setup, how to run on SLURM, what the output files are, verification on the cluster |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | what the code **actually does**, module by module, file by file, formula by formula, with the verification tolerances — plus an audit of its inconsistencies and of where it departs from the thesis |
| [`docs/overview.md`](docs/overview.md) | why: the science the module serves |

## What is computed (short version)

Per layer, from the residual `x` at the input of `post_attention_layernorm`
(float32): `x̂ = x/√(mean(x²)+eps)`, bank = unit rows of `W̃ = W_g·γ`,
`s = bank·x̂/√D` (cosines). The bank is never materialised: only γ and `‖W̃_i‖`
are kept and `s = (W_g.float() @ (γ⊙x̂)) / (‖W̃‖·√D)` converts one layer's `W_g`
to float32 at a time. Then `p = softmax(β*s)`, `E = -(1/β*)·logsumexp(β*s)`,
`H̄ = H(p)/log D_ff`, margin = top1 − top2, stored **per token** for every
position (prompt and response) and every layer in `traces.h5`; the role mask
separates prompt and response (every stop token excluded from the
response). There is no features table: aggregation over tokens is analysis.

**Empty responses** (the model emits a stop token first) are valid samples,
not failures: `empty_response = true`, traces computed, response-only
values (baselines, `hidden.h5:response_last`) NaN by design, counted as
completed.

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.

## Layout

```
energy_llm/              the package (CLI: energy-llm / python -m energy_llm)
  cli.py                 calibrate | extract | merge | status | smoke | verify | inspect
  config.py              per-run YAML (model + revision, prompt, beta grid, shards, ...)
  storage.py             atomic writes, JSONL logs, failure log, hashing, input items
  manifest.py            provenance: git commit, library versions, hardware, seeds
  core/                  what is computed (never touches the output dir)
    model.py             loading at a pinned revision, bf16 check, clean greedy config
    prompting.py         chat template, single system instruction, role mask
    bank.py              gamma + row norms of W_g * gamma (bank never materialised), bank.h5
    metrics.py           float32 primitives (E, H_bar, margin, mean/var score, top-k, KL, JS)
    capture.py           greedy + teacher forcing + hooks + all metrics per question
  pipeline/              how it runs (everything under <output_dir>)
    signals.py           SIGUSR1/SIGTERM -> finish the current id, exit 75
    calibrate.py         beta* on a warm-up JSONL
    extract.py           one shard: identity, resume, per-id files, failure log
    merge.py             shard status; final files + manifest
    smoke.py             extract + merge on N questions in one process, summary
  verification/          never production outputs
    verify.py            checks vs the model and dev/reference.py
    inspection.py        position/phase table of one id
configs/                 qwen25_3b.yaml, llama32_3b.yaml (versioned, revisions pinned)
hpc/                     env.sh, prefetch_models.sh, submit.sh, *.sbatch (calibrate, extract_array,
                         merge, smoke, verify), missing_shards.sh
tests/                   CPU-only, tiny random Qwen2 + Llama models
dev/                     verification only, never for the paper: dev questions, warm-up,
                         naive float64 reference implementation (reference.py)
docs/                    ARCHITECTURE.md, overview.md, CHANGES_v0.2.md (change log)
```

## Setup

```bash
micromamba activate hopfield-llm          # cluster env name (hpc/env.sh: MAMBA_ENV); local dev: energy-lab
pip install -r requirements.txt && pip install -e ".[dev]"
pytest                                    # CPU only, ~45 s
```

`requirements.txt` is the single pinned dependency set. transformers ≥ 4.56
is required (older releases ignore `dtype=` and load float32). Loading fails
unless the parameters are bfloat16; the actual dtype is recorded in every
manifest.

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

Exit codes: 0 done; 75 work remains (signal or retryable failures: the
batch script requeues); anything else is a crash.

### Robustness (summary; details in ARCHITECTURE.md §7)

- **Shards**: contiguous blocks of `extraction.samples_per_shard` ids; one
  array task per shard, each writing only its own directory.
- **Atomic, idempotent**: each id's outputs go to one file (tmp + fsync +
  rename); the id is appended to `completed.jsonl` only afterwards.
- **Failures** are real errors only (exception, OOM, unexpected NaN/inf):
  `failed.jsonl` with the attempt number, retried by the next run up to
  `extraction.max_attempts`, never marked complete. Empty responses are not
  failures.
- **Signals**: `--signal=B:USR1@120` and SIGTERM are trapped; Python finishes
  the current id, closes its files and exits 75; the batch script then
  `scontrol requeue`s the task (up to `MAX_REQUEUES`). That is the only retry
  mechanism. Preemption is requeued by SLURM itself (`--requeue`).
- **Guards**: a shard refuses to resume under a different input, config or
  β*; `extract` refuses a β* calibrated under another model/prompt/generation
  setting, and any evaluation question that is in the warm-up set.
- **Offline compute nodes**: `HF_HUB_OFFLINE=1`; models pre-downloaded to
  `$SCRATCH_ROOT/hf_cache`.
- **Orphaned merge**: every submission is logged in
  `<extraction_dir>/submissions.log`; `missing_shards.sh … --resubmit`
  cancels the pending merge of earlier submissions, relaunches only the
  unfinished shards with no task still active, and submits a new merge with
  `afterok` on the relaunched array and on the still-active earlier tasks.
- **Logs**: `logs/extract_<array>_<task>.out` (appended across requeues).
  Peak GPU memory per shard run is in `runs.jsonl` and
  `manifest.json:shard_runs[].peak_gpu_mem_gb`.
- Batch size is 1 (padding changes bf16 results).

### Shard sizing

Target: ~2 h per shard including model load, inside the 3 h walltime. The
default `samples_per_shard: 1000` is an unmeasured, conservative estimate.
Run the smoke test first: its seconds/question and model load time give the
size directly (`(7200 − load) / s_per_question`). After the first real shard,
`hpc/missing_shards.sh …` prints the same suggestion from measured runs.
Changing it changes the shard plan, so use it for a **new** extraction name.

### MIG profile

`GRES` is a variable in `hpc/env.sh`. For the `a100_2g.10gb` trial:
`GRES=gpu:a100_2g.10gb:1 hpc/submit.sh smoke …` first (check the exact name
with `sinfo -o '%P %G'`) and read the peak-memory line. Beyond the bf16 model
and activations, only one layer's `W_g` in float32 (~90 MB for Qwen2.5-3B) is
resident at a time.

## Verification (before fixing datasets or parameters)

Goal: confirm on the real models that the extraction measures exactly what
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 describes, **before**
fixing datasets or parameters. The sequence on the cluster is below
("Verification on the cluster").

```bash
python -m energy_llm verify  --config C --input dev/dev_questions.jsonl [--n 4] \
                             [--reference-n 2] [--reference-layers 0,18,35]
python -m energy_llm inspect --run <extraction_dir> --id dev-known-01
hpc/submit.sh verify configs/qwen25_3b.yaml dev/dev_questions.jsonl 4     # 45 min walltime
```

`verify` runs the production extract + merge on the first N questions into
`extract/verify-<time>/` and prints PASS/FAIL per check with the maximum
observed error and its tolerance (exit 1 on any FAIL; `verify_report.json`).
The checks (`reference`, `gate`, `tf_vs_cache`, `alignment`, `hidden`,
`files_vs_memory`, `permutation`, `limits`, `monotone`, `self_divergence`,
`determinism`, `calibration`) and their tolerances are tabulated in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §9. `inspect` reads a
finished extraction (no model) and prints each position (token, role,
prompt/response phase) and, per layer, E, H̄ and margin at β\* for the last
prompt token, the response mean and the first response token.

### Verification on the cluster

> **The β\* obtained here from `dev/dev_warmup.jsonl` is for verification
> only.** It must not be used for any reported result. The definitive
> calibration is run once the datasets are fixed, on its own warm-up set and
> into a fresh `output_dir`. Nothing derived from `dev/` goes in the paper.

Run everything from the repo root. One-time setup for the session:

```bash
export SCRATCH_ROOT=/path/to/your/scratch      # required
source hpc/env.sh && activate_env
pip install -r requirements.txt && pip install -e ".[dev]"
hpc/prefetch_models.sh configs/qwen25_3b.yaml configs/llama32_3b.yaml   # login node; export HF_TOKEN first (Llama)
```

Verification runs write under `$SCRATCH_ROOT/energy_llm/runs/<model>/`. To
keep them apart from the definitive runs, point the configs at a separate
root for this session, e.g. `export SCRATCH_ROOT=$SCRATCH_ROOT/verification`
(prefetch first, or copy the HF cache, since `HF_HOME` follows `SCRATCH_ROOT`).


#### 1. Environment: CUDA available and compatible

```bash
srun --partition="$PARTITION_GPU" --gres="$GRES" --cpus-per-task=2 --mem=16G --time=00:10:00 \
  python -c "import torch, transformers; print(torch.__version__, 'cuda', torch.version.cuda, \
  'avail', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'tf', transformers.__version__); \
  a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print('bf16 matmul ok', (a@a).float().norm().item())"
srun --partition="$PARTITION_GPU" --gres="$GRES" --time=00:05:00 nvidia-smi
```

Check: `avail True`, the GPU is the expected A100, `transformers 5.0.0`, and
the driver's "CUDA Version" in `nvidia-smi` is >= `torch.version.cuda`
(12.1 for torch 2.5.1+cu121). If not, install a torch wheel matching the
driver before going on.

#### 2. Chat templates on the real tokenizers (both models)

```bash
QWEN=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-3B-Instruct', revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', local_files_only=True))")
LLAMA=$(python -c "from huggingface_hub import snapshot_download as s; print(s('meta-llama/Llama-3.2-3B-Instruct', revision='0cb88a4f764b7a12671c53f0838cd831a0843b95', local_files_only=True))")
ENERGY_LLM_TOKENIZER_DIRS="$QWEN:$LLAMA" pytest tests/test_real_tokenizer.py -v
```

Check: 2 passed (not skipped). It asserts the token ids equal the official
`apply_chat_template`, no default Qwen system prompt, and the role mask
isolates exactly the instruction and the question.

#### 3. Calibrate on the dev warm-up set (each model)

```bash
hpc/submit.sh calibrate configs/qwen25_3b.yaml dev/dev_warmup.jsonl
hpc/submit.sh calibrate configs/llama32_3b.yaml dev/dev_warmup.jsonl
cat $SCRATCH_ROOT/energy_llm/runs/qwen25_3b/calibration/beta_star.json | head -20
```

Check: the job ends without `not bracketed`; `failed_ids` is empty;
`mean_norm_entropy_curve` decreases from ~1 to ~0 across the grid and β\* is
not at a grid edge. (Verification-only β\*, see the note above.)

#### 4. Verify (each model): everything PASS

```bash
hpc/submit.sh verify configs/qwen25_3b.yaml dev/dev_questions.jsonl 4
hpc/submit.sh verify configs/llama32_3b.yaml dev/dev_questions.jsonl 4
cat logs/verify_<job>.out
```

Check: `=> ALL PASS` and exit code 0 (`sacct -j <job> -o State,ExitCode`).
The table lists each check with its max observed error and tolerance; the
full detail is in `<extraction_dir>/verify_report.json`. Read in particular:

| check | what it proves | what to look at |
|---|---|---|
| `gate` | read point + γ integration: the model's own gate pre-activation equals `‖w̃‖·√D·s` | error ≤ 4·u_bf16; `no_gamma_control_min_error` ≫ tolerance |
| `reference` | production == naive float64 implementation | `max error / tolerance` ≤ 1 (expect ≪ 1) |
| `tf_vs_cache` | teacher-forced x == x during cached generation | tolerance 5e-2 is **provisional** for bf16: note the observed max |
| `alignment` | last prompt position emits the first token; response = generated non-stop tokens | all booleans true per question |
| `hidden`, `files_vs_memory`, `determinism` | stored files hold exactly what was computed | 0 error on the stored arrays; GPU ≤ 1e-5 |
| `permutation`, `limits`, `monotone`, `self_divergence`, `calibration` | invariances and limits on real activations | all PASS; `calibration` inside the bracket |

If `tf_vs_cache` fails only on its bf16 tolerance while tokens are equal,
report the observed value before changing anything.

#### 5. Smoke test on the dev questions (each model)

```bash
hpc/submit.sh smoke configs/qwen25_3b.yaml dev/dev_questions.jsonl 24
hpc/submit.sh smoke configs/llama32_3b.yaml dev/dev_questions.jsonl 24
cat logs/smoke_<job>.out
```

Check:
- `model dtype bfloat16`;
- `TF == greedy 100.00 % of tokens, 100.0 % of answers` (anything less: report the ids from `smoke_report.json`);
- `max | ||x_hat||/sqrt(D) - 1 |` negligible (≲ 1e-3);
- mean H̄ per layer at β\*: not saturated (not ≈ 0 or ≈ 1 in most layers);
- peak GPU memory and seconds/question: note both (they size `samples_per_shard`: `(7200 − load) / s_per_question`);
- empty / truncated counts: plausible for short factual answers.

#### 6. Manual alignment review: 3 questions per model

```bash
RUN=$(ls -d $SCRATCH_ROOT/energy_llm/runs/qwen25_3b/extract/smoke-* | tail -1)
for ID in dev-known-01 dev-obscure-03 dev-misc-01; do
  python -m energy_llm inspect --run "$RUN" --id "$ID"; done
# same with runs/llama32_3b
```

Check by eye: `instruction` rows are exactly the system sentence, `question`
rows exactly the question, chat-template tokens are `template`; exactly one
`prompt` phase row, on the last prompt token (the assistant header); the
`response` rows are the answer tokens and the final `<|im_end|>` /
`<|eot_id|>` is `template`, not response; the per-layer table has one row
per layer with finite prompt values.

#### 7. Requeue test with a very short walltime

```bash
TIME_EXTRACT=00:06:00 SIGNAL_LEAD_S=240 hpc/submit.sh extract configs/qwen25_3b.yaml dev/dev_questions.jsonl requeue-test
squeue -u $USER                       # watch the task go RUNNING -> PENDING (requeued) -> RUNNING
grep -E "python exit|requeue|restart" logs/extract_<array>_0.out
python -m energy_llm status --config configs/qwen25_3b.yaml --input dev/dev_questions.jsonl --name requeue-test
```

USR1 arrives 4 min after start (6 min walltime − 240 s). Check in the log:
`python exit=75 signal=USR1`, `work left: scontrol requeue …`, a second run
with `restart=1` that skips the completed ids, then exit 0; the chained merge
job runs and `status` shows no missing shards. Confirm also that the merge
waited for the requeued task (`sacct -j <merge> -o Start` after the second run).

If the shard finishes before USR1 (fast GPU), send the warning by hand while
it is processing: `scancel --batch --signal=USR1 <array>_0`.

Caveat: if the walltime is shorter than model load + `SIGNAL_LEAD_S`, the
warning arrives before any question is processed, every run exits 75 without
progress and the task is requeued until `MAX_REQUEUES` (5), then left
unfinished — never use such a walltime for real runs.

Then try the MIG profile once (`GRES=gpu:a100_2g.10gb:1 hpc/submit.sh smoke …`)
if it is wanted, and record its peak memory and seconds/question.

## Outputs (`$SCRATCH_ROOT/energy_llm/runs/<model>/extract/<name>/`)

| File | Content |
|---|---|
| `manifest.json` | full config + hash, β* and calibration record, code commit, library versions, hardware and peak GPU memory per shard run, actual model dtype, checks (TF mismatches, ‖x̂‖ deviation, truncations, empty responses), exhausted ids |
| `answers.parquet` | id, prompt, response, prompt/generated token ids, lengths, `empty_response`, `truncated`, `stop_token_id`, per-token log-prob and entropy, TF mismatches |
| `traces.h5` | `/<id>/{energy, norm_entropy, margin}` `[L, T]` at β*, `/<id>/role_mask` `[T]` (0 template, 1 instruction, 2 question, 3 response) |
| `baselines.parquet` | mean / min token log-probability, mean token entropy (response tokens) |
| `hidden.h5` | `/<id>/{prompt_last, response_last}` `[layers, D]` float16 |
| `bank.h5` | γ and `‖W̃_i‖` per layer |

Schemas and the per-shard working files: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §7–8.

## History

- **v0.2, step 0**: the features table (the long per-layer parquet, variants
  `mean`/`first`, Hellinger², top-k Jaccard) removed — aggregation over
  tokens is analysis; the per-token traces replace it (extended in step 4); `downstream/` and the thesis `.tex` removed
  (tag `thesis-e1e3`); the spec-mapping document folded into
  `docs/ARCHITECTURE.md`, the cluster runbook into this README.
- **2026-10-27**: package split into `core/` (what is computed), `pipeline/`
  (how it runs) and `verification/`; duplicated helpers merged (failure log,
  CUDA release, item paths, shard-run reading); behaviour unchanged (same
  106 tests). `docs/ARCHITECTURE.md` added.
- **2026-10 restructure**: labelling, probe/CV/statistics, the old feature
  table and β-sensitivity scripts, the TruthfulQA/TriviaQA loaders, all
  notebooks, the E1–E3 configs and their jobs/tests moved to `downstream/`
  (intact, not installed, written for the previous artefact format);
  `scripts/*.py`, the old sbatch files, `requirements-hpc.txt` and
  `.env.example` removed; bank (γ folded in), hook location, metrics,
  prompting, calibration, runner and output format rewritten. Value/`w2`
  banks and 4-bit loading dropped. Pre-restructure state: commit `28f00ce`.
