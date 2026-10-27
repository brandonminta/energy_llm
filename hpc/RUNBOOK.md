# Verification runbook (cluster)

Goal: confirm on the real models that the extraction measures exactly what
`docs/MAPPING.md` specifies, **before** fixing datasets or parameters.

> **The β\* obtained here from `dev/dev_warmup.jsonl` is for verification
> only.** It must not be used for any reported result. The definitive
> calibration is run once the datasets are fixed, on its own warm-up set and
> into a fresh `output_dir`. Nothing derived from `dev/` goes in the paper.

Run everything from the repo root. One-time setup for this session:

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

---

## 1. Environment: CUDA available and compatible

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

## 2. Chat templates on the real tokenizers (both models)

```bash
QWEN=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-3B-Instruct', revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', local_files_only=True))")
LLAMA=$(python -c "from huggingface_hub import snapshot_download as s; print(s('meta-llama/Llama-3.2-3B-Instruct', revision='0cb88a4f764b7a12671c53f0838cd831a0843b95', local_files_only=True))")
ENERGY_LLM_TOKENIZER_DIRS="$QWEN:$LLAMA" pytest tests/test_real_tokenizer.py -v
```

Check: 2 passed (not skipped). It asserts the token ids equal the official
`apply_chat_template`, no default Qwen system prompt, and the role mask
isolates exactly the instruction and the question.

## 3. Calibrate on the dev warm-up set (each model)

```bash
hpc/submit.sh calibrate configs/qwen25_3b.yaml dev/dev_warmup.jsonl
hpc/submit.sh calibrate configs/llama32_3b.yaml dev/dev_warmup.jsonl
cat $SCRATCH_ROOT/energy_llm/runs/qwen25_3b/calibration/beta_star.json | head -20
```

Check: the job ends without `not bracketed`; `failed_ids` is empty;
`mean_norm_entropy_curve` decreases from ~1 to ~0 across the grid and β\* is
not at a grid edge. (Verification-only β\*, see the note above.)

## 4. Verify (each model): everything PASS

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

## 5. Smoke test on the dev questions (each model)

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

## 6. Manual alignment review: 3 questions per model

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
`<|eot_id|>` is `template`, not response; no `*` in the per-layer table
(`max |traces - features|` ≈ 0).

## 7. Requeue test with a very short walltime

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
