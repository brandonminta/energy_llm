# MAPPING.md — extraction specification → implementation

> **Source of truth.** Until the thesis chapters are revised, the
> **extraction specification** (restructure of 2026-10, summarised below)
> overrides `docs/fundamentals-FINAL.tex` and `docs/metodology-FINAL.tex`
> wherever they disagree. The `.tex` files are intentionally left untouched.
> The previous mapping (which followed the `.tex` exactly) is in git history
> at commit `28f00ce`.

## Deliberate departures from the `.tex` (approved 2026-10-26)

| Topic | `.tex` / previous code | Extraction specification (now implemented) |
|---|---|---|
| Read point | output of `post_attention_layernorm` (`r = γ⊙x̂`, pre-hook on `mlp`) | residual **x at the input** of `post_attention_layernorm`; `x̂ = x/√(mean(x²)+eps)` in float32 with the model's eps; γ folded into the bank |
| Scores | `s = m̂·r`, no `1/√D` | `s = (bank · x̂)/√D`, cosines in [-1, 1] |
| Energy | `lse + ½‖r‖²`, stored decomposed | `-(1/β)·logsumexp(βs)` only; `½‖x̂‖² ≈ D/2` is constant and omitted (deviation of ‖x̂‖ from √D recorded per id) |
| JS / Hellinger | mean over tokens of per-token divergence to the prompt | divergence between the **mean** response distribution and the prompt distribution |
| Response positions | every generated token, incl. the final (stop) token | generated tokens **excluding every stop token** (`generation_config.eos_token_id` list + tokenizer eos) |
| Prompt | checkpoint default (Qwen injects its system prompt; Llama injects today's date) | one explicit system instruction for every dataset; Llama `date_string` fixed |
| Decoding | `model.generate(do_sample=False)` — inherited Qwen's `repetition_penalty=1.05` | clean greedy config (`repetition_penalty=1.0`); teacher-forcing argmax checked against greedy |
| β grid | {1,…,100}, clamped if not bracketed | log grid 10⁰…10⁴ (cosine scale); calibration **fails** if not bracketed |
| Calibration positions | last prompt token only | last prompt token + response tokens (same exclusion as extract); warm-up JSONL disjoint from evaluation |
| Scope | banks → probe → statistics | ends at the metrics; labelling, probes, statistics in `downstream/` |
| Bank storage | float32 `m̂` per layer in `banks.pt` | never materialised: γ and `‖W̃_i‖` only; `s = (W_g.float() @ (γ⊙x̂)) / (‖W̃‖·√D)` one layer at a time |
| Empty responses | n/a | valid samples: `empty_response = true`, prompt features computed, response features NaN by design, marked completed |

## Specification → code

| Specification item | Module / function | Config key / artefact |
|---|---|---|
| Input JSONL `{id, question}`, other fields dropped | `storage.read_items` | `--input` |
| Model + exact revision, bf16 enforced, dims from checkpoint | `model.load_model`, `model.prepare_bundle`, `model.model_dims` | `model.id`, `model.revision`, `model.dtype` |
| Chat template, single system instruction, `date_string` | `prompting.build_prompt`, `prompting.check_template` | `prompt.system`, `prompt.date_string` |
| Role mask (template / instruction / question / response) | `prompting.build_prompt` (offset mapping), `capture.ForwardTrace.roles` | `traces.h5:/<id>/role_mask` |
| Greedy generation, per-token log-prob / entropy, truncation | `capture.forward_item`, `model.clean_generation_config` | `generation.max_new_tokens`; `answers.parquet` |
| Stop-token exclusion | `model.stop_token_ids`, `capture.forward_item` | `answers.parquet:stop_token_id, n_response_tokens` |
| Teacher forcing, argmax == greedy check | `capture.forward_item` | `answers.parquet:tf_mismatch_*`; `manifest.json:checks` |
| Hook at the input of `post_attention_layernorm` | `capture.ResidualHooks` | — |
| Bank `W̃ = W_g·γ`: only γ and row norms precomputed and saved | `bank.build_bank`, `bank.write_bank_h5` | `bank.h5` |
| Scores per layer, transient float32 `W_g` | `bank.Bank.scores`, `capture._layer_scores` | — |
| x̂, p, E, H̄, margin (float32) | `metrics.*` | — |
| Empty response as a valid sample | `capture._phase_features`, `capture.compute_item` (`RESPONSE_FEATURES`) | `answers/features.parquet:empty_response` |
| Peak GPU memory per shard | `manifest.peak_gpu_memory_gb`, `extract.run_extract` | `runs.jsonl`, `manifest.json:shard_runs` |
| Phases, variants mean/first, β*·{½,1,2} | `capture.compute_item`, `capture._phase_features` | `extraction.beta_multipliers`; `features.parquet` |
| JS, Hellinger², Jaccard top-k | `metrics.js_divergence`, `hellinger_sq`, `jaccard_topk` | `extraction.jaccard_top_k` |
| Per-token traces at β* | `capture.compute_item` | `traces.h5` |
| Residuals for the probe, float16 | `capture.compute_item` | `extraction.hidden_layers`; `hidden.h5` |
| Baselines | `capture.compute_item` | `baselines.parquet` |
| β* calibration, H̄* = 0.55 | `calibrate.run_calibrate`, `calibrate.interpolate_log_beta` | `calibration.*`; `calibration/beta_star.json` |
| extract never recalibrates; context + disjointness checks | `extract.load_beta_star` | — |
| Shards, completion index, failures, resume | `extract.run_extract`, `extract.shard_state` | `extraction.samples_per_shard`, `extraction.max_attempts` |
| SIGUSR1 / SIGTERM clean exit | `signals.GracefulStop`; `hpc/env.sh:run_requeueable` | exit code 75 |
| Merge + manifest | `merge.run_merge` | `manifest.json` and the files above |
| Missing-shard report | `merge.status_report`; `hpc/missing_shards.sh` | `energy-llm status` |
| Orphaned-merge recovery | `hpc/missing_shards.sh --resubmit`, `hpc/env.sh:record_submission` | `<extraction_dir>/submissions.log` |
| Smoke test | `smoke.run_smoke`; `hpc/smoke.sbatch` | `energy-llm smoke`; `smoke_report.json` |

## Implementation notes

1. **Response positions.** The state at absolute position `n_prompt + j` is
   the residual when generated token `j` is the input. Response positions are
   those `j` whose token is not a stop token; with greedy decoding only the
   last generated token can be a stop token.
2. **Hidden states.** The spec asks for "the residual of the last token" with
   shape `[layers, D]`. Both candidates are stored, each `[layers, D]`:
   `prompt_last` (last prompt token) and `response_last` (last non-stop
   response token). The residual is the same `x` the metrics read (input of
   `post_attention_layernorm`), not the layer output.
3. **Calibration weighting.** For each warm-up question H̄ is averaged over
   layers × calibration positions; questions are then weighted equally.
4. **Empty responses** (the first generated token is a stop token) are valid
   samples, not failures: greedy retries would reproduce them, and dropping
   them would bias the comparison between datasets. Prompt-phase features
   are computed; the response-phase features, the response baselines and
   `hidden.h5:response_last` are NaN by design. The NaN/inf check exempts
   exactly those fields; any other non-finite value is a real failure.
5. **Stop tokens in the role mask** are coded `template` (they are turn
   delimiters of the chat template); only non-stop generated tokens are
   `response`.
6. **‖x̂‖ check.** `‖x̂‖² = D·ms/(ms+eps)`; it departs from √D only when the
   residual RMS approaches √eps. The maximum relative deviation is stored per
   id (`answers.parquet:xhat_norm_max_rel_dev`) and summarised in the manifest.
