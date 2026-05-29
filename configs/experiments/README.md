# Experiment configurations

Each YAML is consumed by `hopfield-llm run-experiment --config <file>` and is
snapshotted into the run directory for provenance. The schema (hierarchical
YAML → flat `ExperimentConfig`) is defined in
`src/hopfield_llm/pipeline/config.py`.

The headline (`final_*`) runs realise the definitive configuration of
`docs/methodology.tex` §experimental_config: **gated_key** bank (rows of
`W_gate ⊙ W_up`, L2-normalised), `energy_mode=dot`, `normalize_query=false`,
β calibrated to mean normalised entropy `H̄*=0.55`, greedy decoding (50 tokens),
`pool_over_answer=true`, seed 42. The `exp0*` runs are pre-registered ablations
and the completed historical baseline; the `diagnostic_*` run carries the
exploratory signals that are reported as ablations only.

| Config | Model | Dataset | Bank | normalize_query | Role / provenance | Status |
|---|---|---|---|---|---|---|
| `final_qwen_truthfulqa.yaml`  | Qwen2.5-3B | TruthfulQA | gated_key | false | **Headline result** — methodology §experimental_config | to run |
| `final_llama_truthfulqa.yaml` | Llama-3.2-3B | TruthfulQA | gated_key | false | Cross-architecture replication (H4 transfer) | to run |
| `final_qwen_triviaqa.yaml`    | Qwen2.5-3B | TriviaQA | gated_key | false | Cross-dataset transfer (step 10) | to run |
| `final_llama_triviaqa.yaml`   | Llama-3.2-3B | TriviaQA | gated_key | false | Cross-dataset + cross-arch transfer | to run |
| `exp04_qwen_chat_template_fix.yaml`     | Qwen2.5-3B | TruthfulQA | up | true | Completed reference run (AUROC≈0.65); see `docs/reproducibility.md` | **done** |
| `exp04b_qwen_normalize_query_false.yaml`| Qwen2.5-3B | TruthfulQA | up | false | Pre-registered H1/T2 — falsifies the E_quad tautology | pending |
| `exp05_llama_chat_template_fix.yaml`    | Llama-3.2-3B | TruthfulQA | up | true | Pre-registered replication (legacy bank) | pending |
| `exp06_qwen_value_space.yaml`           | Qwen2.5-3B | TruthfulQA | up + down_values | true | Pre-registered key-vs-value paired probe (Geva 2021) | pending |
| `exp07b_qwen_gated_key.yaml`            | Qwen2.5-3B | TruthfulQA | gated_key | false | Pre-registered H3 — gating vs raw W_up | pending |
| `diagnostic_logit_lens_null.yaml`       | Qwen2.5-3B | TruthfulQA | gated_key | false | Exploratory diagnostics (logit-lens + null control), reported as ablations | optional |

Pre-registered hypotheses (H1–H4) and the locked analysis plan live in
`docs/preregistration.md`. Completed-run hashes/versions live in
`docs/reproducibility.md`.

## Notes

- `bank:` under `trajectory:` is a legacy single-probe field; in probe mode
  (`primary_probe`/`secondary_probe` present) `primary_probe.bank` takes
  precedence. The final configs set both consistently for clarity.
- Override any field from the CLI: `--override beta=20.0 dataset_name=triviaqa`.
- For large HPC sweeps set `sharding.num_shards > 1` and submit with
  `scripts/hpc/submit_pipeline.sh`; run `hopfield-llm analyze` once after all
  shards land.
