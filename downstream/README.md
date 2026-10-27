# downstream/ — NOT part of the extraction module

Everything here was moved out of `energy_llm/` during the extraction
restructure (2026-10) so that the module ends at the metrics. Nothing was
modified; nothing here is installed or collected by `pytest` (testpaths =
`tests/`).

| Path | What it is |
|---|---|
| `energy_llm_legacy/` | `labeling.py` (OpenAI judge), `probe.py` (logistic probe, locked splits, CV), `stats.py` (bootstrap, permutation scan), `features.py` (old phi table) |
| `scripts/` | `label_samples.py`, `label_human.py`, `run_probe.py`, `build_features.py`, `run_beta_sensitivity.py` |
| `prep/datasets_legacy.py` | TruthfulQA / TriviaQA loaders — use them to export `{id, question}` JSONL inputs |
| `notebooks/` | labelling, kappa, per-experiment analysis, calibration figures |
| `configs/` | E1–E3 configs of the previous pipeline |
| `hpc/` | `features.sbatch`, `beta_sensitivity.sbatch` |
| `tests/` | their tests |
| `legacy_docs/` | `methodology_conformance.md`, `run_manifest.md` (audits of the old E1–E3 runs) |

**Compatibility:** this code targets the *previous* artefact format
(`trajectories/{id}.npz/.json`, `banks.pt`) and imports modules
(`energy_llm.config`, `energy_llm.io_artifacts`, `energy_llm.metrics` as they
were) that the restructure replaced. To run it as-is, check out commit
`28f00ce`. To use it on the new outputs it must be adapted to read
`features.parquet` / `answers.parquet` / `hidden.h5` by `id`.

Dependencies it needed beyond the module: `scikit-learn`, `scipy`, `openai`,
`datasets`, `matplotlib`, `joblib` (see `requirements.txt` here, unpinned as before).
