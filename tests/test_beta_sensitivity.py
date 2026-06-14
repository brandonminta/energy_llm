"""Tests for scripts/run_beta_sensitivity.py — beta grid, CSV columns, and
no NaN values, exercised on 10 samples of the tiny random Qwen2 model.

The fixture builds the full pre-requisite chain:
  banks → calibrate → trajectories (cache_score_tensors=True) → labels
  → features.csv   ← required by the new single-pass implementation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from energy_llm.banks import build_banks, load_banks
from energy_llm.calibration import calibrate_beta
from energy_llm.config import ExperimentConfig, load_config
from energy_llm.features import build_feature_table
from energy_llm.labeling import label_samples, load_labels
from energy_llm.trajectory import run_trajectories
from scripts.run_beta_sensitivity import BETA_GRID, run_beta_sensitivity

from .conftest import make_tiny_model, make_tiny_samples, make_tiny_tokenizer
from .test_labeling import FakeJudge

N_SAMPLES = 10
EXPECTED_COLUMNS = {"beta", "sample_id", "mean_delta_hbar", "is_hallucination", "probe_auroc"}


@pytest.fixture(scope="module")
def beta_sens_data(tmp_path_factory):
    """Run the minimal pipeline and return (cfg, df) for assertion tests."""
    tmp = tmp_path_factory.mktemp("beta_sens")
    tokenizer = make_tiny_tokenizer()
    model = make_tiny_model(vocab_size=tokenizer.vocab_size + 4)
    samples = make_tiny_samples(N_SAMPLES)

    cfg = ExperimentConfig(name="beta_sens", output_dir=str(tmp / "run"))
    cfg.calibration.num_samples = 3
    cfg.generation.max_new_tokens = 4
    cfg.trajectory.cache_score_tensors = True

    banks = build_banks(
        model, model_id="tiny-qwen2", bank_id="gate", output_path=cfg.banks_path
    )
    banks = load_banks(cfg.banks_path)

    calibrate_beta(cfg, model, tokenizer, banks, samples)

    yaml_path = tmp / "beta_sens.yaml"
    yaml_path.write_text(f"name: beta_sens\noutput_dir: {cfg.output_dir}\n")
    cfg = load_config(yaml_path)  # snapshot wins, carries beta*

    run_trajectories(cfg, model, tokenizer, banks, samples)
    label_samples(cfg.trajectories_dir, cfg.labels_dir, FakeJudge())

    # Build features.csv — required by the single-pass implementation.
    labels = load_labels(cfg.labels_dir)
    df_feat = build_feature_table(cfg.trajectories_dir, labels=labels)
    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(cfg.features_dir / "features.csv", index=False)

    # Use n_jobs=1 to avoid subprocess overhead in tests.
    df = run_beta_sensitivity(cfg, n_jobs=1)
    return cfg, df


class TestBetaSensitivityCSV:
    def test_columns_present(self, beta_sens_data):
        _, df = beta_sens_data
        assert EXPECTED_COLUMNS.issubset(set(df.columns))

    def test_no_nans(self, beta_sens_data):
        _, df = beta_sens_data
        assert not df[sorted(EXPECTED_COLUMNS)].isna().any().any()

    def test_beta_grid_coverage(self, beta_sens_data):
        _, df = beta_sens_data
        assert set(df["beta"].unique()) == set(BETA_GRID)

    def test_row_count(self, beta_sens_data):
        _, df = beta_sens_data
        assert len(df) == N_SAMPLES * len(BETA_GRID)

    def test_mean_delta_hbar_finite(self, beta_sens_data):
        _, df = beta_sens_data
        assert np.isfinite(df["mean_delta_hbar"].to_numpy()).all()

    def test_probe_auroc_in_unit_interval(self, beta_sens_data):
        _, df = beta_sens_data
        aurocs = df["probe_auroc"].unique()
        assert ((aurocs >= 0.0) & (aurocs <= 1.0)).all()

    def test_mean_delta_hbar_varies_with_beta(self, beta_sens_data):
        """Different betas must produce different mean Delta H_bar values."""
        _, df = beta_sens_data
        per_beta = df.groupby("beta")["mean_delta_hbar"].mean()
        assert per_beta.nunique() > 1
