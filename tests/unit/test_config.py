"""Tests for experiment config loading and overrides."""

from __future__ import annotations

import pytest

from hopfield_llm.pipeline.config import (
    ExperimentConfig,
    apply_overrides,
    load_experiment_config,
)


def test_load_default_config():
    cfg = load_experiment_config("configs/experiments/exp01_truthfulqa_baseline.yaml")
    assert cfg.name == "exp01_truthfulqa_baseline"
    assert cfg.model_alias == "qwen25_3b"
    assert cfg.dataset_name == "truthfulqa"
    assert cfg.beta == 15.0
    assert cfg.energy_mode == "dot"
    assert cfg.load_in_4bit is True
    assert cfg.seed == 42
    assert cfg.score_metric == "delta_energy"
    assert cfg.signal_zone is None


def test_config_to_dict_round_trip():
    cfg = ExperimentConfig(name="test", model_alias="phi3_mini")
    d = cfg.to_dict()
    assert d["experiment"]["name"] == "test"
    assert d["model"]["alias"] == "phi3_mini"
    assert isinstance(d["analysis"]["divergence_metrics"], list)
    # Verify trajectory section includes bank (for tracker slug)
    assert "bank" in d["trajectory"]


def test_apply_overrides():
    cfg = ExperimentConfig()
    apply_overrides(cfg, ["beta=20.0", "seed=0", "dataset_name=triviaqa"])
    assert cfg.beta == 20.0
    assert cfg.seed == 0
    assert cfg.dataset_name == "triviaqa"


def test_apply_override_bool():
    cfg = ExperimentConfig(load_in_4bit=True)
    apply_overrides(cfg, ["load_in_4bit=false"])
    assert cfg.load_in_4bit is False


def test_apply_override_list():
    cfg = ExperimentConfig()
    apply_overrides(cfg, ["divergence_metrics=kl_fwd,hellinger"])
    assert cfg.divergence_metrics == ["kl_fwd", "hellinger"]


def test_apply_override_unknown_key():
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_overrides(ExperimentConfig(), ["nonexistent_key=value"])


def test_apply_override_bad_format():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(ExperimentConfig(), ["no_equals_sign"])


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_experiment_config("nonexistent.yaml")


def test_load_config_with_signal_zone(tmp_dir):
    config_path = tmp_dir / "test.yaml"
    config_path.write_text("experiment:\n  name: test\nanalysis:\n  signal_zone: [5, 20]\n")
    cfg = load_experiment_config(config_path)
    assert cfg.signal_zone == (5, 20)
