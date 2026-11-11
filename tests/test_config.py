import pytest
import yaml

from energy_llm.config import ConfigError, RunConfig, load_config
from tests.conftest import make_cfg


def test_round_trip_keeps_values_and_hash(tmp_path):
    cfg = make_cfg(tmp_path / "run")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    back = load_config(p)
    assert back == cfg and back.sha256() == cfg.sha256()


def test_betas_are_the_relative_grid_plus_beta_star(tmp_path):
    e = make_cfg(tmp_path).extraction
    assert e.beta_grid_log10_offset == {"min": -0.7, "max": 0.9, "num": 9}
    assert e.control_topk_ids_k == 0
    for b_star in (123.4, 7.0):
        betas = e.betas(b_star)
        assert len(betas) == 10 and betas[-1] == b_star
        assert betas[:-1] == pytest.approx([b_star * 10 ** (-0.7 + 0.2 * k) for k in range(9)])


def test_shipped_configs_use_the_relative_grid():
    from tests.conftest import REPO_ROOT

    for name in ("qwen25_3b.yaml", "llama32_3b.yaml"):
        raw = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
        assert raw["extraction"]["beta_grid_log10_offset"] == {"min": -0.7, "max": 0.9, "num": 9}
        assert raw["extraction"]["control_topk_ids_k"] == 0


@pytest.mark.parametrize("extraction, match", [
    ({"beta_grid_log10_offset": {"min": 0.0, "max": 4.0}}, "exactly the keys"),
    ({"beta_grid_log10_offset": {"min": 1.0, "max": 1.0, "num": 5}}, "empty"),
    ({"topk_ids_k": 0}, "topk"),
    ({"control_topk_ids_k": -1}, "control_topk_ids_k"),
])
def test_invalid_extraction_values(tmp_path, extraction, match):
    with pytest.raises(ConfigError, match=match):
        make_cfg(tmp_path, **extraction)


@pytest.mark.parametrize("old_key", ["beta_multipliers", "jaccard_top_k", "beta_grid_log10"])
def test_removed_keys_are_rejected(tmp_path, old_key):
    raw = make_cfg(tmp_path).to_dict()
    raw["extraction"][old_key] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        RunConfig.from_dict(raw)
