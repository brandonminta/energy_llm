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


def test_betas_are_the_absolute_grid_plus_beta_star(tmp_path):
    e = make_cfg(tmp_path).extraction
    assert e.beta_grid_log10 == {"min": 0.0, "max": 4.0, "num": 9}
    betas = e.betas(123.4)
    assert len(betas) == 10 and betas[-1] == 123.4
    assert betas[:-1] == pytest.approx([10 ** (k / 2) for k in range(9)])


@pytest.mark.parametrize("extraction, match", [
    ({"beta_grid_log10": {"min": 0.0, "max": 4.0}}, "exactly the keys"),
    ({"beta_grid_log10": {"min": 1.0, "max": 1.0, "num": 5}}, "empty"),
    ({"topk_ids_k": 0}, "topk"),
])
def test_invalid_extraction_values(tmp_path, extraction, match):
    with pytest.raises(ConfigError, match=match):
        make_cfg(tmp_path, **extraction)


@pytest.mark.parametrize("old_key", ["beta_multipliers", "jaccard_top_k"])
def test_removed_keys_are_rejected(tmp_path, old_key):
    raw = make_cfg(tmp_path).to_dict()
    raw["extraction"][old_key] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        RunConfig.from_dict(raw)
