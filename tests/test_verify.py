"""energy-llm verify / inspect end to end on the tiny models, including a
negative control: a wrong read point must make verify fail."""

import json

import numpy as np
import pytest

from energy_llm.pipeline.calibrate import run_calibrate
from energy_llm.verification.inspection import inspect_item
from energy_llm.storage import read_items
from energy_llm.verification.verify import format_report, run_verify
from tests.conftest import QUESTION_WORDS, WARMUP_WORDS, make_bundle, make_cfg, make_questions, write_jsonl


@pytest.fixture()
def run(tmp_path, pair):
    model, tok = pair
    arch = "llama" if "date_string" in tok.chat_template else "qwen2"
    bundle = make_bundle(model, tok)
    cfg = make_cfg(tmp_path / "run", arch)
    warm = write_jsonl(tmp_path / "warm.jsonl", make_questions(4, WARMUP_WORDS, 1), "warm")
    assert run_calibrate(cfg, bundle, read_items(warm), warm) is not None
    ev = write_jsonl(tmp_path / "eval.jsonl", make_questions(4, QUESTION_WORDS, 2), "q")
    return cfg, bundle, read_items(ev)


def test_verify_all_pass_on_tiny_models(run):
    cfg, bundle, items = run
    r = run_verify(cfg, bundle, items, n=3)
    status = {c["name"]: c["status"] for c in r["checks"]}
    assert set(status.values()) == {"PASS", "INFO"}, status
    assert status["fp32_reference"] == "INFO"
    assert set(status) == {"reference", "gate", "tf_vs_cache", "alignment", "hidden",
                           "files_vs_memory", "permutation", "limits", "monotone",
                           "self_divergence", "determinism", "calibration", "fp32_reference"}
    assert r["options"] == {"n": 3, "reference_n": 2, "reference_layers": [0, 1, 2],
                            "tf_max_noise_ratio": 0.1, "fp32_reference": True}
    checks = {c["name"]: c for c in r["checks"]}
    tf = checks["tf_vs_cache"]["detail"]
    L = bundle.dims.n_layers
    assert len(tf["per_layer"]) == L and len(tf["x_rel_informative"]["per_layer"]) == L
    assert set(tf["x_rel_informative"]["max_at"]) == {"id", "layer", "position", "role"}
    for row in tf["per_layer"]:
        assert set(row) == {"layer", "norm_entropy", "energy", "js"}
        assert row["norm_entropy"]["sigma"] > 0
    fp = checks["fp32_reference"]["detail"]
    assert fp["separate_fp32_instance"] is False          # tiny models are float32
    assert fp["precision"]["cuda_matmul_allow_tf32"] is False
    # a float32 model is its own reference: TF residuals identical
    assert max(d["max"] for d in fp["x_rel_tf_vs_fp32"]) == 0.0
    saved = json.loads(open(f"{r['extraction_dir']}/verify_report.json").read())
    assert saved["passed"] is True
    assert "ALL PASS" in format_report(r)


def test_verify_options_skip_fp32_and_threshold(run):
    cfg, bundle, items = run
    r = run_verify(cfg, bundle, items, n=1, tf_max_noise_ratio=0.0, fp32_check=False)
    checks = {c["name"]: c for c in r["checks"]}
    assert checks["fp32_reference"]["status"] == "SKIP"
    assert r["options"]["fp32_reference"] is False and r["options"]["tf_max_noise_ratio"] == 0.0
    # threshold 0 passes only if the cached and TF traces agree exactly
    tf = checks["tf_vs_cache"]
    assert tf["status"] == ("PASS" if tf["max_error"] == 0.0 else "FAIL")


def test_noise_to_signal_ratios():
    from energy_llm.verification.verify import noise_to_signal

    noise = {"m": [[np.full(100, 0.01)], [np.zeros(5)]]}
    signal = {"m": [[np.array([0.0, 1.0]), np.array([0.0, 1.0])], [np.array([2.0, 2.0])]]}
    out = noise_to_signal(noise, signal)
    assert out["per_layer"][0]["m"]["sigma"] == pytest.approx(0.5)
    assert out["per_layer"][0]["m"]["ratio_p99"] == pytest.approx(0.02)
    assert out["per_layer"][1]["m"]["ratio_p99"] == 0.0            # zero noise, zero sigma
    assert out["max_ratio_p99"] == pytest.approx(0.02) and out["worst_p99_at"] == {"layer": 0, "metric": "m"}


def test_fp32_reference_loads_a_separate_instance_and_restores_precision(run, monkeypatch):
    """bf16-like bundle: the float32 model comes from load_fp32_model (never an
    in-place cast) and the float32 matmul settings are restored afterwards."""
    import copy
    import dataclasses

    import torch

    from energy_llm.core.bank import build_bank
    from energy_llm.verification import verify as v

    cfg, bundle, items = run
    loaded = []

    def fake_load(cfg_, device):
        m = copy.deepcopy(bundle.model)
        loaded.append(m)
        return m

    monkeypatch.setattr(v, "load_fp32_model", fake_load)
    fake = dataclasses.replace(bundle, dtype="bfloat16")
    bank = build_bank(bundle.model, bundle.dims)
    L, D = bundle.dims.n_layers, bundle.dims.d_model
    ids = list(range(1, 9))
    g = torch.Generator().manual_seed(0)
    x = {l: torch.randn(8, D, generator=g) for l in range(L)}
    saved = [{"ids": ids, "x_tf": x, "x_cache": {l: v_[:7] for l, v_ in x.items()}}]
    signal = {m: [[np.array([0.0, 1.0])] for _ in range(L)] for m in v.PROPAGATED}
    torch.set_float32_matmul_precision("high")
    try:
        out = v.fp32_reference(cfg, fake, bank, saved, 5.0, signal)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision("highest")
    assert len(loaded) == 1 and out["separate_fp32_instance"] is True
    assert len(out["x_rel_tf_vs_fp32"]) == L and len(out["x_rel_cache_vs_fp32"]) == L
    assert set(out["traces_cache_vs_fp32"]["per_layer"][0]) == {"layer", *v.PROPAGATED}


def test_verify_fails_when_gamma_is_ignored(run, monkeypatch):
    from energy_llm.core import bank as bank_mod

    cfg, bundle, items = run

    def no_gamma(self, l, x_hat):
        w = self.gate_weights[l].float()
        return (x_hat @ w.T) / (w.norm(dim=1) * self.d_model ** 0.5)

    monkeypatch.setattr(bank_mod.Bank, "scores", no_gamma)
    r = run_verify(cfg, bundle, items, n=1)
    status = {c["name"]: c["status"] for c in r["checks"]}
    assert not r["passed"]
    assert status["gate"] == "FAIL" and status["reference"] == "FAIL"


def test_cli_verify_exit_code(run, tmp_path, monkeypatch):
    import yaml

    from energy_llm import cli

    cfg, bundle, items = run
    monkeypatch.setattr(cli, "_load", lambda args: (cfg, items, bundle, 0.0))
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    assert cli.main(["verify", "--config", str(p), "--input", "unused", "--n", "1"]) == 0


def test_inspect_shows_positions_and_phases(run):
    cfg, bundle, items = run
    r = run_verify(cfg, bundle, items, n=2)
    text = inspect_item(r["extraction_dir"], items[0].id, tokenizer=bundle.tokenizer)
    lines = text.splitlines()
    assert lines[0].startswith(f"id {items[0].id}")
    header = lines.index(next(l for l in lines if l.split()[:2] == ["pos", "token_id"]))
    rows = lines[header + 1: lines.index("", header)]
    where = [l.split()[3] if len(l.split()) > 3 else "" for l in rows]
    for r in ("question_last", "prompt_last", "response_first"):
        assert where.count(r) == 1, r
    for name in ("energy", "norm_entropy", "margin", "var_score", "topk_mass"):
        start = lines.index(next(l for l in lines if l.startswith(f"{name} per layer")))
        assert lines[start + 1].split() == ["layer", "question_last", "prompt_last",
                                            "response_first", "response_mean"]
        assert len([l for l in lines[start + 2: start + 2 + bundle.dims.n_layers]
                    if l.split()[0].isdigit()]) == bundle.dims.n_layers
