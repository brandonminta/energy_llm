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
    variants = {"cos_gamma_query", "cos_gamma_key"}
    assert set(r["beta_star"]) == variants and r["primary"] == "cos_gamma_query"
    groups = {"analysis", "role/template", "role/instruction", "role/question", "role/response",
              "position/position_0", "position/question_last", "position/prompt_last",
              "position/response_first"}
    tf = checks["tf_vs_cache"]["detail"]
    L = bundle.dims.n_layers
    assert set(tf["per_variant"]) == variants and "analysis" in tf["pass_positions"]
    for pv in tf["per_variant"].values():
        assert set(pv["groups"]) == groups
        assert pv["max_ratio_p99"] == pv["groups"]["analysis"]["max_ratio_p99"]
        for row in pv["groups"]["analysis"]["per_layer"]:
            assert set(row) == {"layer", "norm_entropy", "energy", "js"}
            assert row["norm_entropy"]["sigma"] > 0 and "ratio_p99" in row["norm_entropy"]
    assert set(tf["x_rel_informative"]["groups"]) == groups
    assert all(len(x) == L for x in tf["x_rel_informative"]["groups"].values())
    assert set(tf["x_rel_informative"]["max_at"]) == {"id", "layer", "position", "role"}
    for name in ("gate", "limits", "monotone", "self_divergence"):
        assert set(checks[name]["detail"]["per_variant"]) == variants, name
    assert set(checks["calibration"]["detail"]["per_variant"]) == variants
    assert checks["gate"]["detail"]["reconstruction"]["cos_gamma_query"] == "||w_i|| ||gamma * x_hat|| c_i"
    pm = checks["permutation"]["detail"]
    assert pm["c"] == 2.0 and "tree" in pm["reduction_model"] and set(pm["per_variant"]) == variants
    for v, p in pm["per_variant"].items():
        assert set(p["worst"]) >= {"ratio", "trace", "layer", "position"}
        assert p["negative_control"]["worst"]["ratio"] > 1.0
        assert all(set(x) == {"max_error", "max_ratio", "at", "tolerance_at_max"} for x in p["per_trace"].values())
        assert "query_gain" in p["per_trace"]
    assert pm["per_variant"]["cos_gamma_query"]["topk_ids_min_row_jaccard"] == 1.0
    assert pm["per_variant"]["cos_gamma_key"]["topk_ids_min_row_jaccard"] is None   # not stored
    fp = checks["fp32_reference"]["detail"]
    assert fp["separate_fp32_instance"] is False          # tiny models are float32
    assert fp["precision"]["cuda_matmul_allow_tf32"] is False
    # a float32 model is its own reference: TF residuals identical, no recommendation
    assert max(d["max"] for d in fp["x_rel_tf_vs_fp32"]["analysis"]) == 0.0
    assert fp["precision_rule"]["value"] == 0.0 and fp["precision_rule"]["recommend"] is None
    assert r["recommend"] is None and "RECOMMEND" not in format_report(r)
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
    assert out["max_ratio_p99"] == pytest.approx(0.02)
    assert out["worst_p99_at"] == {"layer": 0, "metric": "m", "sigma": pytest.approx(0.5)}


def test_position_groups_and_analysis_criterion():
    """analysis = response + the three references; template and position 0 are
    separate groups; the criterion ignores noise outside the analysis set."""
    from energy_llm.verification.verify import ANALYSIS, grouped_noise, position_groups

    roles = [0, 1, 2, 2, 0, 3, 3, 0]          # template, instruction, question x2, template, response x2, stop
    refs = {"question_last": 3, "prompt_last": 4, "response_first": 5}
    g = position_groups(roles, refs, 7)       # last position not compared
    assert g[ANALYSIS].tolist() == [False, False, False, True, True, True, True]
    assert g["role/template"].tolist() == [True, False, False, False, True, False, False]
    assert g["position/position_0"].tolist() == [True] + [False] * 6
    assert g["position/response_first"].nonzero()[0].tolist() == [5]
    assert not position_groups(roles, refs | {"response_first": None}, 7)["position/response_first"].any()
    noise = {"m": [[np.array([5.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01])]]}   # a sink at position 0
    signal = {"m": [[np.array([0.0, 1.0])]]}
    out = grouped_noise(noise, signal, [g])
    assert out[ANALYSIS]["max_ratio_p99"] == pytest.approx(0.02)
    assert out["position/position_0"]["max_ratio_p99"] == pytest.approx(10.0)
    assert out["role/template"]["max_ratio_max"] == pytest.approx(10.0)


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
    roles = [0, 1, 2, 2, 0, 3, 3, 3]
    refs = {"question_last": 3, "prompt_last": 4, "response_first": 5}
    saved = [{"ids": ids, "x_tf": x, "x_cache": {l: v_[:7] for l, v_ in x.items()},
              "roles": roles, "refs": refs}]
    signal = {var: {m: [[np.array([0.0, 1.0])] for _ in range(L)] for m in v.PROPAGATED}
              for var in ("cos_gamma_query", "cos_gamma_key")}
    torch.set_float32_matmul_precision("high")
    try:
        out = v.fp32_reference(cfg, fake, bank, saved, {"cos_gamma_query": 5.0, "cos_gamma_key": 3.0}, signal)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision("highest")
    assert len(loaded) == 1 and out["separate_fp32_instance"] is True
    assert len(out["x_rel_tf_vs_fp32"]["analysis"]) == L and len(out["x_rel_cache_vs_fp32"]["analysis"]) == L
    for var in ("cos_gamma_query", "cos_gamma_key"):
        assert set(out["traces_cache_vs_fp32"][var]["analysis"]["per_layer"][0]) == {"layer", *v.PROPAGATED}
    # x_tf are random vectors, not the model's residuals: far from float32 -> recommendation
    rule = out["precision_rule"]
    assert rule["value"] > rule["threshold"] == v.RECOMMEND_FP32_RATIO == 0.1
    assert rule["recommend"] == "trace_dtype=float32" and rule["at"]["variant"] in signal


def test_verify_fails_when_gamma_is_ignored(run, monkeypatch):
    from energy_llm.core import bank as bank_mod

    cfg, bundle, items = run

    def no_gamma(self, l, x_hat):
        return x_hat @ self.gate_weights[l].float().T

    monkeypatch.setattr(bank_mod.Bank, "preactivation", no_gamma)
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
    rep = run_verify(cfg, bundle, items, n=2)
    text = inspect_item(rep["extraction_dir"], items[0].id, tokenizer=bundle.tokenizer)
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
    assert "variant cos_gamma_query" in lines[0]
    assert any(l.startswith("query_gain per layer") for l in lines)
    ctrl = inspect_item(rep["extraction_dir"], items[0].id, tokenizer=bundle.tokenizer, variant="cos_gamma_key")
    assert "variant cos_gamma_key" in ctrl.splitlines()[0] and ctrl != text
    with pytest.raises(KeyError, match="variant"):
        inspect_item(rep["extraction_dir"], items[0].id, tokenizer=bundle.tokenizer, variant="nope")
