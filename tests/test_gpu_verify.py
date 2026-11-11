"""verify and the smoke diagnostics on CUDA (skipped without a device).

On CPU every tensor lives on one device, so a CPU-saved tensor meeting a live
GPU tensor in one op (verify job 33013: rel_err(x_cache on CPU, x_tf on
cuda:0)) cannot be caught there. These tests run the full verify / smoke
paths, and fp32_reference with a separate float32 instance, with the tiny
models on CUDA; every check must run and pass (float32 tiny models)."""

import copy
import dataclasses

import numpy as np
import pytest
import torch

from energy_llm.pipeline.calibrate import run_calibrate
from energy_llm.storage import read_items
from tests.conftest import QUESTION_WORDS, WARMUP_WORDS, make_bundle, make_cfg, make_questions, write_jsonl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

CHECKS = {"reference", "gate", "tf_vs_cache", "alignment", "hidden", "files_vs_memory", "permutation",
          "limits", "monotone", "self_divergence", "determinism", "calibration", "fp32_reference"}


@pytest.fixture()
def gpu_run(tmp_path, pair):
    model, tok = pair
    arch = "llama" if "date_string" in tok.chat_template else "qwen2"
    bundle = make_bundle(model.to("cuda"), tok)
    assert bundle.device.type == "cuda"
    cfg = make_cfg(tmp_path / "run", arch)
    warm = write_jsonl(tmp_path / "warm.jsonl", make_questions(4, WARMUP_WORDS, 1), "warm")
    assert run_calibrate(cfg, bundle, read_items(warm), warm) is not None
    ev = write_jsonl(tmp_path / "eval.jsonl", make_questions(4, QUESTION_WORDS, 2), "q")
    return cfg, bundle, read_items(ev)


def test_rel_err_and_propagate_accept_mixed_devices():
    from energy_llm.core.bank import Bank
    from energy_llm.verification.verify import propagate, rel_err

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
    ref = rel_err(a, b)
    assert ref.dtype == np.float64
    np.testing.assert_array_equal(rel_err(a.cuda(), b), ref)
    np.testing.assert_array_equal(rel_err(a, b.cuda().bfloat16().float()), rel_err(a, b.bfloat16().float()))
    w = torch.randn(16, 8, generator=g)
    bank = Bank(gate_weights=[w.cuda()], gamma=[torch.ones(8, device="cuda")],
                key_norms=[w.norm(dim=1).cuda()], row_norms=[w.norm(dim=1).cuda()], d_model=8)
    out = propagate(bank, 1e-6, 0, a, b.cuda(), {"cos_gamma_query": 3.0, "cos_gamma_key": 2.0})
    for d in out.values():
        assert set(d) == {"norm_entropy", "energy", "js"} and all(v.dtype == np.float64 for v in d.values())


def test_verify_runs_every_check_on_cuda(gpu_run):
    from energy_llm.verification.verify import run_verify

    cfg, bundle, items = gpu_run
    r = run_verify(cfg, bundle, items, n=2)
    assert r["model"]["device"].startswith("cuda")
    assert {c["name"] for c in r["checks"]} == CHECKS
    # float32 tiny models, TF32 off: the CPU criteria hold on the device too
    assert {c["name"]: c["status"] for c in r["checks"]} == {n: "PASS" for n in CHECKS} | {"fp32_reference": "INFO"}


def test_fp32_reference_separate_instance_on_cuda(gpu_run, monkeypatch):
    """bf16-like bundle on CUDA: the float32 instance is a separate model on
    the device, the saved residuals are on CPU."""
    from energy_llm.core.bank import build_bank
    from energy_llm.verification import verify as v

    cfg, bundle, _ = gpu_run
    monkeypatch.setattr(v, "load_fp32_model", lambda cfg_, device: copy.deepcopy(bundle.model).to(device))
    fake = dataclasses.replace(bundle, dtype="bfloat16")
    bank = build_bank(bundle.model, bundle.dims)
    L, D = bundle.dims.n_layers, bundle.dims.d_model
    g = torch.Generator().manual_seed(0)
    x = {l: torch.randn(8, D, generator=g) for l in range(L)}
    saved = [{"ids": list(range(1, 9)), "x_tf": x, "x_cache": {l: t[:7] for l, t in x.items()},
              "roles": [0, 1, 2, 2, 0, 3, 3, 3],
              "refs": {"question_last": 3, "prompt_last": 4, "response_first": 5}}]
    signal = {var: {m: [[np.array([0.0, 1.0])] for _ in range(L)] for m in v.PROPAGATED}
              for var in ("cos_gamma_query", "cos_gamma_key")}
    out = v.fp32_reference(cfg, fake, bank, saved, {"cos_gamma_query": 5.0, "cos_gamma_key": 3.0}, signal)
    assert out["separate_fp32_instance"] is True and len(out["x_rel_tf_vs_fp32"]["analysis"]) == L


def test_smoke_diagnostics_on_cuda(gpu_run):
    from energy_llm.pipeline.smoke import format_summary, run_smoke

    cfg, bundle, items = gpu_run
    s = run_smoke(cfg, bundle, items, n=3, name="smoke-gpu")
    assert s["n_completed"] == 3 and s["n_failed"] == 0
    assert s["peak_gpu_mem_gb"] is not None
    assert len(s["per_layer"]) == bundle.dims.n_layers
    assert {"ablation_betas", "beta_per_layer", "prompt_last_entropy", "bands_excluding_layer0"} <= set(s)
    assert "ablation dstar_zeroed" in format_summary(s)
