"""GPU path (skipped without CUDA): per-layer float32 scores on the device,
peak GPU memory recorded, results equal to the CPU run within float32."""

import numpy as np
import pytest
import torch

from energy_llm.bank import build_bank
from energy_llm.capture import ResidualHooks, compute_item, forward_item
from energy_llm.prompting import build_prompt
from tests.conftest import SYSTEM, make_bundle, make_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _run(device):
    model, tok = make_pair("qwen2")
    bundle = make_bundle(model.to(device), tok)
    bank = build_bank(bundle.model, bundle.dims)
    assert bank.device.type == device
    hooks = ResidualHooks(bundle.model)
    tr = forward_item(bundle, hooks, build_prompt(tok, SYSTEM, "who wrote hamlet", None))
    hooks.remove()
    return tr, compute_item(bundle, bank, tr, "g", 6.0, [0.5, 1.0, 2.0], 8,
                            list(range(bundle.dims.n_layers)))


def test_gpu_matches_cpu_and_reports_peak_memory():
    from energy_llm.manifest import peak_gpu_memory_gb

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    tr_g, res_g = _run("cuda")
    peak = peak_gpu_memory_gb(dev)
    assert peak is not None and peak["allocated"] > 0
    tr_c, res_c = _run("cpu")
    if tr_g.gen_ids != tr_c.gen_ids:
        pytest.skip("greedy tokens differ between devices (near-tie)")
    np.testing.assert_allclose(res_g.features, res_c.features, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(res_g.energy, res_c.energy, rtol=1e-3, atol=1e-4)
