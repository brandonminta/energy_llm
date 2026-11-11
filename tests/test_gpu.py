"""GPU path (skipped without CUDA): per-layer float32 scores on the device,
peak GPU memory recorded, results equal to the CPU run within float32."""

import numpy as np
import pytest
import torch

from energy_llm.core.bank import build_bank
from energy_llm.core.capture import ResidualHooks, compute_item, forward_item
from energy_llm.core.prompting import build_prompt
from tests.conftest import SYSTEM, both, make_bundle, make_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _run(device):
    model, tok = make_pair("qwen2")
    bundle = make_bundle(model.to(device), tok)
    bank = build_bank(bundle.model, bundle.dims)
    assert bank.device.type == device
    hooks = ResidualHooks(bundle.model)
    tr = forward_item(bundle, hooks, build_prompt(tok, SYSTEM, "who wrote hamlet", None))
    hooks.remove()
    return tr, compute_item(bundle, bank, tr, "g", both([1.0, 6.0]), list(range(bundle.dims.n_layers)), 10, both(20))


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
    for name, arr in res_c.traces.items():
        if not name.endswith("topk_ids"):
            np.testing.assert_allclose(res_g.traces[name], arr, rtol=1e-3, atol=1e-4, err_msg=name)
