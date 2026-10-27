"""Folding gamma into the bank reproduces the model's own computation:
gate pre-activation g_i == ||w_tilde_i|| * sqrt(D) * s_i."""

import math

import numpy as np
import pytest
import torch

from energy_llm import metrics
from energy_llm.bank import build_bank, read_bank_fingerprint, write_bank_h5
from energy_llm.model import model_dims


def _capture(model, ids):
    xs, gates = {}, {}
    handles = []
    for l, layer in enumerate(model.model.layers):
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, l=l: xs.__setitem__(l, a[0][0].clone())))
        handles.append(layer.mlp.gate_proj.register_forward_hook(
            lambda m, a, o, l=l: gates.__setitem__(l, o[0].clone())))
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    return xs, gates


def test_gamma_folded_bank_reproduces_gate_preactivation(pair):
    model, tok = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    ids = torch.randint(3, len(tok), (1, 12))
    xs, gates = _capture(model, ids)
    for l in range(dims.n_layers):
        x_hat = metrics.rms_normalise(xs[l], dims.rms_norm_eps)
        s = metrics.cosine_scores(x_hat, bank.rows[l])
        recon = bank.row_norms[l] * math.sqrt(dims.d_model) * s
        np.testing.assert_allclose(recon.numpy(), gates[l].numpy(), rtol=1e-4, atol=1e-5)
        assert s.abs().max() <= 1.0 + 1e-5


def test_bank_definition(pair):
    model, _ = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    for l, layer in enumerate(model.model.layers):
        w_g = layer.mlp.gate_proj.weight.float()
        gamma = layer.post_attention_layernorm.weight.float()
        assert not torch.allclose(gamma, torch.ones_like(gamma))
        assert bank.rows[l].shape == (dims.d_ff, dims.d_model)
        assert bank.rows[l].dtype == torch.float32
        np.testing.assert_allclose(bank.rows[l].norm(dim=1).numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(bank.row_norms[l].numpy(),
                                   (w_g * gamma).norm(dim=1).numpy(), rtol=1e-6)
        np.testing.assert_array_equal(bank.gamma[l].numpy(), gamma.numpy())


def test_dims_come_from_model_config(pair):
    model, _ = pair
    d = model_dims(model)
    c = model.config
    assert (d.n_layers, d.d_model, d.d_ff, d.rms_norm_eps) == (
        c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.rms_norm_eps)


def test_bank_h5_roundtrip(tmp_path, qwen_pair):
    model, _ = qwen_pair
    bank = build_bank(model, model_dims(model))
    write_bank_h5(tmp_path / "bank.h5", bank, {"model_id": "tiny"})
    assert read_bank_fingerprint(tmp_path / "bank.h5") == bank.fingerprint()
    import h5py
    with h5py.File(tmp_path / "bank.h5") as f:
        assert f["layer_0/gamma"].shape == (32,) and f["layer_0/row_norms"].shape == (48,)


def test_bf16_required_by_default(qwen_pair):
    from energy_llm.model import prepare_bundle

    model, tok = qwen_pair
    with pytest.raises(RuntimeError, match="bfloat16"):
        prepare_bundle(model, tok, 8)
    prepare_bundle(model.to(torch.bfloat16), tok, 8)
