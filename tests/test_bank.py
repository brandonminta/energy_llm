"""Both score variants reproduce the model's own computation: the gate
pre-activation g_i == ||w_tilde_i|| sqrt(D) s_i (cos_gamma_key) ==
||w_i|| ||gamma * x_hat|| c_i (cos_gamma_query); the architecture adapter."""

import math

import numpy as np
import pytest
import torch

from energy_llm.core import metrics
from energy_llm.core.bank import (
    CONTROL, PRIMARY, build_bank, gamma_diagnostic, read_bank_fingerprint, write_bank_h5,
)
from energy_llm.core.model import effective_gain, ffn_norm, model_dims


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
        s = bank.scores(l, x_hat)
        recon = bank.row_norms[l] * math.sqrt(dims.d_model) * s
        np.testing.assert_allclose(recon.numpy(), gates[l].numpy(), rtol=1e-4, atol=1e-5)
        assert s.abs().max() <= 1.0 + 1e-5


def test_bank_definition_without_materialising_it(pair):
    model, _ = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    for l, layer in enumerate(model.model.layers):
        w_g = layer.mlp.gate_proj.weight.float()
        gamma = layer.post_attention_layernorm.weight.float()
        assert not torch.allclose(gamma, torch.ones_like(gamma))
        assert bank.gate_weights[l] is layer.mlp.gate_proj.weight   # no copy kept
        assert bank.row_norms[l].shape == (dims.d_ff,) and bank.row_norms[l].dtype == torch.float32
        np.testing.assert_allclose(bank.row_norms[l].numpy(),
                                   (w_g * gamma).norm(dim=1).numpy(), rtol=1e-6)
        np.testing.assert_array_equal(bank.gamma[l].numpy(), gamma.numpy())


def test_scores_match_precomputed_bank(pair):
    """On-the-fly scores == (unit rows of W_g*gamma) @ x_hat / sqrt(D), and are cosines."""
    model, _ = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    x_hat = metrics.rms_normalise(torch.randn(9, dims.d_model) * 3, dims.rms_norm_eps)
    for l, layer in enumerate(model.model.layers):
        w_tilde = layer.mlp.gate_proj.weight.float() * layer.post_attention_layernorm.weight.float()
        rows = w_tilde / w_tilde.norm(dim=1, keepdim=True)
        ref = x_hat @ rows.T / math.sqrt(dims.d_model)
        s = bank.scores(l, x_hat)
        assert s.dtype == torch.float32
        np.testing.assert_allclose(s.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)
        cos = torch.nn.functional.cosine_similarity(x_hat[:, None, :], rows[None], dim=-1)
        np.testing.assert_allclose(s.numpy(), cos.numpy(), atol=1e-3)  # ||x_hat|| ~ sqrt(D)
        assert s.abs().max() <= 1.0 + 1e-5


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
        np.testing.assert_array_equal(f["layer_0/key_norms"][()], bank.key_norms[0].numpy())
    # the fingerprint covers the key norms of the primary variant too
    bank.key_norms[0] = bank.key_norms[0] * 2
    assert bank.fingerprint() != read_bank_fingerprint(tmp_path / "bank.h5")


def test_gamma_diagnostic(pair):
    model, _ = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    d = gamma_diagnostic(bank, k=5)
    assert d["k"] == 5 and len(d["indices"]) == len(d["fraction"]) == dims.n_layers
    for l, layer in enumerate(model.model.layers):
        gamma = layer.post_attention_layernorm.weight.float()
        idx = d["indices"][l]
        assert idx == torch.topk(gamma.abs(), 5).indices.tolist()
        w_tilde = layer.mlp.gate_proj.weight.float() * gamma
        explicit = (w_tilde[:, idx].pow(2).sum() / w_tilde.pow(2).sum()).item()
        assert d["fraction"][l] == pytest.approx(explicit, rel=1e-5)
        assert 0.0 <= d["fraction"][l] <= 1.0
    # per-dimension fractions over all D dims sum to 1
    full = gamma_diagnostic(bank, k=dims.d_model)
    np.testing.assert_allclose(full["fraction"], 1.0, rtol=1e-5)
    assert gamma_diagnostic(bank, k=5) == d                          # deterministic


def test_bf16_required_by_default(qwen_pair):
    from energy_llm.core.model import prepare_bundle

    model, tok = qwen_pair
    with pytest.raises(RuntimeError, match="bfloat16"):
        prepare_bundle(model, tok, 8)
    prepare_bundle(model.to(torch.bfloat16), tok, 8)


def test_primary_scores_are_cosines_with_gamma_on_the_query(pair):
    """c_i = cos(w_i, gamma * x_hat): explicit cosine, reproduces the gate
    pre-activation as ||w_i|| ||gamma * x_hat|| c_i, and relates to the control
    as s_i = g_t (||w_i|| / ||w_tilde_i||) c_i with g_t = ||gamma * x_hat|| / sqrt(D)."""
    model, tok = pair
    dims = model_dims(model)
    bank = build_bank(model, dims)
    ids = torch.randint(3, len(tok), (1, 12))
    xs, gates = _capture(model, ids)
    D = dims.d_model
    for l, layer in enumerate(model.model.layers):
        x_hat = metrics.rms_normalise(xs[l], dims.rms_norm_eps)
        w = layer.mlp.gate_proj.weight.float()
        gamma = layer.post_attention_layernorm.weight.float()
        q = gamma * x_hat
        c = bank.scores(l, x_hat, PRIMARY)
        assert c.dtype == torch.float32 and c.abs().max() <= 1.0 + 1e-6
        cos = torch.nn.functional.cosine_similarity(q[:, None, :], w[None], dim=-1)
        np.testing.assert_allclose(c.numpy(), cos.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bank.key_norms[l].numpy(), w.norm(dim=1).numpy(), rtol=1e-6)
        g_t = bank.query_gain(l, x_hat)
        np.testing.assert_allclose(g_t.numpy(), (q.norm(dim=-1) / math.sqrt(D)).numpy(), rtol=1e-6)
        recon = bank.key_norms[l] * (g_t * math.sqrt(D))[:, None] * c
        np.testing.assert_allclose(recon.numpy(), gates[l].numpy(), rtol=1e-4, atol=1e-5)
        s = bank.scores(l, x_hat, CONTROL)
        np.testing.assert_allclose(s.numpy(), (g_t[:, None] * bank.key_norms[l] / bank.row_norms[l] * c).numpy(),
                                   rtol=1e-4, atol=1e-6)
        # one matmul serves both variants
        z = bank.preactivation(l, x_hat)
        for v in (PRIMARY, CONTROL):
            np.testing.assert_array_equal(bank.from_preactivation(l, z, x_hat, v).numpy(),
                                          bank.scores(l, x_hat, v).numpy())
    with pytest.raises(ValueError, match="unknown score variant"):
        bank.scores(0, x_hat, "cos_nothing")


def test_architecture_adapter(pair):
    """Qwen2 / Llama: the read point is post_attention_layernorm and the
    effective gain is its weight; Gemma is documented as not implemented;
    anything else is rejected."""
    import types

    model, _ = pair
    for layer in model.model.layers:
        assert ffn_norm(model, layer) is layer.post_attention_layernorm
        g = effective_gain(model, layer)
        assert g.dtype == torch.float32
        np.testing.assert_array_equal(g.numpy(), layer.post_attention_layernorm.weight.numpy())
    layer = model.model.layers[0]
    for mt, err, match in (("gemma2", NotImplementedError, "1 \\+ gamma"),
                           ("gemma", NotImplementedError, "pre_feedforward_layernorm"),
                           ("mistral", ValueError, "unsupported architecture")):
        fake = types.SimpleNamespace(config=types.SimpleNamespace(model_type=mt))
        with pytest.raises(err, match=match):
            effective_gain(fake, layer)
