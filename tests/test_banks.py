"""Bank orientation tests (subsec:stage1): on a tiny random SwiGLU model,
``m_hat @ r`` must equal the model's own gate pre-activations divided by the
stored row norms — i.e. the reconstruction g = s * ||m_i|| of
eq:retrieval_scores is exact."""

import numpy as np
import pytest
import torch

from energy_llm.banks import build_banks, extract_memory_bank, load_banks
from energy_llm.io_artifacts import save_torch_atomic
from energy_llm.models import resolve_decoder_layers, resolve_projection


class TestBankOrientation:
    def test_scores_reconstruct_gate_preactivations(self, tiny_model, tiny_banks):
        layers = resolve_decoder_layers(tiny_model)
        torch.manual_seed(1)
        for l, layer in enumerate(layers):
            r = torch.randn(tiny_model.config.hidden_size)
            g = layer.mlp.gate_proj(r)  # the model's own gate pre-activations
            m_hat = tiny_banks["m_hat"][l]
            norms = tiny_banks["row_norms"][l]
            s = m_hat @ r  # retrieval scores (eq:retrieval_scores)
            torch.testing.assert_close(s * norms, g, rtol=1e-5, atol=1e-5)

    def test_rows_are_unit_norm(self, tiny_banks):
        for l, m_hat in tiny_banks["m_hat"].items():
            np.testing.assert_allclose(
                torch.linalg.vector_norm(m_hat, dim=1).numpy(), 1.0, rtol=1e-5
            )

    def test_bank_shape_is_K_by_D(self, tiny_model, tiny_banks):
        K = tiny_model.config.intermediate_size
        D = tiny_model.config.hidden_size
        for l in range(tiny_model.config.num_hidden_layers):
            assert tuple(tiny_banks["m_hat"][l].shape) == (K, D)
            assert tuple(tiny_banks["row_norms"][l].shape) == (K,)

    def test_norms_match_raw_gate_weight(self, tiny_model, tiny_banks):
        layers = resolve_decoder_layers(tiny_model)
        for l, layer in enumerate(layers):
            raw = layer.mlp.gate_proj.weight.data.float()
            expected = torch.linalg.vector_norm(raw, dim=1)
            torch.testing.assert_close(tiny_banks["row_norms"][l], expected,
                                       rtol=1e-6, atol=1e-6)


class TestBankVariants:
    def test_value_bank_uses_up_proj(self, tiny_model):
        bank = extract_memory_bank(tiny_model, bank_id="value")
        layers = resolve_decoder_layers(tiny_model)
        raw = layers[0].mlp.up_proj.weight.data.float()
        norms = torch.linalg.vector_norm(raw, dim=1)
        torch.testing.assert_close(bank["m_hat"][0] * norms.unsqueeze(1), raw,
                                   rtol=1e-5, atol=1e-6)

    def test_w2_bank_is_transposed_to_K_by_D(self, tiny_model):
        bank = extract_memory_bank(tiny_model, bank_id="w2")
        K = tiny_model.config.intermediate_size
        D = tiny_model.config.hidden_size
        assert tuple(bank["m_hat"][0].shape) == (K, D)

    def test_raw_down_proj_rejected(self, tiny_model):
        for bad in ("down", "down_proj"):
            with pytest.raises(ValueError, match="rejected"):
                extract_memory_bank(tiny_model, bank_id=bad)

    def test_unknown_bank_rejected(self, tiny_model):
        with pytest.raises(ValueError, match="Unknown bank_id"):
            extract_memory_bank(tiny_model, bank_id="nonsense")


class TestBankSerialisation:
    def test_roundtrip(self, tiny_model, tmp_path):
        path = tmp_path / "banks.pt"
        artifact = build_banks(tiny_model, model_id="tiny-qwen2",
                               bank_id="gate", output_path=path)
        loaded = load_banks(path)
        assert loaded["bank_id"] == "gate"
        assert loaded["model_id"] == "tiny-qwen2"
        assert loaded["n_layers"] == tiny_model.config.num_hidden_layers
        for l in artifact["m_hat"]:
            torch.testing.assert_close(loaded["m_hat"][l], artifact["m_hat"][l])
            torch.testing.assert_close(loaded["row_norms"][l], artifact["row_norms"][l])

    def test_missing_keys_rejected(self, tmp_path):
        path = tmp_path / "bad.pt"
        save_torch_atomic(path, {"bank_id": "gate"})
        with pytest.raises(ValueError, match="missing keys"):
            load_banks(path)

    def test_banks_are_float32(self, tiny_banks):
        for l in tiny_banks["m_hat"]:
            assert tiny_banks["m_hat"][l].dtype == torch.float32
            assert tiny_banks["row_norms"][l].dtype == torch.float32
