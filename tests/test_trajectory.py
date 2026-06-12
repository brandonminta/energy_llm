"""Hook placement and trajectory capture tests (subsec:stage2).

The pre-hook on the mlp module must observe the probe point r
(eq:probe_point) — the output of the post-attention RMSNorm — and NOT the
block output."""

import numpy as np
import pytest
import torch

from energy_llm.trajectory import (
    ProbePointHooks,
    build_prompt_ids,
    capture_prefill_reference_scores,
    capture_sample,
)


class TestHookPlacement:
    def test_pre_hook_sees_post_attention_rmsnorm_output(self, tiny_model, tiny_tokenizer):
        """The mlp pre-hook input equals layer.post_attention_layernorm
        output, and differs from the block output."""
        from energy_llm.models import resolve_decoder_layers, resolve_mlp

        layers = resolve_decoder_layers(tiny_model)
        captured_pre: dict[int, torch.Tensor] = {}
        captured_norm: dict[int, torch.Tensor] = {}
        captured_block_out: dict[int, torch.Tensor] = {}
        handles = []
        for l, layer in enumerate(layers):
            handles.append(resolve_mlp(layer).register_forward_pre_hook(
                lambda m, args, l=l: captured_pre.__setitem__(l, args[0].detach().clone())
            ))
            handles.append(layer.post_attention_layernorm.register_forward_hook(
                lambda m, args, out, l=l: captured_norm.__setitem__(l, out.detach().clone())
            ))
            handles.append(layer.register_forward_hook(
                lambda m, args, out, l=l: captured_block_out.__setitem__(
                    l, (out[0] if isinstance(out, tuple) else out).detach().clone())
            ))
        try:
            ids, mask = build_prompt_ids(tiny_tokenizer, "what is the color of the sky",
                                         next(tiny_model.parameters()).device)
            with torch.inference_mode():
                tiny_model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        for l in range(len(layers)):
            torch.testing.assert_close(captured_pre[l], captured_norm[l])
            assert not torch.allclose(captured_pre[l], captured_block_out[l])

    def test_hook_scores_match_manual_bank_product(self, tiny_model, tiny_tokenizer, tiny_banks):
        hooks = ProbePointHooks(tiny_model, tiny_banks)
        try:
            ids, mask = build_prompt_ids(tiny_tokenizer, "who wrote hamlet",
                                         next(tiny_model.parameters()).device)
            # capture every position to cross-check against a manual product
            from energy_llm.models import resolve_decoder_layers, resolve_mlp
            layers = resolve_decoder_layers(tiny_model)
            manual: dict[int, torch.Tensor] = {}
            handles = [
                resolve_mlp(layer).register_forward_pre_hook(
                    lambda m, args, l=l: manual.__setitem__(l, args[0][0].detach().float())
                )
                for l, layer in enumerate(layers)
            ]
            try:
                hooks.capture(position_offset=0)
                with torch.inference_mode():
                    tiny_model(input_ids=ids, attention_mask=mask, use_cache=False)
                hooks.stop()
            finally:
                for h in handles:
                    h.remove()
            for l in range(hooks.n_layers):
                expected = manual[l] @ tiny_banks["m_hat"][l].T
                torch.testing.assert_close(hooks.scores[l], expected, rtol=1e-4, atol=1e-5)
        finally:
            hooks.remove()


class TestCaptureSample:
    @pytest.fixture()
    def traj(self, tiny_model, tiny_tokenizer, tiny_banks, tiny_samples):
        hooks = ProbePointHooks(tiny_model, tiny_banks)
        try:
            yield capture_sample(
                tiny_model, tiny_tokenizer, hooks, tiny_samples[0],
                beta_star=5.0, max_new_tokens=6, cache_score_tensors=True,
            )
        finally:
            hooks.remove()

    def test_array_shapes(self, traj, tiny_model):
        L = tiny_model.config.num_hidden_layers
        t_g = traj.metadata["n_tokens_generated"]
        assert 1 <= t_g <= 6
        for key in ("ref_lse_term", "ref_quadratic_term", "ref_entropy", "ref_norm_entropy"):
            assert traj.arrays[key].shape == (L,)
        for key in ("gen_lse_term", "gen_quadratic_term", "gen_entropy",
                    "gen_norm_entropy", "gen_js", "gen_hellinger_sq"):
            assert traj.arrays[key].shape == (L, t_g)
        assert traj.arrays["token_logprobs"].shape == (t_g,)
        assert traj.arrays["token_predictive_entropy"].shape == (t_g,)

    def test_divergences_bounded(self, traj):
        assert np.all(traj.arrays["gen_js"] >= 0)
        assert np.all(traj.arrays["gen_js"] <= np.log(2) + 1e-9)
        assert np.all(traj.arrays["gen_hellinger_sq"] >= 0)
        assert np.all(traj.arrays["gen_hellinger_sq"] <= 1 + 1e-9)
        assert np.all(traj.arrays["ref_norm_entropy"] >= 0)
        assert np.all(traj.arrays["ref_norm_entropy"] <= 1)

    def test_score_tensor_cache_consistent(self, traj):
        """Recomputing the lse term from the cached fp16 scores at beta*
        must agree with the online values (fp16 storage tolerance)."""
        from energy_llm import metrics
        beta = float(traj.arrays["beta_star"])
        s_ref = traj.score_tensors["ref_retrieval_scores"].astype(np.float64)
        np.testing.assert_allclose(
            metrics.lse_term(s_ref, beta), traj.arrays["ref_lse_term"],
            rtol=1e-2, atol=1e-2,
        )

    def test_metadata(self, traj, tiny_samples):
        assert traj.metadata["sample_id"] == tiny_samples[0].sample_id
        assert traj.metadata["generated_text"] is not None
        assert traj.metadata["n_prompt_tokens"] > 1
        assert traj.metadata["beta_star"] == 5.0

    def test_token_logprobs_are_log_probabilities(self, traj):
        assert np.all(traj.arrays["token_logprobs"] <= 0)
        assert np.all(traj.arrays["token_predictive_entropy"] >= 0)


class TestPrefillReference:
    def test_prefill_scores_match_capture_sample_reference(
        self, tiny_model, tiny_tokenizer, tiny_banks, tiny_samples
    ):
        """The prefill-only path (calibration) and the full-capture reference
        must agree: the teacher-forced pass shares the identical prompt."""
        hooks = ProbePointHooks(tiny_model, tiny_banks)
        try:
            s_prefill = capture_prefill_reference_scores(
                tiny_model, tiny_tokenizer, hooks, tiny_samples[0].question
            )
            traj = capture_sample(
                tiny_model, tiny_tokenizer, hooks, tiny_samples[0],
                beta_star=5.0, max_new_tokens=3, cache_score_tensors=True,
            )
        finally:
            hooks.remove()
        s_ref = traj.score_tensors["ref_retrieval_scores"].astype(np.float32)
        np.testing.assert_allclose(s_prefill, s_ref, rtol=1e-2, atol=1e-3)


class TestSharding:
    def test_shards_are_disjoint_and_cover(self, tiny_samples):
        from energy_llm.data import shard
        pieces = [shard(tiny_samples, k, 3) for k in range(3)]
        ids = [s.sample_id for piece in pieces for s in piece]
        assert sorted(ids) == sorted(s.sample_id for s in tiny_samples)
        assert len(ids) == len(set(ids))

    def test_bad_shard_index_rejected(self, tiny_samples):
        from energy_llm.data import shard
        with pytest.raises(ValueError):
            shard(tiny_samples, 3, 3)
