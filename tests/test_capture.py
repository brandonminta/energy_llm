import math

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from energy_llm import capture, metrics
from energy_llm.bank import build_bank
from energy_llm.capture import (
    FEATURES, EmptyResponseError, NonFiniteError, ResidualHooks, compute_item, forward_item,
)
from energy_llm.prompting import (
    ROLE_INSTRUCTION, ROLE_QUESTION, ROLE_RESPONSE, ROLE_TEMPLATE, PromptError,
    build_prompt, check_template,
)
from tests.conftest import DATE, QUESTION_WORDS, SYSTEM, make_bundle, make_questions

Q = "what is the capital of france"


def _date(tok):
    return DATE if "date_string" in tok.chat_template else None


def _trace(bundle, question=Q):
    hooks = ResidualHooks(bundle.model)
    try:
        prompt = build_prompt(bundle.tokenizer, SYSTEM, question, _date(bundle.tokenizer))
        return forward_item(bundle, hooks, prompt)
    finally:
        hooks.remove()


def _set_stops(bundle, stops):
    bundle.stop_token_ids = sorted(stops)
    bundle.generation_config.eos_token_id = sorted(stops)


# ---------------------------------------------------------------- prompting

def test_system_prompt_explicit_and_roles(qwen_pair):
    _, tok = qwen_pair
    p = build_prompt(tok, SYSTEM, Q, None)
    assert "default" not in p.text  # the template's own system prompt is not injected
    words = tok.convert_ids_to_tokens(p.input_ids)
    by_role = {r: [w for w, rr in zip(words, p.roles) if rr == r] for r in set(p.roles)}
    assert " ".join(by_role[ROLE_INSTRUCTION]) == SYSTEM
    assert " ".join(by_role[ROLE_QUESTION]) == Q
    assert set(by_role[ROLE_TEMPLATE]) == {"system", "user", "eot", "assistant"}


def test_llama_template_requires_date_string():
    from tests.conftest import LLAMA_TEMPLATE, make_tokenizer

    tok = make_tokenizer(LLAMA_TEMPLATE)
    with pytest.raises(PromptError, match="date_string"):
        check_template(tok, None)
    p = build_prompt(tok, SYSTEM, Q, DATE)
    assert f"today {DATE} {SYSTEM}" in p.text


# --------------------------------------------------------- generation + TF

def test_hub_sampling_settings_are_overridden(pair):
    model, tok = pair
    assert model.generation_config.repetition_penalty == 1.3
    bundle = make_bundle(model, tok)
    gc = model.generation_config
    assert gc is bundle.generation_config
    assert (gc.do_sample, gc.repetition_penalty, gc.num_beams) == (False, 1.0, 1)
    assert bundle.stop_token_ids == sorted([tok.eos_token_id, tok.convert_tokens_to_ids("eot")])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_teacher_forcing_reproduces_greedy(pair, seed):
    bundle = make_bundle(*pair, max_new_tokens=10)
    checked = 0
    for q in make_questions(4, QUESTION_WORDS, seed):
        try:
            tr = _trace(bundle, q)
        except EmptyResponseError:
            continue
        checked += 1
        assert tr.tf_mismatch_positions == []
        # and greedy equals an explicit argmax loop without any logits processor
        ids = list(tr.prompt.input_ids)
        for t in tr.gen_ids:
            with torch.no_grad():
                nxt = bundle.model(input_ids=torch.tensor([ids])).logits[0, -1].argmax().item()
            assert nxt == t
            ids.append(t)
    assert checked >= 2


def test_residual_hook_reads_norm_input(qwen_pair):
    bundle = make_bundle(*qwen_pair)
    tr = _trace(bundle)
    T = tr.n_prompt + len(tr.gen_ids)
    for l, layer in enumerate(bundle.model.model.layers):
        x = tr.residuals[l]
        assert x.shape == (T, bundle.dims.d_model)
        normed = layer.post_attention_layernorm(x[None])[0]
        x_hat = metrics.rms_normalise(x, bundle.dims.rms_norm_eps)
        np.testing.assert_allclose((x_hat * layer.post_attention_layernorm.weight).numpy(),
                                   normed.numpy(), rtol=1e-5, atol=1e-6)


def test_truncation_flag_without_stop(qwen_pair):
    bundle = make_bundle(*qwen_pair, max_new_tokens=6)
    _set_stops(bundle, [bundle.tokenizer.eos_token_id])
    tr = _trace(bundle)
    if tr.stop_token_id is None:
        assert tr.truncated and len(tr.gen_ids) == 6
        assert len(tr.response_positions) == 6


def test_every_stop_token_is_excluded(pair):
    bundle = make_bundle(*pair, max_new_tokens=16)
    # declare a token the model emits (first seen at j >= 1) as an extra stop token
    for q in [Q] + make_questions(20, QUESTION_WORDS, 9):
        base = _trace(bundle, q)
        cand = [j for j, t in enumerate(base.gen_ids) if j >= 1 and t not in base.gen_ids[:j]]
        if cand:
            break
    else:
        raise AssertionError("no question with two distinct generated tokens")
    first = cand[0]
    stop = base.gen_ids[first]
    _set_stops(bundle, set(bundle.stop_token_ids) | {stop})
    tr = _trace(bundle, q)
    assert tr.gen_ids == base.gen_ids[: first + 1]
    assert tr.stop_token_id == stop and not tr.truncated
    assert tr.response_positions == [tr.n_prompt + j for j in range(first)]
    assert tr.roles[-1] == ROLE_TEMPLATE
    assert tr.roles.count(ROLE_RESPONSE) == first


def test_empty_response_is_an_error(qwen_pair):
    bundle = make_bundle(*qwen_pair)
    first = _trace(bundle).gen_ids[0]
    _set_stops(bundle, set(bundle.stop_token_ids) | {first})
    with pytest.raises(EmptyResponseError):
        _trace(bundle)


# ------------------------------------------------------------ item metrics

@pytest.fixture()
def item(pair):
    bundle = make_bundle(*pair, max_new_tokens=8)
    bank = build_bank(bundle.model, bundle.dims)
    tr = _trace(bundle)
    beta = 7.5
    res = compute_item(bundle, bank, tr, "x1", beta, [0.5, 1.0, 2.0], 8,
                       list(range(bundle.dims.n_layers)))
    return bundle, bank, tr, beta, res


def test_item_shapes_and_bounds(item):
    bundle, _, tr, _, res = item
    L, D = bundle.dims.n_layers, bundle.dims.d_model
    T = tr.n_prompt + len(tr.gen_ids)
    assert res.features.shape == (L, 3, 2, len(FEATURES))
    assert res.features.dtype == np.float32
    for name in ("energy", "norm_entropy", "margin"):
        assert getattr(res, name).shape == (L, T)
    assert res.role_mask.shape == (T,) and res.role_mask.dtype == np.int8
    assert res.hidden_prompt_last.shape == (L, D) and res.hidden_prompt_last.dtype == np.float16
    f = {k: res.features[..., i] for i, k in enumerate(FEATURES)}
    assert np.all((f["js"] >= 0) & (f["js"] <= math.log(2)))
    assert np.all((f["hellinger_sq"] >= 0) & (f["hellinger_sq"] <= 1 + 1e-6))
    assert np.all((f["jaccard_topk"] >= 0) & (f["jaccard_topk"] <= 1))
    for k in ("norm_entropy_prompt", "norm_entropy_resp"):
        assert np.all((f[k] >= -1e-6) & (f[k] <= 1 + 1e-6))
    np.testing.assert_allclose(f["delta_energy"], f["energy_resp"] - f["energy_prompt"], atol=1e-5)
    assert np.all(res.norm_entropy >= -1e-6) and np.all(res.norm_entropy <= 1 + 1e-6)
    assert 0.0 <= res.answer["xhat_norm_max_rel_dev"] < 0.05  # tiny residuals: eps is visible
    assert res.answer["n_response_tokens"] == len(tr.response_positions)
    assert "gold" not in str(res.answer)


def test_phases_and_variants_match_traces(item):
    bundle, bank, tr, beta, res = item
    f = {k: res.features[:, 1, :, i] for i, k in enumerate(FEATURES)}  # multiplier 1.0
    pi, R = tr.n_prompt - 1, tr.response_positions
    np.testing.assert_allclose(f["energy_prompt"][:, 0], res.energy[:, pi], rtol=1e-5)
    np.testing.assert_allclose(f["energy_resp"][:, 0], res.energy[:, R].mean(1), rtol=1e-5)
    np.testing.assert_allclose(f["energy_resp"][:, 1], res.energy[:, R[0]], rtol=1e-5)
    np.testing.assert_allclose(f["norm_entropy_resp"][:, 0], res.norm_entropy[:, R].mean(1), rtol=1e-5)
    np.testing.assert_allclose(f["margin_resp"][:, 1], res.margin[:, R[0]], rtol=1e-6)
    # JS of the mean distribution, not mean of per-token JS
    l = 0
    _, s = capture._layer_scores(bank, tr.residuals[l], l, bundle.dims.rms_norm_eps)
    p = torch.softmax(beta * s, dim=-1)
    js = metrics.js_divergence(p[R].mean(0), p[pi]).item()
    assert f["js"][l, 0] == pytest.approx(js, rel=1e-5)


def test_energy_trace_matches_scipy(item):
    bundle, bank, tr, beta, res = item
    l = bundle.dims.n_layers - 1
    _, s = capture._layer_scores(bank, tr.residuals[l], l, bundle.dims.rms_norm_eps)
    ref = -logsumexp(beta * s.double().numpy(), axis=-1) / beta
    np.testing.assert_allclose(res.energy[l], ref, rtol=1e-5)


def test_betas_scale_beta_star(item):
    *_, beta, res = item
    assert res.betas == [beta * 0.5, beta, beta * 2.0]


def test_non_finite_values_raise(qwen_pair, monkeypatch):
    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    tr = _trace(bundle)
    monkeypatch.setattr(capture.metrics, "energy", lambda s, b: torch.full(s.shape[:-1], float("nan")))
    with pytest.raises(NonFiniteError):
        compute_item(bundle, bank, tr, "x", 5.0, [1.0], 8, [0])
