import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from energy_llm.core import capture, metrics
from energy_llm.core.bank import build_bank
from energy_llm.core.capture import NonFiniteError, ResidualHooks, compute_item, forward_item
from energy_llm.core.prompting import (
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
        tr = _trace(bundle, q)
        checked += 1
        assert tr.tf_mismatch_positions == []
        # and greedy equals an explicit argmax loop without any logits processor
        ids = list(tr.prompt.input_ids)
        for t in tr.gen_ids:
            with torch.no_grad():
                nxt = bundle.model(input_ids=torch.tensor([ids])).logits[0, -1].argmax().item()
            assert nxt == t
            ids.append(t)
    assert checked == 4


def test_tf_logit_discrepancy_recorded_and_exact_answers(pair):
    bundle = make_bundle(*pair, max_new_tokens=6)
    tr = _trace(bundle)
    assert tr.tf_logit_diff.shape == (len(tr.gen_ids),) and tr.tf_logit_diff.dtype == np.float32
    assert (tr.tf_logit_diff >= 0).all() and tr.tf_logit_diff.max() < 1e-3   # float32 tiny model
    assert tr.tf_mismatch_margin_tf == tr.tf_mismatch_margin_gen == tr.tf_mismatch_tf_tokens == []


def test_tf_mismatch_margins_are_recorded(qwen_pair, monkeypatch):
    """Force the TF argmax off the greedy token at step 1: the mismatch, the TF
    token and both top1 - top2 margins are recorded."""
    bundle = make_bundle(*qwen_pair, max_new_tokens=6)
    model = bundle.model
    orig = model.forward
    bump = {}

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        if kwargs.get("use_cache") is False and out.logits.shape[1] > 2:
            row = out.logits[0, 1]
            top2 = torch.topk(row, 2)
            bump["token"] = int(top2.indices[1])
            row[top2.indices[1]] = top2.values[0] + 0.25        # runner-up wins by 0.25
        return out

    monkeypatch.setattr(model, "forward", forward)
    tr = _trace(bundle)
    if len(tr.gen_ids) < 2:
        pytest.skip("needs two generated tokens")
    assert tr.tf_mismatch_positions == [1]
    assert tr.tf_mismatch_tf_tokens == [bump["token"]]
    assert tr.tf_mismatch_margin_tf[0] == pytest.approx(0.25, abs=1e-5)
    assert tr.tf_mismatch_margin_gen[0] > 0
    assert tr.tf_mismatch_top_logit[0] > 0
    assert tr.tf_logit_diff[1] >= 0.25 - 1e-5


def test_token_logits_are_the_decoding_logits_of_the_chosen_tokens(qwen_pair):
    bundle = make_bundle(*qwen_pair, max_new_tokens=6)
    tr = _trace(bundle)
    ids = torch.tensor([tr.prompt.input_ids + tr.gen_ids])
    with torch.no_grad():
        logits = bundle.model(input_ids=ids).logits[0, tr.n_prompt - 1: -1].float()
    want = logits.gather(1, torch.tensor(tr.gen_ids)[:, None])[:, 0].numpy()
    assert tr.token_logits.dtype == np.float32
    np.testing.assert_allclose(tr.token_logits, want, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(tr.token_logprobs, want - torch.logsumexp(logits, -1).numpy(), atol=1e-4)


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


def test_empty_response_is_a_valid_sample(pair):
    bundle = make_bundle(*pair)
    bank = build_bank(bundle.model, bundle.dims)
    first = _trace(bundle).gen_ids[0]
    _set_stops(bundle, set(bundle.stop_token_ids) | {first})   # the model stops at once
    tr = _trace(bundle)
    assert tr.empty_response and tr.gen_ids == [first] and not tr.truncated
    assert tr.reference_positions["response_first"] is None
    res = compute_item(bundle, bank, tr, "e", [1.0, 5.0], list(range(bundle.dims.n_layers)), 4, 8)
    assert res.answer["empty_response"] and res.answer["n_response_tokens"] == 0
    assert res.answer["response"] == ""
    for name, arr in res.traces.items():
        if name.endswith("/response_first"):
            assert np.isnan(arr).all(), name
        else:
            assert np.isfinite(arr).all(), name
    for name in ("response_first", "response_last"):
        assert np.isnan(res.hidden[name]).all()
    for name in ("question_last", "prompt_last"):
        assert np.isfinite(res.hidden[name]).all()
    assert res.baselines["empty_response"] and np.isnan(res.baselines["mean_logprob"])


# ------------------------------------------------------------ item metrics

K_MASS, K_IDS = 4, 8


@pytest.fixture()
def item(pair):
    bundle = make_bundle(*pair, max_new_tokens=8)
    bank = build_bank(bundle.model, bundle.dims)
    tr = _trace(bundle)
    beta = 7.5
    res = compute_item(bundle, bank, tr, "x1", [1.0, 30.0, beta], list(range(bundle.dims.n_layers)),
                       K_MASS, K_IDS)
    return bundle, bank, tr, beta, res


def _scores(bundle, bank, tr, l):
    return capture._layer_scores(bank, tr.residuals[l], l, bundle.dims.rms_norm_eps)


def test_item_shapes_and_bounds(item):
    bundle, _, tr, _, res = item
    L, D = bundle.dims.n_layers, bundle.dims.d_model
    T = tr.n_prompt + len(tr.gen_ids)
    per_beta = {"energy", "norm_entropy", "topk_mass", "mean_score", "var_score"} | {
        f"{d}/{r}" for d in ("js", "dkl") for r in capture.REFERENCES}
    assert set(res.traces) == per_beta | {"margin", "xhat_top1_frac", "topk_ids"}
    for name in per_beta:
        assert res.traces[name].shape == (L, T, 3) and res.traces[name].dtype == np.float32, name
    for name in ("margin", "xhat_top1_frac"):
        assert res.traces[name].shape == (L, T)
    assert res.traces["topk_ids"].shape == (L, T, K_IDS) and res.traces["topk_ids"].dtype == np.int32
    assert res.role_mask.shape == (T,) and res.role_mask.dtype == np.int8
    assert set(res.hidden) == {"question_last", "prompt_last", "response_first", "response_last"}
    for h in res.hidden.values():
        assert h.shape == (L, D) and h.dtype == np.float16
    t = res.traces
    assert np.all((t["norm_entropy"] >= -1e-6) & (t["norm_entropy"] <= 1 + 1e-6))
    assert np.all((t["topk_mass"] > 0) & (t["topk_mass"] <= 1 + 1e-6))
    assert np.all(t["var_score"] >= 0)
    assert np.all((t["xhat_top1_frac"] >= 1 / D - 1e-6) & (t["xhat_top1_frac"] <= 1 + 1e-6))
    for r in capture.REFERENCES:
        assert np.all((t[f"js/{r}"] >= 0) & (t[f"js/{r}"] <= np.log(2)))
    assert 0.0 <= res.answer["xhat_norm_max_rel_dev"] < 0.05  # tiny residuals: eps is visible
    assert res.answer["n_response_tokens"] == len(tr.response_positions)
    assert res.answer["tf_exact"] is True and res.answer["tf_mismatch_margin_tf"] == []
    assert len(res.answer["tf_logit_diff"]) == res.answer["n_generated_tokens"]
    assert res.answer["tf_logit_diff_max"] == max(res.answer["tf_logit_diff"])
    assert res.answer["tf_logit_diff_step0"] == res.answer["tf_logit_diff"][0]
    assert "gold" not in str(res.answer)


def test_reference_positions(item):
    _, _, tr, _, res = item
    refs = res.reference_positions
    q = [i for i, r in enumerate(tr.roles) if r == ROLE_QUESTION]
    assert refs == {"question_last": q[-1], "prompt_last": tr.n_prompt - 1,
                    "response_first": tr.response_positions[0]}
    assert refs["question_last"] < refs["prompt_last"] < refs["response_first"]


def test_reference_requires_a_question_token(qwen_pair):
    tr = _trace(make_bundle(*qwen_pair))
    tr.prompt.roles = [ROLE_TEMPLATE] * tr.n_prompt
    with pytest.raises(PromptError, match="question"):
        tr.reference_positions


def test_own_reference_rows_are_zero(item):
    *_, res = item
    for r, pos in res.reference_positions.items():
        assert np.abs(res.traces[f"js/{r}"][:, pos]).max() <= 1e-7      # one rounding in logaddexp
        assert np.abs(res.traces[f"dkl/{r}"][:, pos]).max() == 0.0


def test_divergences_match_explicit_formula(item):
    bundle, bank, tr, _, res = item
    l, bi, beta = 1, 1, 30.0
    _, s = _scores(bundle, bank, tr, l)
    lp = torch.log_softmax(beta * s.double(), dim=-1)
    for r, pos in res.reference_positions.items():
        p, q = lp.exp(), lp[pos].exp()
        kl_pq = (p * (lp - lp[pos])).sum(-1)
        kl_qp = (q * (lp[pos] - lp)).sum(-1)
        np.testing.assert_allclose(res.traces[f"dkl/{r}"][l, :, bi], (kl_pq - kl_qp).numpy(), atol=1e-4)
        m = 0.5 * (p + q)
        js = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
        np.testing.assert_allclose(res.traces[f"js/{r}"][l, :, bi], js.numpy(), atol=1e-5)


def test_topk_ids_sorted_and_unique(item):
    bundle, bank, tr, _, res = item
    for l in range(bundle.dims.n_layers):
        _, s = _scores(bundle, bank, tr, l)
        for t, row in enumerate(res.traces["topk_ids"][l]):
            assert len(set(row.tolist())) == K_IDS
            vals = s[t, torch.as_tensor(row, dtype=torch.long)]
            assert (vals[:-1] >= vals[1:]).all()
            assert vals[-1] >= torch.topk(s[t], K_IDS + 1).values[-1]


def test_energy_at_beta_star_equals_single_beta_trace(item):
    """The last beta column is the former single-beta trace at beta*; stored as
    float16 it stays within one float16 rounding."""
    bundle, bank, tr, beta, res = item
    for l in range(bundle.dims.n_layers):
        _, s = _scores(bundle, bank, tr, l)
        single = metrics.energy(s, beta).numpy()
        assert np.array_equal(res.traces["energy"][l, :, -1], single)
        np.testing.assert_allclose(res.traces["energy"][l, :, -1].astype(np.float16), single,
                                   rtol=2.0 ** -11, atol=0)
    l = bundle.dims.n_layers - 1
    _, s = _scores(bundle, bank, tr, l)
    ref = -logsumexp(beta * s.double().numpy(), axis=-1) / beta
    np.testing.assert_allclose(res.traces["energy"][l, :, -1], ref, rtol=1e-5)


def test_betas_record_the_grid_with_beta_star_last(item):
    *_, beta, res = item
    assert res.betas == [1.0, 30.0, beta]


def test_non_finite_values_raise(qwen_pair, monkeypatch):
    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    tr = _trace(bundle)
    monkeypatch.setattr(capture.metrics, "energy", lambda s, b: torch.full(s.shape[:-1], float("nan")))
    with pytest.raises(NonFiniteError):
        compute_item(bundle, bank, tr, "x", [5.0], [0], K_MASS, K_IDS)
