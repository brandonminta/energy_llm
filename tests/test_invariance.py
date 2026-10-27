"""Invariances and limits on the tiny models (verify runs the same on the real ones)."""

import math

import numpy as np
import pytest
import torch

from energy_llm import metrics
from energy_llm.bank import build_bank
from energy_llm.capture import ResidualHooks, compute_item, forward_item
from energy_llm.prompting import build_prompt
from energy_llm.verify import PermutedBank
from tests.conftest import DATE, SYSTEM, make_bundle

Q = "what is the capital of france"


@pytest.fixture()
def setup(pair):
    bundle = make_bundle(*pair)
    bank = build_bank(bundle.model, bundle.dims)
    hooks = ResidualHooks(bundle.model)
    date = DATE if "date_string" in bundle.tokenizer.chat_template else None
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, Q, date))
    hooks.remove()
    return bundle, bank, tr


def _scores(bundle, bank, tr, l):
    return bank.scores(l, metrics.rms_normalise(tr.residuals[l], bundle.dims.rms_norm_eps))


def test_permuting_memories_changes_nothing(setup):
    bundle, bank, tr = setup
    L = list(range(bundle.dims.n_layers))
    a = compute_item(bundle, bank, tr, "x", 9.0, [0.5, 1.0, 2.0], 8, L)
    pbank = PermutedBank(bank, seed=3)
    assert any((p != torch.arange(len(p))).any() for p in pbank.perm)
    s, sp = _scores(bundle, bank, tr, 0), _scores(bundle, pbank, tr, 0)
    np.testing.assert_allclose(sp.numpy(), s[:, pbank.perm[0]].numpy(), atol=1e-6)
    b = compute_item(bundle, pbank, tr, "x", 9.0, [0.5, 1.0, 2.0], 8, L)
    for name in ("features", "energy", "norm_entropy", "margin"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), atol=1e-5, rtol=0)


def test_beta_limits(setup):
    bundle, bank, tr = setup
    for l in range(bundle.dims.n_layers):
        s = _scores(bundle, bank, tr, l)
        for t in range(s.shape[0]):
            st = s[t]
            assert metrics.norm_entropy(metrics.log_distribution(st, 1e-8)).item() >= 1 - 1e-5
            m = metrics.margin(st).item()
            if m <= 0:
                continue
            b_inf = 50.0 / m
            assert metrics.norm_entropy(metrics.log_distribution(st, b_inf)).item() <= 1e-5
            gap = abs(metrics.energy(st, b_inf).item() + st.max().item())
            assert gap <= math.log(st.shape[0]) / b_inf + 1e-6


def test_norm_entropy_monotone_in_beta(setup):
    bundle, bank, tr = setup
    grid = [10 ** (x / 10) for x in range(-20, 41)]
    for l in range(bundle.dims.n_layers):
        s = _scores(bundle, bank, tr, l)
        hs = torch.stack([metrics.norm_entropy(metrics.log_distribution(s, b)) for b in grid])
        assert (hs[1:] - hs[:-1]).max().item() <= 1e-6


def test_self_divergence_is_zero(setup):
    bundle, bank, tr = setup
    for l in range(bundle.dims.n_layers):
        lp = metrics.log_distribution(_scores(bundle, bank, tr, l), 9.0)
        # logaddexp(a, a) - log 2 equals a up to one float32 rounding
        assert metrics.js_divergence(lp, lp).abs().max().item() <= 1e-7
        assert metrics.hellinger_sq(lp, lp).abs().max().item() == 0.0


def test_two_identical_runs_are_bit_identical_on_cpu(pair):
    outs = []
    for _ in range(2):
        bundle = make_bundle(*pair)
        bank = build_bank(bundle.model, bundle.dims)
        hooks = ResidualHooks(bundle.model)
        date = DATE if "date_string" in bundle.tokenizer.chat_template else None
        tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, Q, date))
        hooks.remove()
        outs.append((tr.gen_ids, compute_item(bundle, bank, tr, "x", 9.0, [0.5, 1.0, 2.0], 8,
                                               list(range(bundle.dims.n_layers)))))
    assert outs[0][0] == outs[1][0]
    for name in ("features", "energy", "norm_entropy", "margin", "hidden_prompt_last"):
        assert np.array_equal(getattr(outs[0][1], name), getattr(outs[1][1], name), equal_nan=True)
