"""Invariances and limits on the tiny models (verify runs the same on the real ones)."""

import math

import numpy as np
import pytest
import torch

from energy_llm.core import metrics
from energy_llm.core.bank import build_bank
from energy_llm.core.capture import ResidualHooks, compute_item, forward_item
from energy_llm.core.prompting import build_prompt
from energy_llm.verification.verify import (PermutedBank, gamma_shuffled_bank, permutation_tolerances,
                                             trace_ratios)
from energy_llm.core.bank import VARIANTS
from energy_llm.verification.verify import view
from tests.conftest import DATE, SYSTEM, both, make_bundle

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


def _scores(bundle, bank, tr, l, v):
    return bank.scores(l, metrics.rms_normalise(tr.residuals[l], bundle.dims.rms_norm_eps), v)


def test_permuting_memories_changes_nothing(setup):
    bundle, bank, tr = setup
    L = list(range(bundle.dims.n_layers))
    a = compute_item(bundle, bank, tr, "x", both([2.0, 9.0]), L, 4, both(8))
    pbank = PermutedBank(bank, seed=3)
    assert any((p != torch.arange(len(p))).any() for p in pbank.perm)
    for v in VARIANTS:
        s, sp = _scores(bundle, bank, tr, 0, v), _scores(bundle, pbank, tr, 0, v)
        np.testing.assert_allclose(sp.numpy(), s[:, pbank.perm[0]].numpy(), atol=1e-6)
    b = compute_item(bundle, pbank, tr, "x", both([2.0, 9.0]), L, 4, both(8))
    for name, arr in a.traces.items():
        if name.endswith("topk_ids"):     # memory indices move with the permutation
            np.testing.assert_array_equal(pbank.perm[0][b.traces[name][0]].numpy(), arr[0])
        else:
            np.testing.assert_allclose(b.traces[name], arr, atol=1e-5, rtol=0, err_msg=name)


@pytest.mark.parametrize("v", VARIANTS)
def test_permutation_within_derived_tolerances_and_gamma_control_fails(setup, v):
    """Per-trace tolerances from float32 reduction bounds: the true permutation
    stays within them (and not by orders of magnitude for the noisy traces);
    shuffling gamma without permuting W_g must exceed them."""
    bundle, bank, tr = setup
    L = list(range(bundle.dims.n_layers))
    betas = [1.0, 31.6, 72.776, 1e4]
    a = compute_item(bundle, bank, tr, "x", both(betas), L, 4, both(8))
    pbank = PermutedBank(bank, seed=3)
    b = compute_item(bundle, pbank, tr, "x", both(betas), L, 4, both(8))
    av, bv = view(a.traces, v), view(b.traces, v)
    tol = permutation_tolerances(bundle, bank, pbank, tr, av, betas, 4, v)
    assert set(tol) == set(av) - {"topk_ids"}
    r = trace_ratios(bv, av, tol, betas)
    assert 0 < r["worst"]["ratio"] <= 1.0, r["worst"]
    assert r["worst"]["beta"] in betas
    assert r["traces"]["dkl/question_last"]["max_ratio"] > 1e-3      # not loose by orders
    ctrl = compute_item(bundle, gamma_shuffled_bank(bank), tr, "x", both(betas), L, 4, both(8))
    assert trace_ratios(view(ctrl.traces, v), av, tol, betas)["worst"]["ratio"] > 1e3


@pytest.mark.parametrize("v", VARIANTS)
def test_beta_limits(setup, v):
    bundle, bank, tr = setup
    for l in range(bundle.dims.n_layers):
        s = _scores(bundle, bank, tr, l, v)
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


@pytest.mark.parametrize("v", VARIANTS)
def test_norm_entropy_monotone_in_beta(setup, v):
    bundle, bank, tr = setup
    grid = [10 ** (x / 10) for x in range(-20, 41)]
    for l in range(bundle.dims.n_layers):
        s = _scores(bundle, bank, tr, l, v)
        hs = torch.stack([metrics.norm_entropy(metrics.log_distribution(s, b)) for b in grid])
        assert (hs[1:] - hs[:-1]).max().item() <= 1e-6


@pytest.mark.parametrize("v", VARIANTS)
def test_self_divergence_is_zero(setup, v):
    bundle, bank, tr = setup
    for l in range(bundle.dims.n_layers):
        lp = metrics.log_distribution(_scores(bundle, bank, tr, l, v), 9.0)
        # pointwise JS and KL from one log_softmax: exactly zero
        assert (metrics.js_divergence(lp, lp) == 0).all()
        assert (metrics.kl_divergence(lp, lp) == 0).all()


def test_two_identical_runs_are_bit_identical_on_cpu(pair):
    outs = []
    for _ in range(2):
        bundle = make_bundle(*pair)
        bank = build_bank(bundle.model, bundle.dims)
        hooks = ResidualHooks(bundle.model)
        date = DATE if "date_string" in bundle.tokenizer.chat_template else None
        tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, Q, date))
        hooks.remove()
        outs.append((tr.gen_ids, compute_item(bundle, bank, tr, "x", both([2.0, 9.0]),
                                               list(range(bundle.dims.n_layers)), 4, both(8))))
    assert outs[0][0] == outs[1][0]
    a, b = outs[0][1], outs[1][1]
    for name in a.traces:
        assert np.array_equal(a.traces[name], b.traces[name], equal_nan=True), name
    for name in a.hidden:
        assert np.array_equal(a.hidden[name], b.hidden[name], equal_nan=True), name
