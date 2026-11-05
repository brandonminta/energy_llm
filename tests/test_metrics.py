import math

import numpy as np
import pytest
import torch
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp

from energy_llm.core import metrics

K = 512


def _dist(seed, beta=5.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(K, generator=g) * 2 - 1
    return torch.softmax(beta * s, dim=-1), s


@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e3])
def test_xhat_norm(scale):
    """||x_hat||^2 = D * ms / (ms + eps): ~ sqrt(D) unless the residual RMS
    approaches sqrt(eps), which is why extraction records the deviation."""
    d, eps = 2048, 1e-6
    x = torch.randn(16, d) * scale
    x_hat = metrics.rms_normalise(x, eps=eps)
    assert x_hat.dtype == torch.float32
    ms = x.double().pow(2).mean(-1)
    exact = torch.sqrt(d * ms / (ms + eps)).numpy()
    np.testing.assert_allclose(x_hat.norm(dim=-1).numpy(), exact, rtol=1e-5)
    if scale >= 1.0:
        np.testing.assert_allclose(x_hat.norm(dim=-1).numpy(), math.sqrt(d), rtol=1e-5)


@pytest.mark.parametrize("beta", [1e-3, 1.0, 50.0, 1e4])
def test_distribution_sums_to_one_and_entropy_bounded(beta):
    _, s = _dist(0)
    logp = metrics.log_distribution(s, beta)
    assert abs(logp.exp().sum().item() - 1.0) < 1e-5
    h = metrics.norm_entropy(logp).item()
    assert -1e-6 <= h <= 1.0 + 1e-6


def test_entropy_extremes():
    s = torch.zeros(K)
    assert metrics.norm_entropy(metrics.log_distribution(s, 1.0)).item() == pytest.approx(1.0, abs=1e-6)
    s[3] = 1.0
    assert metrics.norm_entropy(metrics.log_distribution(s, 1e4)).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("beta", [0.5, 20.0, 3e3])
def test_energy_matches_scipy_logsumexp(beta):
    _, s = _dist(1)
    ref = -logsumexp(beta * s.double().numpy()) / beta
    assert metrics.energy(s, beta).item() == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_js_symmetric_bounded_and_matches_scipy_convention():
    p, _ = _dist(2)
    q, _ = _dist(3)
    js_pq = metrics.js_divergence(p.log(), q.log()).item()
    assert js_pq == pytest.approx(metrics.js_divergence(q.log(), p.log()).item(), abs=1e-7)
    assert 0.0 <= js_pq <= math.log(2)
    ref = jensenshannon(p.double().numpy(), q.double().numpy(), base=2) ** 2 * math.log(2)
    assert js_pq == pytest.approx(ref, rel=1e-4)
    assert metrics.js_divergence(p.log(), p.log()).item() == pytest.approx(0.0, abs=1e-6)


def test_js_reaches_log2_on_disjoint_supports():
    s_p = torch.full((K,), -1.0); s_p[:4] = 1.0
    s_q = torch.full((K,), -1.0); s_q[10:12] = 1.0
    lp, lq = metrics.log_distribution(s_p, 200.0), metrics.log_distribution(s_q, 200.0)
    assert metrics.js_divergence(lp, lq).item() == pytest.approx(math.log(2), abs=1e-6)


def test_divergences_survive_float32_underflow():
    """Regression: near-one-hot distributions (large beta) whose tails underflow
    in float32 must still give the float64 values (found by dev/reference.py)."""
    g = torch.Generator().manual_seed(7)
    s = torch.rand(2, K, generator=g) * 0.4 - 0.2
    s[1, 5] = s[0, 5] + 0.01        # shared near-top memory with different weights
    for beta in (600.0, 5e3):
        lp = metrics.log_distribution(s, beta)
        l64 = torch.log_softmax(beta * s.double(), dim=-1)
        p64, q64 = l64.exp()
        m = 0.5 * (p64 + q64)
        kl = lambda a: torch.where(a > 0, a * (a.log() - m.log()), torch.zeros_like(a)).sum()
        js64 = (0.5 * kl(p64) + 0.5 * kl(q64)).item()
        assert metrics.js_divergence(lp[0], lp[1]).item() == pytest.approx(js64, abs=1e-5)


def test_margin():
    s = torch.tensor([0.1, 0.9, 0.4, 0.7])
    assert metrics.margin(s).item() == pytest.approx(0.2)


def test_kl_zero_on_identical_and_dkl_antisymmetric():
    lp = metrics.log_distribution(_dist(6)[1], 30.0)
    lq = metrics.log_distribution(_dist(7)[1], 30.0)
    assert metrics.kl_divergence(lp, lp).item() == 0.0
    kl_pq, kl_qp = metrics.kl_divergence(lp, lq), metrics.kl_divergence(lq, lp)
    assert kl_pq.item() > 0 and kl_qp.item() > 0
    assert (kl_pq - kl_qp).item() == -(kl_qp - kl_pq).item()
    ref = np.sum(lp.exp().double().numpy() * (lp.double().numpy() - lq.double().numpy()))
    assert kl_pq.item() == pytest.approx(ref, rel=1e-4)


def test_var_score_limits():
    _, s = _dist(8)
    lp0 = metrics.log_distribution(s, 1e-8)
    assert metrics.mean_score(lp0, s).item() == pytest.approx(s.mean().item(), abs=1e-6)
    assert metrics.var_score(lp0, s).item() == pytest.approx(np.var(s.double().numpy()), rel=1e-4)
    b_inf = 50.0 / metrics.margin(s).item()
    lp_inf = metrics.log_distribution(s, b_inf)
    assert metrics.var_score(lp_inf, s).item() <= 1e-6
    assert metrics.mean_score(lp_inf, s).item() == pytest.approx(s.max().item(), abs=1e-6)


def test_topk_mass_limits():
    _, s = _dist(9)
    b_inf = 50.0 / metrics.margin(s).item()
    for k in (1, 10):
        assert metrics.topk_mass(metrics.log_distribution(s, b_inf), k).item() == pytest.approx(1.0, abs=1e-6)
        assert metrics.topk_mass(metrics.log_distribution(s, 1e-8), k).item() == pytest.approx(k / K, rel=1e-4)


def test_topk_indices_sorted_unique_int32():
    lp = metrics.log_distribution(torch.randn(4, K), 20.0)
    idx = metrics.topk_indices(lp, 100)
    assert idx.dtype == torch.int32 and idx.shape == (4, 100)
    for row, l in zip(idx, lp):
        assert len(set(row.tolist())) == 100
        vals = l[row.long()]
        assert (vals[:-1] >= vals[1:]).all()
        assert vals[-1] >= torch.topk(l, 101).values[-1]


def test_xhat_top1_frac_extremes():
    d = 64
    one_hot = torch.zeros(d); one_hot[5] = 3.0
    assert metrics.xhat_top1_frac(metrics.rms_normalise(one_hot, 1e-12)).item() == pytest.approx(1.0, rel=1e-6)
    const = metrics.rms_normalise(torch.full((d,), 2.0), 1e-12)
    assert metrics.xhat_top1_frac(const).item() == pytest.approx(1.0 / d, rel=1e-6)


def test_primitives_match_reference():
    from energy_llm.verification.verify import load_reference

    ref = load_reference()
    _, s = _dist(10)
    _, s2 = _dist(11)
    x = torch.randn(K)
    for beta in (1.0, 40.0, 600.0):
        lp, lq = metrics.log_distribution(s, beta), metrics.log_distribution(s2, beta)
        sl, s2l = [float(v) for v in s.double()], [float(v) for v in s2.double()]
        p, lse = ref.softmax_and_lse(sl, beta)
        _, lse2 = ref.softmax_and_lse(s2l, beta)
        logp = [beta * v - lse for v in sl]
        logq = [beta * v - lse2 for v in s2l]
        assert metrics.mean_score(lp, s).item() == pytest.approx(ref.mean_score(p, sl), abs=1e-5)
        assert metrics.var_score(lp, s).item() == pytest.approx(ref.var_score(p, sl), abs=1e-5)
        assert metrics.topk_mass(lp, 10).item() == pytest.approx(ref.topk_mass(p, 10), abs=1e-5)
        assert metrics.topk_indices(lp, 10).tolist() == ref.topk_indices(p, 10)
        assert metrics.kl_divergence(lp, lq).item() == pytest.approx(ref.kl(logp, logq), rel=1e-4, abs=1e-5)
    xh = metrics.rms_normalise(x, 1e-6)
    assert metrics.xhat_top1_frac(xh).item() == pytest.approx(
        ref.xhat_top1_frac(ref.x_hat([float(v) for v in x.double()], 1e-6)), rel=1e-5)
