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
